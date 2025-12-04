
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PGM Course Project: CNN–CRF for Vertebrae Segmentation
======================================================

This script implements the full pipeline aligned with the proposal:

1) U-Net baseline trained on your *Processed* images and corresponding *Mask*s.
2) DenseCRF post-processing on top of U-Net logits (using pydensecrf if available).
3) CRF-as-RNN: unrolled mean-field layer trained end-to-end with the CNN
   (a practical ConvCRF-style approximation with spatial + bilateral filtering).
4) Evaluation metrics: Dice, IoU, Precision/Recall, HD95, ASSD, Boundary-F1.
5) Reproducible train/val/test split and a simple CLI.

Usage examples
--------------
# Train a U-Net backbone (baseline)
python pgm_spine_crf_project.py --data_root <PATH_TO_DATA> --out_dir outputs --mode train_unet

# Evaluate the trained U-Net and save predictions
python pgm_spine_crf_project.py --data_root <PATH_TO_DATA> --out_dir outputs --mode eval_unet --ckpt outputs/unet_best.pth

# Apply DenseCRF post-processing to saved logits (or on-the-fly) and evaluate
python pgm_spine_crf_project.py --data_root <PATH_TO_DATA> --out_dir outputs --mode crf_post --ckpt outputs/unet_best.pth

# Train end-to-end U-Net + CRF-as-RNN head
python pgm_spine_crf_project.py --data_root <PATH_TO_DATA> --out_dir outputs --mode train_crfrnn

# Evaluate the end-to-end CRF-as-RNN model
python pgm_spine_crf_project.py --data_root <PATH_TO_DATA> --out_dir outputs --mode eval_crfrnn --ckpt outputs/unet_crfrnn_best.pth

Notes
-----
- The code expects the following folder structure under --data_root:

    data_root/
      Original/   (unused for training, provided for completeness)
      Processed/  (inputs)
      Mask/       (binary masks, with suffix "_template" in filenames)

- Each Processed image `X.jpg` is paired with mask `X_template.jpg` in Mask/.
- Images and masks are treated as 2D grayscale; everything is internally resized to --size (default: 512).
- DenseCRF requires `pydensecrf`. If it's missing, the script will fall back to a fast edge-aware
  bilateral approximation to remain fully runnable.

Author: Mostafa Karami & Yeonsoo Chung (with assistance for scaffolding)
"""

import argparse
import os
import sys
import math
import time
import random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image

# Optional OpenCV (for robust IO/filters); code falls back to PIL if cv2 is missing.
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Metrics helpers
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.ndimage import generate_binary_structure
from scipy.special import expit

# For boundary F1 (tolerant matching)
try:
    from skimage.segmentation import find_boundaries
    from skimage.morphology import dilation, disk
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def imread_gray(path: str) -> np.ndarray:
    """
    Read image as float32 in [0, 1], grayscale.
    """
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = img.astype(np.float32) / 255.0
    else:
        img = Image.open(path).convert('L')
        img = np.asarray(img, dtype=np.float32) / 255.0
    return img


def imwrite_gray(path: str, arr: np.ndarray):
    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if _HAS_CV2:
        cv2.imwrite(path, arr_u8)
    else:
        Image.fromarray(arr_u8).save(path)


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    H, W = size
    if _HAS_CV2:
        return cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        pil = Image.fromarray((image * 255).astype(np.uint8))
        return np.asarray(pil.resize((W, H), resample=Image.BILINEAR), dtype=np.float32) / 255.0


def resize_nearest(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    H, W = size
    if _HAS_CV2:
        return cv2.resize(image, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        pil = Image.fromarray(image.astype(np.uint8) * 255)
        return (np.asarray(pil.resize((W, H), resample=Image.NEAREST)) > 127).astype(np.uint8)


def threshold_mask(mask: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (mask >= thr).astype(np.uint8)


def list_images(folder: str, exts=('.png', '.jpg', '.jpeg', '.tif', '.bmp')) -> List[str]:
    files = []
    for e in exts:
        files.extend(sorted([str(p) for p in Path(folder).glob(f'*{e}') ]))
    return files


def split_train_val_test(files: List[str], ratios=(0.7, 0.15, 0.15), seed=1337):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    ids = np.arange(len(files))
    rng.shuffle(ids)
    n_train = int(len(files) * ratios[0])
    n_val = int(len(files) * ratios[1])
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train+n_val]
    test_ids = ids[n_train+n_val:]
    files = np.array(files)
    return files[train_ids].tolist(), files[val_ids].tolist(), files[test_ids].tolist()


def paired_mask_path(processed_path: str, data_root: str) -> str:
    # Processed/NAME.jpg  -> Mask/NAME_template.jpg
    base = os.path.basename(processed_path)
    name, ext = os.path.splitext(base)
    mask_name = name + "_template" + ext
    return os.path.join(data_root, "Mask", mask_name)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class SpineProcessedDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 file_list: List[str],
                 size: Tuple[int, int] = (512, 512),
                 augment: bool = False):
        super().__init__()
        self.data_root = data_root
        self.files = file_list
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _augment(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Simple geometric + intensity augmentation
        if random.random() < 0.5:
            img = np.ascontiguousarray(np.flip(img, axis=1))
            msk = np.ascontiguousarray(np.flip(msk, axis=1))
        if random.random() < 0.5:
            img = np.ascontiguousarray(np.flip(img, axis=0))
            msk = np.ascontiguousarray(np.flip(msk, axis=0))
        # small rotations
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)  # mild rotation
            if _HAS_CV2:
                H, W = img.shape
                M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                msk = cv2.warpAffine(msk.astype(np.float32), M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0.5
            else:
                pil = Image.fromarray((img*255).astype(np.uint8))
                pil_m = pil.rotate(angle, resample=Image.BILINEAR)
                img = np.asarray(pil_m, dtype=np.float32) / 255.0
                pil_msk = Image.fromarray((msk*255).astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
                msk = (np.asarray(pil_msk) > 127).astype(np.uint8)
        # mild brightness/contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-0.05, 0.05)
            img = np.clip(alpha * img + beta, 0.0, 1.0)
        return img, msk

    def __getitem__(self, idx):
        proc_path = self.files[idx]
        img = imread_gray(proc_path)
        mask_path = paired_mask_path(proc_path, self.data_root)
        msk = imread_gray(mask_path)  # 0/1 mask but read as [0,1]
        msk = (msk >= 0.5).astype(np.uint8)

        # resize
        img = resize(img, self.size)
        msk = resize_nearest(msk, self.size).astype(np.uint8)

        if self.augment:
            img, msk = self._augment(img, msk)

        # Normalize to zero mean, unit variance (per-image)
        mean = img.mean()
        std = img.std() + 1e-6
        img = (img - mean) / std

        # to tensors
        img_t = torch.from_numpy(img).float().unsqueeze(0)  # (1,H,W)
        msk_t = torch.from_numpy(msk).float().unsqueeze(0)  # (1,H,W)
        return {
            "image": img_t,
            "mask": msk_t,
            "path": proc_path,
        }


# --------------------------------------------------------------------------------------
# Model: U-Net backbone (small-ish)
# --------------------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*8))

        self.up1 = nn.ConvTranspose2d(base*8, base*8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base*16, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base*8, base*2)

        self.up3 = nn.ConvTranspose2d(base*2, base*2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base*4, base)

        self.up4 = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base*2, base)

        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        return logits


# --------------------------------------------------------------------------------------
# CRF-as-RNN: mean-field layer (ConvCRF-style approximation)
# --------------------------------------------------------------------------------------

def gaussian_kernel_2d(ks: int, sigma: float, device=None, dtype=torch.float32):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


class CRFMeanField(nn.Module):
    """
    A differentiable mean-field CRF layer with:
      - spatial Gaussian filtering (fixed-size conv with learnable weight)
      - bilateral edge-aware filtering (local dynamic filter, 7x7 by default)
      - learnable label compatibility transform (2x2 for binary segmentation)

    This is a *practical* approximation of DenseCRF-as-RNN suitable for PyTorch training.
    """
    def __init__(self,
                 n_classes: int = 2,
                 n_iters: int = 5,
                 spatial_ks: int = 7,
                 spatial_sigma: float = 3.0,
                 bilateral_ks: int = 7,
                 bilateral_sigma_spatial: float = 3.0,
                 bilateral_sigma_color: float = 0.1):
        super().__init__()
        assert n_classes >= 2
        self.n_classes = n_classes
        self.n_iters = n_iters
        self.spatial_ks = spatial_ks
        self.bilateral_ks = bilateral_ks

        # Fixed normalized Gaussian conv for spatial kernel (depthwise)
        kernel = gaussian_kernel_2d(spatial_ks, spatial_sigma)
        self.register_buffer('spatial_kernel', kernel[None, None, :, :])  # (1,1,k,k)

        # Learnable mixing weights for the two kernels
        self.w_spatial = nn.Parameter(torch.tensor(3.0))     # compat weight (log-space friendly)
        self.w_bilateral = nn.Parameter(torch.tensor(5.0))   # compat weight

        # Compatibility transform (Potts-like); initialized to encourage different labels to repel
        comp_init = torch.tensor([[0.0, 1.0],
                                  [1.0, 0.0]], dtype=torch.float32)  # off-diagonal encourages separation
        self.compat = nn.Parameter(comp_init)

        # Bilateral filter parameters
        self.bilateral_sigma_spatial = bilateral_sigma_spatial
        self.bilateral_sigma_color = bilateral_sigma_color

    def spatial_filter(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B,C,H,W) -> depthwise conv per channel
        B, C, H, W = q.shape
        kernel = self.spatial_kernel.expand(C, 1, self.spatial_ks, self.spatial_ks)  # (C,1,k,k)
        return F.conv2d(q, kernel, padding=self.spatial_ks // 2, groups=C)

    def bilateral_filter(self, q: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Local edge-aware bilateral filter implemented with unfold for practicality.
        img: (B,1,H,W) or (B,3,H,W) in [0,1], used as guidance.
        """
        B, C, H, W = q.shape
        k = self.bilateral_ks
        pad = k // 2

        # Feature patches around each location
        # Image patches for color distance
        img_patches = F.unfold(img, kernel_size=k, padding=pad)  # (B, F*k*k, H*W)
        Fch = img.shape[1]
        img_patches = img_patches.view(B, Fch, k*k, H*W)  # (B, F, K, HW)
        img_center = img.view(B, Fch, 1, H*W)  # (B, F, 1, HW)
        color_diff2 = (img_patches - img_center).pow(2).sum(dim=1)  # (B, K, HW)

        # Spatial Gaussian weights (constant within the window)
        device = q.device
        dtype = q.dtype
        ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        spatial = torch.exp(-(xx**2 + yy**2) / (2 * (self.bilateral_sigma_spatial**2))).reshape(1, k*k, 1)  # (1,K,1)

        # Color kernel
        color = torch.exp(-color_diff2 / (2 * (self.bilateral_sigma_color**2)))  # (B, K, HW)

        Wts = spatial * color  # (B, K, HW)
        Wts = Wts / (Wts.sum(dim=1, keepdim=True) + 1e-8)

        # Now filter q per-class
        out = []
        for c in range(C):
            qc = q[:, c:c+1, :, :]  # (B,1,H,W)
            qpatch = F.unfold(qc, kernel_size=k, padding=pad)  # (B, K, HW)
            filtered = (Wts * qpatch).sum(dim=1)  # (B, HW)
            out.append(filtered.view(B, 1, H, W))
        return torch.cat(out, dim=1)  # (B,C,H,W)

    def forward(self, unary_logits: torch.Tensor, img_guidance: torch.Tensor) -> torch.Tensor:
        """
        unary_logits: (B, C, H, W), raw CNN logits for each class
        img_guidance: (B, 1, H, W), normalized image in [0,1] for bilateral filter guidance
        Returns refined marginal logits (pre-softmax). During training, apply with CE/Dice losses.
        """
        B, C, H, W = unary_logits.shape
        assert C == self.n_classes, "Mismatch in #classes"

        # Initialize Q with softmax of unary (negative energy)
        Q = F.softmax(unary_logits, dim=1)

        for _ in range(self.n_iters):
            # Message passing
            spatial_out = self.spatial_filter(Q)
            bilateral_out = self.bilateral_filter(Q, img_guidance)

            # Weighted sum of pairwise messages
            pairwise = torch.exp(self.w_spatial) * spatial_out + torch.exp(self.w_bilateral) * bilateral_out  # positivity via exp

            # Compatibility transform: reshape to (B, C, H*W) -> matmul with (C,C)
            pairwise_flat = pairwise.view(B, C, -1)  # (B,C,N)
            compat = self.compat  # (C,C)
            msg = torch.matmul(compat, pairwise_flat)  # (C,C) * (B,C,N) -> (B,C,N)
            msg = msg.view(B, C, H, W)

            # Update
            new_unary = unary_logits - msg  # subtract pairwise (as in CRF mean-field)
            Q = F.softmax(new_unary, dim=1)

        # Return logits corresponding to the last update (inverse softmax via log)
        eps = 1e-10
        return torch.log(Q + eps)  # log-probabilities consistent with logits scale


class UNetWithCRF(nn.Module):
    def __init__(self, in_ch=1, base=32, n_iters=5):
        super().__init__()
        self.backbone = UNet(in_ch=in_ch, out_ch=2, base=base)
        self.crf = CRFMeanField(n_classes=2, n_iters=n_iters,
                                spatial_ks=7, spatial_sigma=3.0,
                                bilateral_ks=7, bilateral_sigma_spatial=3.0,
                                bilateral_sigma_color=0.1)

    def forward(self, x_raw):
        # x_raw in standardized space (mean/std); guidance expects [0,1]
        logits = self.backbone(x_raw)  # (B,2,H,W)
        # guidance image: map input back to [0,1] roughly by sigmoid on z-score;
        # alternatively provide raw image via dataloader; here we approximate:
        x_guidance = torch.clamp((x_raw - x_raw.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
                                 / (x_raw.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] -
                                    x_raw.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] + 1e-6), 0, 1)
        crf_logprobs = self.crf(logits, x_guidance)
        return crf_logprobs  # log probabilities


# --------------------------------------------------------------------------------------
# Losses & Metrics
# --------------------------------------------------------------------------------------

def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return inter / union


def precision_recall(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return float(prec), float(rec)


def _surface_distances(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distances from boundary of a to boundary of b and vice versa.
    """
    a = a.astype(bool)
    b = b.astype(bool)
    if a.sum() == 0 and b.sum() == 0:
        return np.array([0.0]), np.array([0.0])

    # surface extraction via binary erosion
    struct = generate_binary_structure(2, 1)
    a_er = binary_erosion(a, structure=struct)
    b_er = binary_erosion(b, structure=struct)
    a_surface = np.logical_xor(a, a_er)
    b_surface = np.logical_xor(b, b_er)

    if a_surface.sum() == 0:
        a_surface = a
    if b_surface.sum() == 0:
        b_surface = b

    # distance transform on the complement of surfaces
    dt_b = distance_transform_edt(~b_surface)
    dt_a = distance_transform_edt(~a_surface)
    a2b = dt_b[a_surface]
    b2a = dt_a[b_surface]
    return a2b, b2a


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    a2b, b2a = _surface_distances(pred, target)
    distances = np.concatenate([a2b, b2a])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def assd(pred: np.ndarray, target: np.ndarray) -> float:
    a2b, b2a = _surface_distances(pred, target)
    distances = np.concatenate([a2b, b2a])
    if distances.size == 0:
        return 0.0
    return float(distances.mean())


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    if not _HAS_SKIMAGE:
        # Fallback: approximate via edge maps and IoU of dilated edges
        a2b, b2a = _surface_distances(pred, target)
        # coarse mapping to F1-like score
        hd = np.percentile(np.concatenate([a2b, b2a]), 95) if (pred.any() or target.any()) else 0.0
        return float(1.0 / (1.0 + hd))
    b_pred = find_boundaries(pred.astype(bool), mode='outer')
    b_gt = find_boundaries(target.astype(bool), mode='outer')
    se = disk(max(1, tolerance))
    b_pred_d = dilation(b_pred, se)
    b_gt_d = dilation(b_gt, se)
    # True positives: pred boundary pixels that hit a dilated GT boundary
    tp = np.logical_and(b_pred, b_gt_d).sum()
    fp = np.logical_and(b_pred, np.logical_not(b_gt_d)).sum()
    fn = np.logical_and(b_gt, np.logical_not(b_pred_d)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec + 1e-8)


# Composite training losses
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: (B,1,H,W) or (B,2,H,W) -> use sigmoid/softmax accordingly
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float()
            inter = (probs * targets).sum(dim=[2,3])
            denom = (probs + targets).sum(dim=[2,3])
            dice = (2 * inter + self.eps) / (denom + self.eps)
            return 1 - dice.mean()
        else:
            probs = F.softmax(logits, dim=1)[:,1:2,:,:]  # foreground
            targets = targets.float()
            inter = (probs * targets).sum(dim=[2,3])
            denom = (probs + targets).sum(dim=[2,3])
            dice = (2 * inter + self.eps) / (denom + self.eps)
            return 1 - dice.mean()


class BCEWithLogitsDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.shape[1] == 1:
            bce = self.bce(logits, targets.float())
            dice = self.dice(logits, targets)
        else:
            # Convert 2-class logits to logits for foreground
            logits_fg = logits[:,1:2,:,:]
            bce = self.bce(logits_fg, targets.float())
            dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


# --------------------------------------------------------------------------------------
# DenseCRF post-processing (pydensecrf if available)
# --------------------------------------------------------------------------------------

def apply_densecrf(image_gray01: np.ndarray,
                   prob_fg01: np.ndarray,
                   n_iters: int = 5,
                   sxy_gaussian: float = 3.0,
                   compat_gaussian: float = 3.0,
                   sxy_bilateral: float = 50.0,
                   srgb_bilateral: float = 5.0,
                   compat_bilateral: float = 10.0) -> np.ndarray:
    """
    image_gray01: HxW float in [0,1]
    prob_fg01:   HxW float in [0,1] (foreground probability)
    Returns refined prob_fg01 after DenseCRF mean-field iterations.
    """
    # Try to use pydensecrf; otherwise fall back to bilateral approximation
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
        H, W = image_gray01.shape
        # softmax over classes: background, foreground
        soft = np.stack([1.0 - prob_fg01, prob_fg01], axis=0)
        unary = unary_from_softmax(soft)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(unary)
        feats_gauss = create_pairwise_gaussian(sdims=(sxy_gaussian, sxy_gaussian), shape=(H, W))
        d.addPairwiseEnergy(feats_gauss, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        # Need 3-channel image for bilateral term
        if image_gray01.ndim == 2:
            if _HAS_CV2:
                img_rgb = cv2.cvtColor((image_gray01 * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = np.repeat((image_gray01 * 255).astype(np.uint8)[..., None], 3, axis=2)
        else:
            img_rgb = (image_gray01 * 255).astype(np.uint8)
        feats_bi = create_pairwise_bilateral(sdims=(sxy_bilateral, sxy_bilateral),
                                             schan=(srgb_bilateral, srgb_bilateral, srgb_bilateral),
                                             img=img_rgb, chdim=2)
        d.addPairwiseEnergy(feats_bi, compat=compat_bilateral, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(n_iters)
        refined = np.array(Q)[1, :].reshape((H, W))
        return refined.astype(np.float32)
    except Exception as e:
        # Fallback: edge-aware bilateral filtering + sharpening
        H, W = image_gray01.shape
        if _HAS_CV2:
            # bilateral filter the probability map guided by the image edges
            prob = prob_fg01.astype(np.float32)
            # cv2 bilateral filter expects 8-bit or float32; we'll rescale
            prob_bi = cv2.bilateralFilter(prob, d=7, sigmaColor=srgb_bilateral, sigmaSpace=sxy_bilateral)
            # light sharpening towards edges
            if _HAS_CV2:
                edges = cv2.Canny((image_gray01 * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
                prob_ref = np.clip(prob_bi + 0.2 * edges, 0.0, 1.0)
            else:
                prob_ref = prob_bi
            return prob_ref.astype(np.float32)
        else:
            # Minimal fallback: Gaussian blur (no edges)
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(prob_fg01, sigma=1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Training & Evaluation
# --------------------------------------------------------------------------------------

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    dsc = dice_coefficient(pred_mask, gt_mask)
    iou = iou_score(pred_mask, gt_mask)
    pr, rc = precision_recall(pred_mask, gt_mask)
    hd = hausdorff95(pred_mask, gt_mask)
    asd = assd(pred_mask, gt_mask)
    bf1 = boundary_f1(pred_mask, gt_mask, tolerance=2)
    return {"dice": dsc, "iou": iou, "precision": pr, "recall": rc,
            "hd95": hd, "assd": asd, "bf1": bf1}


def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   out_dir: str = None,
                   use_softmax_2c: bool = True) -> Dict[str, float]:
    model.eval()
    agg = {"dice": [], "iou": [], "precision": [], "recall": [], "hd95": [], "assd": [], "bf1": []}
    ensure_dir(out_dir) if out_dir else None
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)        # (B,1,H,W) normalized
            msk = batch["mask"].cpu().numpy()      # (B,1,H,W) 0/1
            paths = batch["path"]
            logits = model(img)                     # U-Net: (B,1,...) or (B,2,...) or log-probs if CRF
            if use_softmax_2c and logits.shape[1] == 2:
                prob = torch.softmax(logits, dim=1)[:,1:2,:,:]
            else:
                prob = torch.sigmoid(logits)
            prob_np = prob.cpu().numpy()

            for b in range(prob_np.shape[0]):
                p = prob_np[b,0]
                t = msk[b,0]
                pred_mask = (p >= 0.5).astype(np.uint8)
                metrics = compute_metrics(pred_mask, t)
                for k,v in metrics.items():
                    agg[k].append(v)

                # optionally save predictions
                if out_dir is not None:
                    base = os.path.splitext(os.path.basename(paths[b]))[0]
                    imwrite_gray(os.path.join(out_dir, f"{base}_prob.png"), p)
                    imwrite_gray(os.path.join(out_dir, f"{base}_pred.png"), pred_mask.astype(np.float32))
    # average
    return {k: float(np.mean(v)) if len(v)>0 else float('nan') for k,v in agg.items()}


def train_unet(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[Device] {device}")

    data_root = args.data_root
    proc_dir = os.path.join(data_root, "Processed")
    files_all = list_images(proc_dir)
    assert len(files_all) > 0, "No images found under Processed/"
    train_list, val_list, test_list = split_train_val_test(files_all, seed=args.seed)

    train_ds = SpineProcessedDataset(data_root, train_list, size=(args.size, args.size), augment=True)
    val_ds   = SpineProcessedDataset(data_root, val_list, size=(args.size, args.size), augment=False)
    test_ds  = SpineProcessedDataset(data_root, test_list, size=(args.size, args.size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = UNet(in_ch=1, out_ch=2, base=args.base).to(device)
    # Loss
    crit = BCEWithLogitsDiceLoss(bce_weight=0.5)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    ensure_dir(args.out_dir)
    ckpt_path = os.path.join(args.out_dir, "unet_best.pth")

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(img)  # (B,2,H,W)
            loss = crit(logits, msk)
            loss.backward()
            optimizer.step()
            running += loss.item() * img.size(0)
        train_loss = running / len(train_loader.dataset)

        # Validation loss using same criterion
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch in val_loader:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                logits = model(img)
                loss = crit(logits, msk)
                val_running += loss.item() * img.size(0)
            val_loss = val_running / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"  -> New best model saved to {ckpt_path}")

    # Final evaluation
    print("\n[Evaluation: U-Net baseline]")
    model.load_state_dict(torch.load(ckpt_path)["model"])
    metrics_val = evaluate_model(model, val_loader, device, out_dir=os.path.join(args.out_dir, "pred_val_unet"))
    metrics_test = evaluate_model(model, test_loader, device, out_dir=os.path.join(args.out_dir, "pred_test_unet"))
    print("Val metrics:", metrics_val)
    print("Test metrics:", metrics_test)


def train_crfrnn(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[Device] {device}")

    data_root = args.data_root
    proc_dir = os.path.join(data_root, "Processed")
    files_all = list_images(proc_dir)
    assert len(files_all) > 0, "No images found under Processed/"
    train_list, val_list, test_list = split_train_val_test(files_all, seed=args.seed)

    train_ds = SpineProcessedDataset(data_root, train_list, size=(args.size, args.size), augment=True)
    val_ds   = SpineProcessedDataset(data_root, val_list, size=(args.size, args.size), augment=False)
    test_ds  = SpineProcessedDataset(data_root, test_list, size=(args.size, args.size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model (U-Net + CRF-as-RNN produces log-probabilities for 2 classes)
    model = UNetWithCRF(in_ch=1, base=args.base, n_iters=args.crf_iters).to(device)
    # Loss: NLL loss on log-probabilities + Dice
    nll = nn.NLLLoss()  # expects log-probs
    dice = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    ensure_dir(args.out_dir)
    ckpt_path = os.path.join(args.out_dir, "unet_crfrnn_best.pth")

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)          # (B,1,H,W) normalized
            msk = batch["mask"].to(device)           # (B,1,H,W) 0/1

            optimizer.zero_grad()
            log_probs = model(img)                   # (B,2,H,W), log-softmax outputs
            # Prepare target for NLL (long)
            target_long = msk[:,0,:,:].long()
            loss_nll = nll(log_probs, target_long)
            # Dice on foreground channel computed directly on probabilities
            probs = torch.exp(log_probs)[:,1:2,:,:]
            targets = msk.float()
            inter = (probs * targets).sum(dim=[2,3])
            denom = (probs + targets).sum(dim=[2,3])
            loss_dice = (1 - (2*inter + 1e-6) / (denom + 1e-6)).mean()
            loss = 0.5 * loss_nll + 0.5 * loss_dice
            loss.backward()
            optimizer.step()
            running += loss.item() * img.size(0)

        train_loss = running / len(train_loader.dataset)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch in val_loader:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                log_probs = model(img)
                target_long = msk[:,0,:,:].long()
                loss_nll = nll(log_probs, target_long)
                probs = torch.exp(log_probs)[:,1:2,:,:]
                loss_dice = DiceLoss()(torch.log(probs.clamp(min=1e-6)), msk)
                loss = 0.5 * loss_nll + 0.5 * loss_dice
                val_running += loss.item() * img.size(0)
            val_loss = val_running / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"  -> New best model saved to {ckpt_path}")

    # Final evaluation
    print("\n[Evaluation: U-Net + CRF-as-RNN]")
    model.load_state_dict(torch.load(ckpt_path)["model"])
    # For evaluation, convert log-probabilities to logits equivalent by subtracting background channel
    def crfrnn_eval_wrapper(x):
        log_probs = model(x)
        return log_probs  # evaluate_model expects logits; it handles 2-class softmax
    metrics_val = evaluate_model(model, val_loader, device, out_dir=os.path.join(args.out_dir, "pred_val_crfrnn"))
    metrics_test = evaluate_model(model, test_loader, device, out_dir=os.path.join(args.out_dir, "pred_test_crfrnn"))
    print("Val metrics:", metrics_val)
    print("Test metrics:", metrics_test)


def eval_unet(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data_root = args.data_root
    proc_dir = os.path.join(data_root, "Processed")
    files_all = list_images(proc_dir)
    _, val_list, test_list = split_train_val_test(files_all, seed=args.seed)

    val_ds   = SpineProcessedDataset(data_root, val_list, size=(args.size, args.size), augment=False)
    test_ds  = SpineProcessedDataset(data_root, test_list, size=(args.size, args.size), augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_ch=1, out_ch=2, base=args.base).to(device)
    assert args.ckpt is not None and os.path.isfile(args.ckpt), "Provide --ckpt for the trained U-Net"
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    metrics_val = evaluate_model(model, val_loader, device, out_dir=os.path.join(args.out_dir, "pred_val_unet"))
    metrics_test = evaluate_model(model, test_loader, device, out_dir=os.path.join(args.out_dir, "pred_test_unet"))
    print("Val metrics:", metrics_val)
    print("Test metrics:", metrics_test)


def eval_crfrnn(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data_root = args.data_root
    proc_dir = os.path.join(data_root, "Processed")
    files_all = list_images(proc_dir)
    _, val_list, test_list = split_train_val_test(files_all, seed=args.seed)

    val_ds   = SpineProcessedDataset(data_root, val_list, size=(args.size, args.size), augment=False)
    test_ds  = SpineProcessedDataset(data_root, test_list, size=(args.size, args.size), augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetWithCRF(in_ch=1, base=args.base, n_iters=args.crf_iters).to(device)
    assert args.ckpt is not None and os.path.isfile(args.ckpt), "Provide --ckpt for the trained CRF-RNN"
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    metrics_val = evaluate_model(model, val_loader, device, out_dir=os.path.join(args.out_dir, "pred_val_crfrnn"))
    metrics_test = evaluate_model(model, test_loader, device, out_dir=os.path.join(args.out_dir, "pred_test_crfrnn"))
    print("Val metrics:", metrics_val)
    print("Test metrics:", metrics_test)


def crf_postprocess_eval(args):
    """
    Run U-Net to get probabilities, then apply DenseCRF post-processing and evaluate.
    """
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data_root = args.data_root
    proc_dir = os.path.join(data_root, "Processed")
    files_all = list_images(proc_dir)
    _, val_list, test_list = split_train_val_test(files_all, seed=args.seed)

    val_ds   = SpineProcessedDataset(data_root, val_list, size=(args.size, args.size), augment=False)
    test_ds  = SpineProcessedDataset(data_root, test_list, size=(args.size, args.size), augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)  # batch=1 for CRF convenience
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Backbone U-Net
    model = UNet(in_ch=1, out_ch=2, base=args.base).to(device)
    assert args.ckpt is not None and os.path.isfile(args.ckpt), "Provide --ckpt for the trained U-Net"
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    def eval_loader(crf_out_dir, loader):
        ensure_dir(crf_out_dir)
        agg = {"dice": [], "iou": [], "precision": [], "recall": [], "hd95": [], "assd": [], "bf1": []}
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)   # normalized
                msk = batch["mask"].numpy()       # (1,1,H,W)
                path = batch["path"][0]
                # predict
                logits = model(img)               # (1,2,H,W)
                prob = torch.softmax(logits, dim=1)[:,1:2,:,:].cpu().numpy()[0,0]  # (H,W)

                # bring back guidance image in [0,1] from Z-score input (approximation)
                img_np = img.cpu().numpy()[0,0]
                img_01 = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)

                # Apply DenseCRF
                p_ref = apply_densecrf(img_01, prob, n_iters=args.densecrf_iters,
                                       sxy_gaussian=args.sxy_gauss,
                                       compat_gaussian=args.compat_gauss,
                                       sxy_bilateral=args.sxy_bi,
                                       srgb_bilateral=args.srgb_bi,
                                       compat_bilateral=args.compat_bi)

                pred_mask = (p_ref >= 0.5).astype(np.uint8)
                gt = msk[0,0].astype(np.uint8)
                metrics = compute_metrics(pred_mask, gt)
                for k,v in metrics.items():
                    agg[k].append(v)

                base = os.path.splitext(os.path.basename(path))[0]
                imwrite_gray(os.path.join(crf_out_dir, f"{base}_prob_crf.png"), p_ref)
                imwrite_gray(os.path.join(crf_out_dir, f"{base}_pred_crf.png"), pred_mask.astype(np.float32))
        return {k: float(np.mean(v)) for k,v in agg.items()}

    m_val = eval_loader(os.path.join(args.out_dir, "pred_val_crf"), val_loader)
    m_test = eval_loader(os.path.join(args.out_dir, "pred_test_crf"), test_loader)
    print("[DenseCRF] Val metrics:", m_val)
    print("[DenseCRF] Test metrics:", m_test)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="CNN--CRF for Vertebrae Segmentation")
    p.add_argument("--data_root", type=str, required=True, help="Path to dataset root with Processed/ and Mask/")
    p.add_argument("--out_dir", type=str, default="outputs", help="Where to save models and predictions")
    p.add_argument("--mode", type=str, required=True,
                   choices=["train_unet", "eval_unet", "crf_post", "train_crfrnn", "eval_crfrnn"],
                   help="What to run")
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint for evaluation")
    p.add_argument("--size", type=int, default=512, help="Square resize for training & eval")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32, help="Base channel width of U-Net")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # DenseCRF hyperparams
    p.add_argument("--densecrf_iters", type=int, default=5)
    p.add_argument("--sxy_gauss", type=float, default=3.0)
    p.add_argument("--compat_gauss", type=float, default=3.0)
    p.add_argument("--sxy_bi", type=float, default=50.0)
    p.add_argument("--srgb_bi", type=float, default=5.0)
    p.add_argument("--compat_bi", type=float, default=10.0)

    # CRF-as-RNN iterations
    p.add_argument("--crf_iters", type=int, default=5)

    return p


def main():
    args = build_argparser().parse_args()

    if args.mode == "train_unet":
        train_unet(args)
    elif args.mode == "eval_unet":
        eval_unet(args)
    elif args.mode == "crf_post":
        crf_postprocess_eval(args)
    elif args.mode == "train_crfrnn":
        train_crfrnn(args)
    elif args.mode == "eval_crfrnn":
        eval_crfrnn(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
