# CNN–CRF for Vertebrae Segmentation

This repository contains the code for a probabilistic graphical models (PGM) course project on **CNN–CRF models for vertebrae segmentation**.

The central question is:

> **When and how does a Conditional Random Field (CRF) prior help a convolutional neural network (CNN) segment vertebrae under limited supervision?**

We study this by combining a U-Net backbone with different CRF formulations and comparing their behavior in terms of spatial coherence and boundary quality.

---

## Project Overview

### Motivation

Pure CNN segmenters (e.g., U-Net) produce per-pixel predictions but have no explicit notion of spatial smoothness or label consistency. In low-label medical imaging regimes, this can lead to noisy or fragmented segmentations, especially near boundaries.

A **Conditional Random Field (CRF)** provides a principled way to encode spatial priors on top of CNN “unary” evidence:

- Encourage nearby pixels (or voxels) with similar appearance to have the same label.
- Preserve edges by reducing smoothing across strong image gradients.
- Connect PGM concepts (undirected models, mean-field inference, MAP decoding) with modern deep nets.

This project explores **three models** built on the same vertebra segmentation dataset:

1. **U-Net baseline** — CNN that produces per-pixel logits (unary terms).
2. **Post-hoc DenseCRF** — A fully connected CRF with Gaussian and bilateral kernels, applied to the CNN probabilities after training.
3. **CRF-as-RNN** — An unrolled mean-field CRF layer (spatial + bilateral pairwise terms with learnable Potts compatibility) trained end-to-end with the CNN.

