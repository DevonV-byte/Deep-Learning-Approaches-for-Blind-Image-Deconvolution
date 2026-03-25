# Deep Learning Approaches for Blind Image Deconvolution

Master thesis — Devon Vanaenrode — Universiteit Hasselt, 2024–2025
Promotor: Prof. dr. Jori Liesenborgs

---

## Overview

Blind image deconvolution is the problem of recovering a sharp latent image and its unknown blur kernel (PSF) from a single degraded photograph. This repository contains two complementary deep learning approaches investigated in the context of astronomical imaging (Hubble Space Telescope):

1. **SelfDeblur** (PyTorch) — A zero-shot method that jointly optimizes an image generator and a kernel generator using only the blurry test image itself, relying on deep architectural priors and softmax PSF normalization.
2. **Supervised CNN pipeline** (TensorFlow) — A data-driven approach that generates synthetic blurred image pairs from natural images and telescopic PSFs, then trains a convolutional network with an attention layer to regress blur kernels directly.

A key contribution is the **Maximum Normalized Convolution (MNC)** metric, which directly measures kernel recovery accuracy and proved more informative than PSNR/SSIM for evaluating blind deconvolution.

---

## Repository Structure

```
.
├── Code/                          # Supervised TensorFlow pipeline
│   ├── control.py                 # Main entry point (data engine + training)
│   ├── training.py                # Model architectures and training loop
│   ├── kernels.py                 # PSF generation (Gaussian, Moffat, Airy, empirical)
│   ├── convolution_mode.py        # Conv-mode training pipeline
│   ├── augmentation.py            # Data augmentation
│   ├── convolute.py               # Image-kernel convolution
│   ├── test_mode.py               # Evaluation routines
│   ├── visualise.py               # Result plotting
│   ├── utils.py                   # Kernel normalization, data splitting
│   ├── MSE_MNC_Plots.py           # Metric visualization
│   ├── pipeline.slurm             # HPC job script (WICE cluster, H100 GPUs)
│   └── pipeline_pymod.slurm       # HPC job script (Genius cluster, P100 GPUs)
│
├── SelfDeblur/                    # Zero-shot PyTorch implementation
│   ├── selfdeblur_levin.py        # Blind deconvolution on Levin dataset
│   ├── selfdeblur_levin_reproduce.py  # Reproduce published Levin results
│   ├── selfdeblur_lai.py          # Blind deconvolution on Lai dataset
│   ├── selfdeblur_lai_reproduce.py    # Reproduce published Lai results
│   ├── selfdeblur_ycbcr.py        # Color (YCbCr) image deconvolution
│   ├── selfdeblur_nonblind.py     # Non-blind variant (fixed known kernel)
│   ├── SSIM.py                    # SSIM loss implementation
│   ├── MNC.py                     # Maximum Normalized Convolution metric
│   ├── add_gaussian_noise.py      # Noise injection utility
│   ├── add_iterative_convolution.py   # Iterative blur application
│   ├── create_cutouts.py          # Image cropping
│   ├── visualise.py               # Visualization utilities
│   ├── models/                    # Network architectures (Skip, U-Net, ResNet, non-local)
│   ├── networks/                  # Alternative architecture implementations
│   ├── utils/common_utils.py      # Tensor ops, image I/O, cropping helpers
│   ├── Datasets/                  # Dataset directory (see Datasets section below)
│   └── pipeline_pymod.slurm       # HPC job script (Genius cluster)
│
└── Kernels and Loss Metrics/      # Standalone kernel analysis tools
    ├── kernels.py                 # Kernel generation library
    ├── kernel_comparison.py       # Comparative kernel analysis
    └── kernel_metric_visualizer.py    # Metric visualization
```

---

## Methods

### 1. SelfDeblur — Zero-Shot Neural Blind Deconvolution

Based on ["Neural Blind Deconvolution Using Deep Priors"](https://arxiv.org/abs/1908.02197) (Ren et al., 2020), extended with:
- **Hybrid loss functions**: MSE-only, SSIM-only, and MSE+SSIM variants evaluated on the Levin dataset
- **Novel MSE+MNC loss**: directly optimizes kernel structural alignment during training
- **Astronomical application**: applied to Hubble Space Telescope imagery with telescope-derived PSFs
- **Robustness analysis**: 10-run statistical evaluation on kernel recovery stability

The method jointly trains an encoder-decoder image generator and a fully-connected kernel generator on a single test image. Softmax normalization enforces physical PSF constraints (non-negative, unit sum).

### 2. Supervised CNN Pipeline

A modular data engine that:
- Loads high-resolution natural images (512×512) from the Kaggle Landmarks dataset
- Generates two PSF families: **Analytic PSFs (A-PSF)** from Photutils (Gaussian, Moffat, Airy-disk) and **Empirical PSFs (E-PSF)** from telescope data
- Applies synthetic convolution to produce (sharp, blurry) training pairs

Two model architectures were evaluated:
- **Model A** — Lightweight convolutional encoder with attention layer + dense decoder (32×32 kernel output). Trained with MSE+MNC loss → avoids mode collapse.
- **Model B** — Pure fully-connected regressor (control baseline) → exhibits catastrophic mode collapse, demonstrating that architectural inductive bias is critical.

---

## Key Results

### SelfDeblur — Loss Function Ablation (Levin Dataset)

| Loss | PSNR (dB) | SSIM | MNC |
|------|-----------|------|-----|
| MSE | 22.70 | 0.686 | 0.711 |
| SSIM | 17.79 | 0.416 | 0.714 |
| MSE + SSIM | 21.45 | 0.629 | 0.726 |

### SelfDeblur — Custom Astronomical Kernels (8 kernels)

| Metric | Value |
|--------|-------|
| Mean MNC | **0.918** (excellent structural alignment) |
| Mean PSNR | 20.08 dB |
| Mean SSIM | 0.576 |

### SelfDeblur — Robustness (10 independent runs)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Kernel MSE | 9.0 | 22.1 | 15.51 |
| MNC | 0.64 | 0.78 | 0.73 |

### Supervised Pipeline — Model A with MSE+MNC Loss

| PSF Type | MNC |
|----------|-----|
| Analytic PSFs (A-PSF) | ~0.98 |
| Empirical PSFs (E-PSF) | ~0.95 |

Model B (pure FC) collapsed to predicting the mean kernel regardless of input — confirming that convolutional architecture with proper inductive bias is essential.

---

## Requirements

### SelfDeblur (PyTorch)

```
Python >= 3.6
torch >= 0.4
torchvision
opencv-python
numpy
scipy
scikit-image
matplotlib
tqdm
Pillow
```

### Supervised Pipeline (TensorFlow)

```
Python 3.11
tensorflow >= 2.x
numpy
scipy
scikit-image
astropy
Pillow
seaborn
matplotlib
pandas
sklearn
```

---

## Datasets

Training was performed on KU Leuven's HPC cluster (WICE/Genius). Datasets are not included in this repository.

| Dataset | Used by | Source |
|---------|---------|--------|
| Google Landmarks v2 | Supervised pipeline | [Kaggle](https://www.kaggle.com/competitions/landmark-recognition-2021) — place at `../Base_Images/landmark/` relative to `Code/` |
| Levin et al. (2009) | SelfDeblur | [Published benchmark](http://www.wisdom.weizmann.ac.il/~levina/papers/LevinEtalCVPR09Data.rar) — place at `SelfDeblur/datasets/levin/` |
| Lai et al. (2016) | SelfDeblur | [Published benchmark](http://vllab.ucmerced.edu/wlai24/cvpr16_deblur_study/) — place at `SelfDeblur/datasets/lai/` |
| Hubble "Molten Ring" | SelfDeblur | [ESA/Hubble potw2050a](https://esahubble.org/images/potw2050a/) — place at `SelfDeblur/Datasets/Hubble/Base/potw2050a.tif` |

---

## HPC / Training

The `.slurm` files in `Code/` and `SelfDeblur/` are provided as reference for KU Leuven's WICE and Genius clusters. They contain cluster-specific paths and module loads that will not work outside that environment.

For local GPU training, call the entry points directly:

```bash
# Supervised pipeline
cd Code/
python control.py \
  --dataset-path ../Base_Images/landmark/ \
  --sample-percentage 0.3 \
  --batches 261 \
  --augmentation-rate 1

# SelfDeblur on Levin dataset
cd SelfDeblur/
python selfdeblur_levin.py
```

Run `python control.py --help` for the full list of arguments (loss function, learning rate, kernel size, epochs, etc.).

---

## Thesis

The full thesis is available as a release asset: [Master_Thesis_Devon_Vanaenrode.pdf](../../releases/latest)

```bibtex
@mastersthesis{Vanaenrode2025,
  author  = {Devon Vanaenrode},
  title   = {Deep Learning Approaches for Blind Image Deconvolution,
             with application to astronomical images},
  school  = {Universiteit Hasselt},
  year    = {2025},
  type    = {Master's thesis in Informatics},
  advisor = {Prof. dr. Jori Liesenborgs}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
