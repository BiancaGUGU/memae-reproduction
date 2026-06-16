# MemAE Reproduction on UCSD Ped2

## Overview

This repository contains a simplified reproduction of the paper:

**Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection**
(Gong et al., ICCV 2019)

The objective is to reproduce the core idea of MemAE and evaluate it on the **UCSD Ped2** anomaly detection dataset.

The implementation includes:

* Convolutional Autoencoder
* Memory Module
* Training on normal video frames only
* Reconstruction-based anomaly scoring
* Frame-level ROC AUC evaluation

---

## Repository Structure

```text
.
├── datasets/
│   └── ucsd_dataset.py
├── models/
│   └── conv_memae.py
├── train_ucsd.py
├── test_ucsd.py
├── evaluate_uscd.py
├── dataset_test.py
├── conv_test.py
└── README.md
```

---

## Dataset

Dataset used:

**UCSD Ped2**

Expected structure:

```text
data/
└── UCSD_Anomaly_Dataset.v1p2/
    └── UCSDped2/
        ├── Train/
        │   ├── Train001
        │   ├── Train002
        │   └── ...
        └── Test/
            ├── Test001
            ├── Test001_gt
            └── ...
```

Training uses only normal frames from the Train split.

Ground-truth masks from the Test split are used for evaluation.

---

## Environment

Python 3.12

Main dependencies:

```bash
pip install torch torchvision pillow scikit-learn matplotlib
```

---

## Model Architecture

Encoder:

```text
1×128×128
↓
Conv(32)
↓
Conv(64)
↓
Conv(128)
↓
256-dimensional latent vector
```

Memory module:

```text
Memory size = 200
Feature dimension = 256
```

Decoder:

```text
Latent
↓
Memory reconstruction
↓
Transposed convolutions
↓
1×128×128 reconstruction
```

---

## Training

Train the model:

```bash
python train_ucsd.py
```

Default settings:

```text
Epochs: 5
Batch size: 8
Learning rate: 1e-4
Optimizer: Adam
Loss: MSE
```

Output:

```text
conv_memae_ucsd.pth
```

---

## Evaluation

Run:

```bash
python evaluate_uscd.py
```

The script:

* computes reconstruction error
* generates anomaly scores
* computes frame-level ROC AUC
* reports inference FPS

---

## Results

Obtained results on UCSD Ped2:

| Metric          | Value |
| --------------- | ----- |
| Frame-level AUC | 0.637 |
| FPS (CPU)       | 64.64 |

Per-video AUC:

| Video   | AUC   |
| ------- | ----- |
| Test001 | 0.307 |
| Test002 | 1.000 |
| Test003 | 0.986 |
| Test004 | 1.000 |
| Test005 | 0.835 |
| Test006 | 0.827 |
| Test007 | 1.000 |
| Test012 | 0.810 |

Some videos produce NaN AUC values because only one class is present in the corresponding ground-truth labels, making ROC AUC undefined.

---

## Comparison with Original Paper

| Metric        | Paper | Reproduction |
| ------------- | ----- | ------------ |
| UCSD Ped2 AUC | 0.949 | 0.637        |
| FPS           | 38    | 64.64        |

Differences are expected because:

* simplified architecture
* limited training (5 epochs)
* partial corruption in the downloaded dataset
* no sparsity regularization
* no memory compactness loss
* no thresholding strategy from the original paper

---

## Verification Steps

To reproduce the results:

```bash
python dataset_test.py
python conv_test.py
python train_ucsd.py
python evaluate_uscd.py
```

Expected output:

```text
FRAME-LEVEL AUC: ~0.63
FPS: ~64
```

---

## References

Gong, D., Liu, L., Le, V., Saha, B., Mansour, M. R., Venkatesh, S., & van den Hengel, A.

Memorizing Normality to Detect Anomaly:
Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection.

ICCV 2019.

## Example Results

### Test001

![Test001](results/Test001_scores.png)

### Test005

![Test005](results/Test005_scores.png)

### Test012

![Test012](results/Test012_scores.png)
