# MemAE Reproduction and Extension for Video Anomaly Detection

## Overview

This repository contains a reproduction and extension of the MemAE (Memory-Augmented Autoencoder) architecture for anomaly detection.

The project reproduces the main MemAE concept using a convolutional autoencoder combined with a learnable memory module and evaluates the model on the UCSD Ped2 anomaly detection dataset.

In addition to the original reproduction, several extensions were implemented:

* Memory size optimization study
* Sparse memory addressing through entropy regularization
* Memory compactness loss
* Extended training (120 epochs)
* Automatic removal of corrupted frames

---

# Quick Start

Clone the repository:

```bash
git clone https://github.com/BiancaGUGU/memae-reproduction.git
cd memae-reproduction
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the extended model:

```bash
python train_memory_experiments.py
```

Evaluate the best checkpoint:

```bash
python evaluate_ucsd_extended.py
```

---

# Dataset

Experiments were performed on the UCSD Ped2 dataset.

Download the UCSD Anomaly Detection Dataset and place it inside:

```text
data/
└── UCSD_Anomaly_Dataset.v1p2/
```

Expected structure:

```text
data/
└── UCSD_Anomaly_Dataset.v1p2/
    └── UCSDped2/
        ├── Train/
        └── Test/
```

Corrupted frames are automatically detected and excluded during training.

---

# Implemented Extensions

## 1. Memory Size Study

The influence of memory capacity was investigated using:

* Memory Size = 500
* Memory Size = 1000
* Memory Size = 1200
* Memory Size = 1500

---

## 2. Sparse Memory Addressing

Entropy regularization was added to encourage sparse memory activation:

Loss = Reconstruction Loss + λ₁ × Entropy Loss

This approximates the sparse memory retrieval mechanism proposed in the original MemAE paper.

---

## 3. Memory Compactness Loss

A compactness loss was introduced:

Loss_compact = ||z − z_mem||²

where:

* z = latent representation
* z_mem = memory-reconstructed latent representation

This encourages memory retrieval consistency.

---

## 4. Extended Training

Training duration was increased to:

```text
120 epochs
```

to improve convergence.

---

## 5. Corrupted Data Filtering

Corrupted UCSD frames are automatically removed using image verification before training.

---

# Memory Size Study Results

| Memory Size | Best Epoch | Best Validation Loss |
| ----------- | ---------- | -------------------- |
| 500         | 119        | 0.000875             |
| 1000        | 120        | 0.000891             |
| 1200        | 120        | 0.000885             |
| 1500        | 118        | 0.000879             |

The best-performing configuration used a memory size of 500.

---

# Final Evaluation

The best checkpoint was evaluated on the UCSD Ped2 test set.

## Frame-Level Performance

| Metric          | Value    |
| --------------- | -------- |
| Frame-Level AUC | 0.631782 |
| FPS             | 56.53    |

---

## Per-Video AUC

| Sequence | AUC    |
| -------- | ------ |
| Test001  | 0.2939 |
| Test002  | 1.0000 |
| Test003  | 0.9829 |
| Test004  | 1.0000 |
| Test005  | 0.8162 |
| Test006  | 0.8071 |
| Test007  | 1.0000 |
| Test012  | 0.8257 |

Generated anomaly score plots are stored in:

```text
results/
```

---

# Repository Structure

```text
memae-reproduction/
│
├── datasets/
│   └── ucsd_dataset.py
│
├── models/
│   ├── conv_memae.py
│   ├── conv_memae_extended.py
│   ├── memory_module.py
│   ├── autoencoder.py
│   └── memae.py
│
├── results/
│
├── legacy/
│
├── train_memory_experiments.py
├── train_ucsd_extended.py
├── evaluate_ucsd_extended.py
├── plot_results.py
│
├── log_memory_500.csv
├── log_memory_1000.csv
├── log_memory_1200.csv
├── log_memory_1500.csv
│
├── README.md
└── requirements.txt
```

---

# Discussion

The obtained performance is lower than the results reported in the original MemAE paper.

This is expected because the current implementation uses:

* 2D convolutions
* frame-wise processing
* simplified memory representations

whereas the original MemAE architecture uses:

* 3D convolutions
* 16-frame clips
* spatio-temporal memory addressing

Therefore, this project should be considered a reproduction of the MemAE concept with additional experimental extensions rather than a complete reproduction of the original architecture.


