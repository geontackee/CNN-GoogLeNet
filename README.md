# CNN (GoogLeNet/Inception) on CIFAR-10 — PyTorch Implementation

This repository contains a **Convolutional Neural Network** implemented in **PyTorch**, training a **GoogLeNet-style** architecture with **Inception modules** on the **CIFAR-10** dataset.  
The notebook covers GPU setup, data transforms, model definition (Inception + GoogLeNet), training/evaluation loops, checkpointing, and accuracy reporting.

---

## 📌 Features
- **Dataset**: CIFAR-10 (auto-downloaded via `torchvision.datasets.CIFAR10`)
- **Transforms**:
  - Train: normalization + common augmentations (as defined in notebook)
  - Test: normalization only
- **Model**:
  - **Inception module** with dimensionality reduction
  - **GoogLeNet** composition of stacked Inception blocks
  - Implemented with `torch.nn.Module`
- **Training**:
  - GPU/CPU auto-selection (`torch.cuda.is_available()`)
  - Configurable hyperparameters (learning rate, batch sizes, epochs)
  - **Adam** optimizer; optional LR scheduling per milestones (as provided in notebook)
- **Evaluation**:
  - Running train/test loss & accuracy
  - Final accuracy printout
  - Model checkpoint saving

---

## ⚙️ Requirements
Install the essentials (CUDA optional):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the right CUDA/CPU wheel
pip install numpy matplotlib pillow
```

## How to Run
1. Clone github repository
2. Open Jupyter Notebook
3. Run all cells

