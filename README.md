<div align="center">

# 🧠 DL-Digit-Scanner

### **Handwritten Digits Recognition System using PyTorch**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-000000?style=for-the-badge&logo=pytorch&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9AB00.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

<p align="center">
  A high-performance Deep Learning model built from scratch to recognize and classify handwritten digits ($0 \text{ to } 9$) with high precision.
</p>

---
</div>

## 🚀 Key Features

* **Custom Architecture:** Built using `nn.Sequential` featuring optimized linear layers ($784 \to 512 \to 512 \to 10$) and ReLU activation functions.
* **Smart Streaming:** Implements PyTorch `DataLoader` to stream and process images efficiently in mini-batches of 32.
* **Adaptive Inference:** Includes an intelligent, zero-setup testing block that automatically checks image properties (background color) to accurately predict your own custom handwriting.
* **Hardware Accelerated:** Fully configured to leverage NVIDIA CUDA GPUs for rapid training.

---

## 🛠️ Tech Stack & Dynamic Badges

| Technology / Library | Purpose |
| :--- | :--- |
| **Python** | Core Programming Language |
| **PyTorch** | Deep Learning Framework & Neural Network API |
| **TorchVision** | Dataset Loading & Image Transformations |
| **Matplotlib** | Data Visualization & Plotting Predictions |
| **Pillow (PIL)** | Image Preprocessing & Format Conversions |

---

## 📊 Dataset: MNIST

The model is trained and validated on the historic **MNIST Dataset**, which consists of:
* **60,000** training examples.
* **10,000** testing examples.
* Each sample is a **$28 \times 28$ pixel grayscale image** of a handwritten digit centered in a black background.

---

## 🧠 Neural Network Architecture

The network processes the 2D image matrix by flattening it into a 1D vector before passing it through dense computational layers:

```text
Input (28x28 Image) 
     │
     ▼
[nn.Flatten()] ──────────────► Reshapes to a 784-dimensional vector
     │
     ▼
[nn.Linear(784, 512)] ───────► Hidden Layer 1 (Dense)
     │
     ▼
[nn.ReLU()] ─────────────────► Activation function for Non-Linearity
     │
     ▼
[nn.Linear(512, 512)] ───────► Hidden Layer 2 (Dense)
     │
     ▼
[nn.ReLU()] ─────────────────► Activation function
     │
     ▼
[nn.Linear(512, 10)] ────────► Output Layer (10 Class Probabilities)
