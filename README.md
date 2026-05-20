# SLFP-CNN: Small Logarithmic Floating-Point CNN Accelerator

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.0+](https://img.shields.io/badge/pytorch-1.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of Convolutional Neural Networks based on **SLFP (Small Logarithmic Floating-Point)** arithmetic. This project focuses on simulating low-bit-width floating-point operations to achieve hardware-friendly CNN acceleration.

---

## 🌟 Key Features

*   **Custom Arithmetic**: Optimized simulation of SLFP (Small Logarithmic Floating-Point) for deep learning.
*   **Comprehensive Model Support**: 
    *   VGG, AlexNet, ResNet50, MobileNet (v1/v2/v3), ShuffleNetV2, and more.
*   **Advanced Activation Functions**: 
    *   ReLU, GELU, PReLU, Swish, Mish, etc.
*   **Hardware-Friendly Optimizers**: 
    *   Adam, SGD, RMSprop, SSGD, and **SGDNW** (as proposed in *"FPGA-friendly Architecture of Processing Elements For Efficient and Accurate CNNs"*).
*   **Standard Benchmarks**: 
    *   MNIST, CIFAR-100, and ImageNet-1K.

---

## 📖 Theoretical Background

The implementation is inspired by and supports research on:
1.  *"Small Logarithmic Floating-Point Multiplier Based on FPGA and Its Application on MobileNet"*
2.  *"FPGA-Friendly Architecture of Processing Elements For Efficient and Accurate CNNs"*

### Weight Update Scheme
The figure below compares the weight update logic between Fixed-Point and Floating-Point (SLFP) schemes.
![Weight Update Scheme](https://raw.githubusercontent.com/DLUT-IIS/CNN_Accelerator/main/your_image_path.png) 
*(Note: (1) FP32 update; (2) Rounding; (3) Update failure; (4) Update success.)*

---

## 🚀 Quick Start

### Prerequisites
- Python 3.6+
- PyTorch 1.0+
- torchvision 0.2.2+

### Installation
```bash
# Clone the repository
git clone https://github.com/DLUT-IIS/CNN_Accelerator.git
cd CNN_Accelerator
.
├── nets_cifar/             # Model definitions for CIFAR (LeNet, MobileNetV1, VGG16)
├── nets_imgnet/            # Model definitions for ImageNet (ResNet50, MobileNetV1/V2, etc.)
├── training_txt/           # Core quantization and functional logic
│   ├── activation_func.py  # Custom activation implementations
│   ├── sfp_quant.py        # SLFP quantization core
│   ├── optimizer.py        # Optimizers including SGDNW
│   └── conv2d_func.py      # Quantized convolution layers
├── max_act_wgt/            # Pre-calculated Activation/Weight statistics
├── cifar100_train_eval.py  # Training entry for CIFAR-100
└── imgnet_train_eval.py    # Training entry for ImageNet-1K
