# SLFP-CNN: Small Logarithmic Floating-Point CNN Accelerator

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.0+](https://img.shields.io/badge/pytorch-1.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: DUT](https://img.shields.io/badge/License-DLUT-yellow.svg)](https://opensource.org/licenses/DLUT)

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

## 📊 Experimental Results (ImageNet-1K)

The following table summarizes the Top-1 accuracy of quantized networks using different activation functions and SLFP arithmetic.

| Model | Activation | Quantization | Optimizer | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV1** | ReLU | SLFP (8-bit) | SGDNW | 68.06% |
| **MobileNetV2** | ReLU | SLFP (8-bit) | SGDNW | 71.87% |
| **ResNet50** | ReLU | SLFP (8-bit) | SGD | 76.35% |
| **ResNet50** | Swish | SLFP (8-bit) | SGDNW | 78.71% |
| **ResNet50** | GELU | SLFP (8-bit) | SGDNW | 73.43% |
| **MobileNetV2** | Mish | SLFP (8-bit) | SGDNW | 71.91% |
| **MobileNetV2** | PReLU | SLFP (8-bit) | SGDNW | 71.50% |

> **Note:** For detailed training logs and pre-trained `.pth` file references, please check the code comments in `models.net_weights`.

