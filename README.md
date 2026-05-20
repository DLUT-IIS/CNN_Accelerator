SLFP-CNN: Small Logarithmic Floating-Point CNN Accelerator
Python 3.6+
PyTorch 1.0+
License: MIT

This repository implements a Convolutional Neural Network (CNN) framework based on Small Logarithmic Floating-Point (SLFP) arithmetic using Python/PyTorch. It is designed to simulate and evaluate high-efficiency hardware-friendly quantization schemes for deep learning.

🌟 Key Features
Custom Arithmetic: Full support for SLFP (Small Logarithmic Floating-Point) simulation.
Diverse Architectures: Support for mainstream models including VGG, AlexNet, ResNet50, MobileNet (v1/v2/v3), and ShuffleNetV2.
Activation Functions: Includes standard and advanced functions like ReLU, GELU, PReLU, Swish, and Mish.
FPGA-Friendly Optimizers: Supports Adam, SGD, RMSprop, and the specialized SGDNW (FPGA-friendly Architecture of Processing Elements For Efficient and Accurate CNNs).
Benchmark Datasets: Integrated support for MNIST, CIFAR-100, and ImageNet-1K.
📖 Background & References
The core logic is based on research into low-bit-width floating-point arithmetic for hardware accelerators:

"Small Logarithmic Floating-Point Multiplier Based on FPGA and Its Application on MobileNet"
"FPGA-Friendly Architecture of Processing Elements For Efficient and Accurate CNNs"
Weight Update Scheme
The following diagram illustrates our weight update logic comparing Fixed-Point and Floating-Point (SLFP) schemes:
Weight Update Scheme
(Note: (1) FP32 update; (2) Rounding to low bit-width; (3) Update failure; (4) Update success.)

🚀 Quick Start
Prerequisites
Python 3.6+
PyTorch 1.0+
torchvision 0.2.2+
Installation
bash
# Clone the repository
git clone https://github.com/DLUT-IIS/CNN_Accelerator.git
cd CNN_Accelerator

# (Optional) Install dependencies
# pip install -r requirements.txt
📂 Project Structure
text
.
├── nets_cifar/             # Model definitions for CIFAR (LeNet, MobileNetV1, VGG16)
├── nets_imgnet/            # Model definitions for ImageNet (ResNet50, MobileNetV1/V2, InceptionV3)
├── training_txt/           # Core logic for quantization and functions
│   ├── activation_func.py  # Custom activation functions (GELU, Mish, etc.)
│   ├── sfp_quant.py        # SLFP quantization core implementation
│   ├── optimizer.py        # Optimizers including SGDNW
│   └── conv2d_func.py      # Quantized convolution layers
├── max_act_wgt/            # Activation and weight statistics (Batch/Layer-wise)
├── cifar100_train_eval.py  # Entry point for CIFAR-100
└── imgnet_train_eval.py    # Entry point for ImageNet-1K
📊 Performance & Pre-trained Models
Below is the accuracy (Top-1) of Quantized Networks using various activation functions on ImageNet-1K.

Model	Activation	Quantization	Optimizer	Accuracy
MobileNetV1	ReLU	SLFP (8-bit)	SGDNW	68.06%
MobileNetV2	ReLU	SLFP (8-bit)	SGDNW	71.87%
ResNet50	ReLU	SLFP (8-bit)	SGD	76.35%
ResNet50	Swish	SLFP (8-bit)	SGDNW	78.71%
ResNet50	GELU	SLFP (8-bit)	SGDNW	73.43%
MobileNetV2	Mish	SLFP (8-bit)	SGDNW	71.91%
Note: For a full list of pre-trained weights and detailed logs, please refer to the models.net_weights documentation sections in the code comments.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
