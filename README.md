# ProjectName
Project Description.  

Implement a convolutional neural network based on SLFP (Small Logarithmic Floating-Point) using python.  

*Note:* Supported network models: vgg, alexnet, resnet50, mobelinetv1, mobelinetv2, mobelinetv3, shufflenetv2, etc.;   

*Note:* Supported activation functions include: relu, qelu, prelu, swish, mish, etc.;  

*Note:* Optimisers include:Adam, SGD, RMSPOP,SSGD, SGDNW ("FPGA-friendly Architecture of Processing Elements For Eficient and AccurateCNNS"), etc.;  

*Note:* Datasets include: MNIST, Cifar100, lmgnet1K.  

Related articles on SLFP:

"Small Logarithmic Floating-Point Multiplier Based on FPGA and Its Application on MobileNet"  

"FPGA-Friendly Architecture of Processing Elements For Efficient and Accurate CNNs"  

# User Guide

Please replace ’https://github.com/DLUT-IIS/CNN_Accelerator' with ‘your_github_name/your_repository’ in all links. ’

## Pre-development Configuration Requirement

Dependencies:
- Python 3.6+
- PyTorch 1.0+
- torchvision 0.2.2+

## **Installation steps**

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone 

```sh
git clone https://github.com/DLUT-IIS/CNN_Accelerator
```

## Description of the document catalogue
eg:

```
filetree 
├── LICENSE.txt
├── README.md
├── histograms/
├── logs/
├── max_act_wgt/
│  │  ├── cifar/
│  │  |  |  |── bitch_size_input_max.txt
│  │  |  |  └── bitch_size_weight_max.txt
│  │  ├── imgnet/
│  │  |  |── gelu/
│  │  |  |  |── mobilenetv2_gelu_layer_input_max.txt
│  │  |  |  |── mobilenetv2_gelu_layer_weight_max.txt
│  │  |  |  |── resnet_gelu_bitch_size_weight_max.txt
│  │  |  |  └── resnet_gelu_layer_input_max.txt
│  │  |  |── mish/
│  │  |  |  |── mobilenetv2_mish_layer_input_max.txt
│  │  |  |  |── mobilenetv2_mish_layer_weight_max.txt
│  │  |  |  |── resnet_mish_bitch_size_weight_max.txt
│  │  |  |  └── resnet_mish_layer_input_max.txt
│  │  |  |── prelu/
│  │  |  |  |── mobilenetv2_prelu_layer_input_max.txt
│  │  |  |  |── mobilenetv2_prelu_layer_weight_max.txt
│  │  |  |  |── resnet_prelu_bitch_size_weight_max.txt
│  │  |  |  └── resnet_prelu_layer_input_max.txt
│  │  |  |── relu/
│  │  |  |  |── bitch_size_weight_max.txt
│  │  |  |  |── layer_input_max.txt
│  │  |  |  |── resnet_layer_input_max.txt
│  │  |  |  └── resnet_layer_weight_max.txt
│  │  |  └── swish/
│  │  |  |  |── mobilenetv2_swish_layer_input_max.txt
│  │  |  |  |── mobilenetv2_swish_layer_weight_max.txt
│  │  |  |  |── resnet_swish_bitch_size_weight_max.txt
│  │  |  |  └── resnet_swish_layer_input_max.txt
├── nets_cifar
│  │  ├── LeNet.py
│  │  ├── mobilenetv1.py
│  │  ├── vgg16.py
├── nets_imgnet
│  │  ├── alexnet.py
│  │  ├── inception_v3.py
│  │  ├── mobilenetv1.py
│  │  ├── mobilenetv2.py
│  │  ├── resnet50.py
│  │  ├── util_resnet.py
│  │  ├── vgg16.py
├── training_txt/
│  │  ├── activation_func.py
│  │  ├── conv2d_func.py
│  │  ├── optimizer.py
│  │  ├── preprocessing.py
│  │  ├── scale_bitch_size_conv2d_func.py
│  │  ├── scale_factor.py
│  │  ├── sfp_quant.py
└── cifar100_train_eval.py
└── imgnet_train_eval.py

```




