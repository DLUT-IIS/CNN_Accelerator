# Project Description

Implement a convolutional neural network based on SLFP (Small Logarithmic Floating-Point) using python.  

**Note:** Supported network models: vgg, alexnet, resnet50, mobelinetv1, mobelinetv2, mobelinetv3, shufflenetv2, etc.;   

**Note:** Supported activation functions include: relu, qelu, prelu, swish, mish, etc.;  

**Note:** Optimisers include:Adam, SGD, RMSPOP,SSGD, SGDNW ("FPGA-friendly Architecture of Processing Elements For Eficient and AccurateCNNS"), etc.;  

**Note:** Datasets include: MNIST, Cifar100, lmgnet1K.  


*Related articles on SLFP:*

*"Small Logarithmic Floating-Point Multiplier Based on FPGA and Its Application on MobileNet"*  

*"FPGA-Friendly Architecture of Processing Elements For Efficient and Accurate CNNs"*  

****

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
## models.net_weights
Accuracy of Quantized NET using various nonlinear activation functions on ImageNet-1K   
## models.net_weights_relu
\`\`\`python

    # "mobilenetv1":
    # pretrain_dir = 'mobnetv1_m1_base.pth'                                   # paper: 68.786%
    # pretrain_dir = '20241023_mobilenet_imgnet_1e6_bitch_size_68.068.pth'    # val-8bits:68.068%  
    # pretrain_dir = '20241023_mobilenet_imgnet_1e6_bitch_size_68.068.pth'    # val-8bits:68.068%
    # pretrain_dir = '20241119_mobilenetv1_imgnet_1e5_channel_SGDNW_slfp.pth' # val-8bits:68.064% 

    #"mobilenetv2":
    # pretrain_dir = 'mobilenetv2-c5e733a8.pth'                               # 72.85% 
    # pretrain_dir = 'mobilenetv2_1.0-0c6065bc.pth'                           # 72.186%
    # pretrain_dir = 'mobilenetv2_128x128-fd66a69d.pth'                       # 68.186%
    # pretrain_dir = 'mobilenetv2_160x160-64dc7fa1.pth'                       # 71.39%
    # pretrain_dir = 'mobilenetv2_192x192-e423d99e.pth'                       # 72.254% 
    # pretrain_dir = 'mobilenetv2_192x192-e423d99e.pth'                       # 72.254% 
    # pretrain_dir = '20241118_mobilenetv2_imgnet_1e5_layer_SGDNW_slfp.pth'   # 71.872% 
                                     
    # "resnet":
    # pretrain_dir = 'resnet-50.pth'                                          # 76.148% 
    # pretrain_dir = 'resnet_8perclass_slfp34_76.pth'                         # 75.864%
    # pretrain_dir = '20241120_resnet_imgnet_1e5_channel_SGDNW_slfp.pth'      # 76.276%
    # pretrain_dir = '20241120_resnet_imgnet_1e5_channel_SGD_WD0_slfp.pth'    # 76.352% 
\`\`\`




