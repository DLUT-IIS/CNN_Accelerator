# Project Description **

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

### models.net_weights_relu

```python
    # "mobilenetv1":
    # pretrain_dir = 'mobnetv1_m1_base.pth'                                   # 68.786%
    # pretrain_dir = '20241023_mobilenet_imgnet_1e6_bitch_size_68.068.pth'    # 68.068%  
    # pretrain_dir = '20241023_mobilenet_imgnet_1e6_bitch_size_68.068.pth'    # 68.068%
    # pretrain_dir = '20241119_mobilenetv1_imgnet_1e5_channel_SGDNW_slfp.pth' # 68.064% 

    # "mobilenetv2":
    # pretrain_dir = 'mobilenetv2_192x192-e423d99e.pth'                       # 72.254% 
    # pretrain_dir = '20241118_mobilenetv2_imgnet_1e5_layer_SGDNW_slfp.pth'   # 71.872% 
                                     
    # "resnet50":
    # pretrain_dir = 'resnet_8perclass_slfp34_76.pth'                         # 75.864%
    # pretrain_dir = '20241120_resnet_imgnet_1e5_channel_SGDNW_slfp.pth'      # 76.276%
    # pretrain_dir = '20241120_resnet_imgnet_1e5_channel_SGD_WD0_slfp.pth'    # 76.352% 
```

### models.net_weights_swish

```python
    # "mobilenetv1":  
    # pretrain_dir = '20241101_mobilenet_imagenet_swish_8bits_rmsprop_1e4_32.pth' # 66.644%
    # pretrain_dir = '20241101_mobilenet_imagenet_swish_8bits_rmsprop_5e6_32.pth' # 67.294%
    # pretrain_dir = '20241102_mobilenet_imagenet_swish_8bits_rmsprop_1e6_64.pth' # 67.396%
    # pretrain_dir = '20241102_mobilenet_imagenet_swish_8bits_rmsprop_1e6_128.pth'# 67.418%

  
    # "mobilenetv2":
    # pretrain_dir = '20241122_mobilenetv2_imgnet_swish_fp32.pth'             # 71.862%
    # pretrain_dir = '20241123_mobilenetv2_imgnet_swish_fp32.pth'             # 71.976%
    # pretrain_dir = '20241124_mobilenetv2_imgnet_swish_fp32.pth'             # 71.952%
    # pretrain_dir = '20241125_mobilenetv2_imgnet_swish_fp32.pth'             # 72.366%
    # pretrain_dir = '20241125_mobilenetv2_imgnet_swish_slfp_sgdnw.pth'       # 71.688% 
    # pretrain_dir = '20241126_mobilenetv2_imgnet_swish_slfp_sgd'             # 71.404%

    # "resnet50":
    # pretrain_dir = 20241109_resnet_imgnet_swish_32bits_1e5.pth'             # 79.64%
    # pretrain_dir = 20241112_resnet_imgnet_swish_8bits_1e5.pth'              # 78.482, 78.8%
    # pretrain_dir = 20241121_resnet_imgnet_swish_SGDNW_8bits_1e5.pth'        # 78.608%
    # pretrain_dir = 20241121_resnet_imgnet_swish_SGDNW_8bits_1e6.pth'        # 78.64%
    # pretrain_dir = 20241122_resnet_imgnet_swish_SGDNW_8bits_1e6.pth'        # 78.71%
```

### models.net_weights_gelu

```python
    # "mobilenetv1":
    # pretrain_dir = 'mobnetv1_m1_base.pth'                                  # 68.786%
    # pretrain_dir = '20241025_mobilenet_imgnet_gelu_32bits.pth'             # 64.928%
  
    # "mobilenetv2":
    # pretrain_dir = '20241122_mobilenetv2_imgnet_gelu_fp32.pth'             # 71.782% 
    # pretrain_dir = '20241126_mobilenetv2_imgnet_gelu_fp32.pth'             # 72.122% 
    # pretrain_dir = '20241126_mobilenetv2_imgnet_gelu_slfp_sgd.pth'         # 71.276%
    # pretrain_dir = '20241126_mobilenetv2_imgnet_gelu_slfp_sgdnw.pth'       # 71.588%

    # "resnet50":
    # pretrain_dir = 'resnet-50.pth'
    # pretrain_dir = 'resnet50-11ad3fa6.pth'  # 80.344
    # pretrain_dir = '20241108_resnet_imgnet_gelu_32bits_1e5.pth'            # 74.11%
    # pretrain_dir = '20241112_resnet_imgnet_gelu_8bits_1e5.pth'             # 73.118%
    # pretrain_dir = '20241120_resnet_imgnet_gelu_SGDNW_8bits_1e5.pth'       # 73.432%
```

### models.net_weights_mish

```python
    # "mobilenetv1"   
    # pretrain_dir = 'mobnetv1_m1_base.pth'                                  # 68.786% 

    # "mobilenetv2":
    # pretrain_dir = '20241123_mobilenetv2_imgnet_mish_fp32.pth'             # 71.628%
    # pretrain_dir = '20241126_mobilenetv2_imgnet_mish_fp32.pth'             # 72.298%
    # pretrain_dir = '20241126_mobilenetv2_imgnet_mish_slfp_sgd.pth'         # 71.466%
    # pretrain_dir = '20241126_mobilenetv2_imgnet_mish_slfp_sgdnw.pth'       # 71.914%

    # resnet50
    # pretrain_dir = '20241108_resnet_imgnet_mish_32bits_1e5.pth'            # 79.78%
    # pretrain_dir = '20241111_resnet_imgnet_mish_8bits_1e5.pth'             # 77.722%
    # pretrain_dir = '20241118_resnet_imgnet_mish_SGDNW_8bits_1e5.pth'       # 78.154%
```

### models.net_weights_prelu

```python
    # "mobilenetv1":
    # pretrain_dir = '20241030_mobilenet_imgnet_prelu_32bits_1e6.pth'        # 68.302%


    # "mobilenetv2":
    # pretrain_dir = '20241123_mobilenetv2_imgnet_prelu_fp32.pth'            # 71.968%
    # pretrain_dir = '20241125_mobilenetv2_imgnet_prelu_fp32.pth'            # 72.334%
    # pretrain_dir = '20241125_mobilenetv2_imgnet_prelu_slfp_sgdnw.pth'      # 71.502%
    # pretrain_dir = '20241126_mobilenetv2_imgnet_prelu_slfp_sgd.pth'        # 71.366%

    # "resnet50":
    # pretrain_dir = '220241109_resnet_imgnet_prelu_32bits_1e5.pth'          # 79.22% 
    # pretrain_dir = '20241119_resnet_imgnet_prelu_8bits_1e5.pth'            # 76.51% 
    # pretrain_dir = '20241120_resnet_imgnet_prelu_SGDNW_8bits_1e5.pth'      # 77.128% 
```


