# ProjectName
Project Description.  

Implement a convolutional neural network based on SLFP (Small Logarithmic Floating-Point) using python.  

(1)  Supported network models: vgg, alexnet, resnet50, mobelinetv1, mobelinetv2, mobelinetv3, shufflenetv2, etc.;   

(2)  Supported activation functions include: relu, qelu, prelu, swish, mish, etc.;  

(3)  Optimisers include:Adam, SGD, RMSPOP,SSGD, SGDNW ("FPGA-friendly Architecture of Processing Elements For Eficient and AccurateCNNS"), etc.;  

(4) Datasets include: MNIST, Cifar100, lmgnet1K.  

Related articles on SLFP:

"Small Logarithmic Floating-Point Multiplier Based on FPGA and Its Application on MobileNet"  

"FPGA-Friendly Architecture of Processing Elements For Efficient and Accurate CNNs"  



## contents

- [User Guide](#User Guide)
  - [Pre-development Configuration Requirement](#Pre-development Configuration Requirement)
  - [Installation steps](#Installation steps)
- [Description of the document catalogue](#Description of the document catalogue)
- [Developed Architecture](#Developed Architecture)
- [Deployment](#Deployment)
- [Frameworks used](#Frameworks used)
- [Benefactor](#Benefactor)
  - [How to participate in open source projects](#How to participate in open source projects)
- [Version control](#Version control)
- [Author](#Author)
- [Acknowledgement](#Acknowledgement)

### User Guide

Please replace ‘zhangshize1103/Universal-CNN-accelerator’ with ‘your_github_name/your_repository’ in all links. ’

###### Pre-development Configuration Requirement

Dependencies:
- Python 3.6+
- PyTorch 1.0+
- torchvision 0.2.2+

###### **Installation steps**

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo

```sh
git clone https://github.com/zhangshize1103/Universal-CNN-accelerator/tree/main/Software)
```

### Description of the document catalogue
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


### Developed Architecture 

[ARCHITECTURE.md](https://github.com/zhangshize1103/Universal-CNN-accelerator/edit/main/Software/README.md)

### Deployment



### Frameworks used

- [xxxxxxx](https://getbootstrap.com)
- [xxxxxxx](https://jquery.com)
- [xxxxxxx](https://laravel.com)

### Benefactor



#### How to participate in open source projects




1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



### Version control


### Author


### Copyright


### Acknowledgement


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [xxxxxxxxxxxxxx](https://connoratherton.com/loaders)

<!-- links -->





