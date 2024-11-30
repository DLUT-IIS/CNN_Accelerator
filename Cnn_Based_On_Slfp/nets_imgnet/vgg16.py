import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from utils.sfp_quant import * 
from utils.activation_func import *
from utils.conv2d_func import *

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class vgg16_bn(nn.Module):
    def __init__(self, qbit, num_classes=1000):
        super(vgg16_bn, self).__init__()

        Ka = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        Kw = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
       
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}

        Linear = linear_Q(q_bit=qbit, Kw = Kw, Ka = Ka)

        self.features = self.make_layers(cfg, qbit, Ka = Ka, Kw = Kw, batch_norm=True)
        self.classifier = nn.Sequential(
            Linear(512*7*7, 4096, Ka[13], Kw[13]),
            #nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            Linear(4096, 4096, Ka[14], Kw[14]),
            #nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            Linear(4096, num_classes, Ka[15], Kw[15]),
            #nn.Linear(4096, num_classes),
        )

    def get_layer_inputs(self):
            return self.layer_inputs
        
    def get_layer_outputs(self):
            return self.layer_outputs
        
    def reset_layer_inputs_outputs(self):
            self.layer_inputs = {}
            self.layer_outputs = {}

    def get_layer_weights(self):
            return self.layer_weights
        
    def reset_layer_weights(self):
            self.layer_weights = {}

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        self.layer_inputs[0] = self.features[0].input_q
        self.layer_weights[0] = self.features[0].weight_q

        self.layer_inputs[1] = self.features[3].input_q
        self.layer_weights[1] = self.features[3].weight_q

        self.layer_inputs[2] = self.features[7].input_q
        self.layer_weights[2] = self.features[7].weight_q

        self.layer_inputs[3] = self.features[10].input_q
        self.layer_weights[3] = self.features[10].weight_q

        self.layer_inputs[4] = self.features[14].input_q
        self.layer_weights[4] = self.features[14].weight_q

        self.layer_inputs[5] = self.features[17].input_q
        self.layer_weights[5] = self.features[17].weight_q

        self.layer_inputs[6] = self.features[20].input_q
        self.layer_weights[6] = self.features[20].weight_q

        self.layer_inputs[7] = self.features[24].input_q
        self.layer_weights[7] = self.features[24].weight_q

        self.layer_inputs[8] = self.features[27].input_q
        self.layer_weights[8] = self.features[27].weight_q

        self.layer_inputs[9] = self.features[30].input_q
        self.layer_weights[9] = self.features[30].weight_q

        self.layer_inputs[10] = self.features[34].input_q
        self.layer_weights[10] = self.features[34].weight_q

        self.layer_inputs[11] = self.features[37].input_q
        self.layer_weights[11] = self.features[37].weight_q

        self.layer_inputs[12] = self.features[40].input_q
        self.layer_weights[12] = self.features[40].weight_q

        self.layer_inputs[13] = self.classifier[0].input_q
        self.layer_weights[13] = self.classifier[0].weight_q

        self.layer_inputs[14] = self.classifier[3].input_q
        self.layer_weights[14] = self.classifier[3].weight_q

        self.layer_inputs[15] = self.classifier[6].input_q
        self.layer_weights[15] = self.classifier[6].weight_q
       
        return x

    def make_layers(self, cfg, qbit, Ka, Kw, batch_norm=False):

        Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
        layers = []
        input_channel = 3
        conv_index = 0
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [Conv2d(input_channel, l, 3, Ka[conv_index], Kw[conv_index], padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = l
            conv_index += 1

        return nn.Sequential(*layers)



