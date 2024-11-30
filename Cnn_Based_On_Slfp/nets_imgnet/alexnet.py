import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append('..')
from utils.sfp_quant import *
from utils.activation_func import *
from utils.conv2d_func import *

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, qbit, num_classes=1000):
        super(AlexNet, self).__init__()

        ka = [2.640000104904175,
        60.637901306152344,
        107.72291564941406,
        106.95045471191406,
        73.95004272460938,
        55.72099685668945,
        33.84217834472656,
        34.404571533203125 ]
        Ka = np.array(ka)/15.5

        kw = [0.9354032874107361,
        2.226973533630371,
        0.8556197881698608,
        0.3884967863559723,
        0.2260424792766571,
        0.07037778943777084,
        0.10437046736478806,
        0.21982255578041077]
        Kw = np.array(kw)/15.5

        Conv2d = conv2d_Q_bias(q_bit=qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q(q_bit=qbit, Kw = Kw, Ka = Ka)

        self.features = nn.Sequential(
            Conv2d(3, 64, 11, Kw[0], Ka[0], stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, 5, Kw[1], Ka[1], padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, 3, Kw[2], Ka[2], padding=1),
            nn.ReLU(inplace=True),
            Conv2d(384, 256, 3, Kw[3], Ka[3], padding=1),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, 3, Kw[4], Ka[4], padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * 6 * 6,  4096, Kw[5], Ka[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Linear(4096, 4096, Kw[6], Ka[6]),
            nn.ReLU(inplace=True),
            Linear(4096, num_classes, Kw[7], Ka[7]),
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
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        '''
        self.layer_inputs[0] = self.features[0].input_q
        self.layer_weights[0] = self.features[0].weight_q

        self.layer_inputs[1] = self.features[3].input_q
        self.layer_weights[1] = self.features[3].weight_q

        self.layer_inputs[2] = self.features[6].input_q
        self.layer_weights[2] = self.features[6].weight_q

        self.layer_inputs[3] = self.features[8].input_q
        self.layer_weights[3] = self.features[8].weight_q

        self.layer_inputs[4] = self.features[10].input_q
        self.layer_weights[4] = self.features[10].weight_q

        self.layer_inputs[5] = self.classifier[1].input_q
        self.layer_weights[5] = self.classifier[1].weight_q

        self.layer_inputs[6] = self.classifier[4].input_q
        self.layer_weights[6] = self.classifier[4].weight_q

        self.layer_inputs[7] = self.classifier[6].input_q
        self.layer_weights[7] = self.classifier[6].weight_q
        '''
        return x


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model

if __name__=='__main__':
    # model check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alexnet().to(device)
    print(model)