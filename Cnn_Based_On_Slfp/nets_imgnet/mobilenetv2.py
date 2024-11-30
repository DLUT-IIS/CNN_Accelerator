import torch.nn as nn
import torch
import math


import sys
import torch
import torch.nn as nn
from torchsummary import summary

sys.path.append('..')
from utils.sfp_quant import *
# from utils.conv2d_func import *
from utils.scale_factor import *
from utils.activation_func import *
from utils.scale_bitch_size_conv2d_func import *

__all__ = ['mobilenetv2']



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class InvertedResidual(nn.Module):
    def __init__(self, qbit, Kw, Ka, inp, oup, stride, expand_ratio, activation_function):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        """--------------- choose activation function ----------------"""
        if activation_function == 'relu':
               act_func = nn.ReLU6
        elif activation_function == 'swish':
               act_func = Swish
        elif activation_function == "mish":
               act_func = nn.Mish
        elif activation_function == "gelu":
               act_func = nn.GELU
        elif activation_function == "prelu":
               act_func = nn.PReLU

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        self.expand_ratio = expand_ratio

        Conv2d_layer2_1 = conv2d_Q(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
        Conv2d_layer2_2 = conv2d_Q(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])

        Conv2d_1 = conv2d_Q(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
        Conv2d_2 = conv2d_Q_Stride(q_bit = qbit, Kw = Kw[1], Ka = Ka[1], stride = stride)
        Conv2d_3 = conv2d_Q(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d_layer2_1(hidden_dim, hidden_dim, kernel_size = 3, stride = stride, padding = 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act_func(),

                # pw-linear
                Conv2d_layer2_2(hidden_dim, oup, kernel_size = 1, stride = 1, padding = 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d_1(inp, hidden_dim, kernel_size = 1, stride = 1, padding = 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act_func(),

                # dw
                Conv2d_2(hidden_dim, hidden_dim, kernel_size = 3, stride = stride, padding = 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act_func(),

                # pw-linear
                Conv2d_3(hidden_dim, oup, kernel_size = 1, stride = 1, padding = 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, qbit, pre_reference, activation_function, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__() 

        """--------------- acquired scaling factor ----------------"""
        self.pre_reference = pre_reference

        """--------------- choose activation function ----------------"""
        if activation_function == 'relu':
               act_func = nn.ReLU6
        elif activation_function == 'swish':
               act_func = Swish
        elif activation_function == "mish":
               act_func = nn.Mish
        elif activation_function == "gelu":
               act_func = nn.GELU
        elif activation_function == "prelu":
               act_func = nn.PReLU
        
        if self.pre_reference == True or qbit == 32:
            Ka = np.array([1]*100)
            Kw = np.array([1]*100)

        else:
            """--------------- choose bitch size scale----------------"""


            """--------------- choose layer scale ----------------"""
            MAXIMUM_MAGNITUDE = 15.5
            ka = acquire_input_layer_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/mobilenetv2_layer_input_max.txt')   
            kw = acquire_weight_layer_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/mobilenetv2_layer_weight_max.txt')   
            Ka = np.array(ka)/MAXIMUM_MAGNITUDE
            Kw = np.array(kw)/MAXIMUM_MAGNITUDE


        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q(q_bit = qbit, Kw = Kw[52], Ka = Ka[52])
        

        def conv_3x3_bn(inp, oup, stride, Kw, Ka):
            return nn.Sequential(
                Conv2d(inp, oup, 3, Kw, Ka, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                act_func()
            )


        def conv_1x1_bn(inp, oup, Kw, Ka):
            return nn.Sequential(
                Conv2d(inp, oup, 1, Kw, Ka, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                act_func()
            )

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, Kw[0], Ka[0])]
        # building inverted residual blocks
        block = InvertedResidual
        bottleneck_index = 1
        for t, c, n, s in self.cfgs:
           
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)

            if bottleneck_index == 1:
                for i in range(n):
                    layers.append(block(qbit, Kw[bottleneck_index:], Ka[bottleneck_index:], input_channel, output_channel, s if i == 0 else 1, t, activation_function))
                    input_channel = output_channel

                bottleneck_index = bottleneck_index + 2
            else :
                for i in range(n):
                    layers.append(block(qbit, Kw[bottleneck_index:], Ka[bottleneck_index:], input_channel, output_channel, s if i == 0 else 1, t, activation_function))
                    input_channel = output_channel
                    bottleneck_index = bottleneck_index + 3

        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, Kw[50], Ka[50])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(output_channel, num_classes)
        self.classifier = Linear(output_channel, num_classes)

        self._initialize_weights()

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
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.pre_reference == True:
            self.layer_inputs[0] = self.features[0][0].input_q
            self.layer_weights[0] = self.features[0][0].weight_q

            self.layer_inputs[1] = self.features[1].conv[0].input_q
            self.layer_inputs[2] = self.features[1].conv[3].input_q
            self.layer_weights[1] = self.features[1].conv[0].weight_q
            self.layer_weights[2] = self.features[1].conv[3].weight_q

            self.layer_inputs[3] = self.features[2].conv[0].input_q
            self.layer_inputs[4] = self.features[2].conv[3].input_q
            self.layer_inputs[5] = self.features[2].conv[6].input_q
            self.layer_weights[3] = self.features[2].conv[0].weight_q
            self.layer_weights[4] = self.features[2].conv[3].weight_q
            self.layer_weights[5] = self.features[2].conv[6].weight_q

            self.layer_inputs[6] = self.features[3].conv[0].input_q
            self.layer_inputs[7] = self.features[3].conv[3].input_q
            self.layer_inputs[8] = self.features[3].conv[6].input_q
            self.layer_weights[6] = self.features[3].conv[0].weight_q
            self.layer_weights[7] = self.features[3].conv[3].weight_q
            self.layer_weights[8] = self.features[3].conv[6].weight_q

            self.layer_inputs[9] = self.features[4].conv[0].input_q
            self.layer_inputs[10] = self.features[4].conv[3].input_q
            self.layer_inputs[11] = self.features[4].conv[6].input_q
            self.layer_weights[9] = self.features[4].conv[0].weight_q
            self.layer_weights[10] = self.features[4].conv[3].weight_q
            self.layer_weights[11] = self.features[4].conv[6].weight_q

            self.layer_inputs[12] = self.features[5].conv[0].input_q
            self.layer_inputs[13] = self.features[5].conv[3].input_q
            self.layer_inputs[14] = self.features[5].conv[6].input_q
            self.layer_weights[12] = self.features[5].conv[0].weight_q
            self.layer_weights[13] = self.features[5].conv[3].weight_q
            self.layer_weights[14] = self.features[5].conv[6].weight_q

            self.layer_inputs[15] = self.features[6].conv[0].input_q
            self.layer_inputs[16] = self.features[6].conv[3].input_q
            self.layer_inputs[17] = self.features[6].conv[6].input_q
            self.layer_weights[15] = self.features[6].conv[0].weight_q
            self.layer_weights[16] = self.features[6].conv[3].weight_q
            self.layer_weights[17] = self.features[6].conv[6].weight_q

            self.layer_inputs[18] = self.features[7].conv[0].input_q
            self.layer_inputs[19] = self.features[7].conv[3].input_q
            self.layer_inputs[20] = self.features[7].conv[6].input_q
            self.layer_weights[18] = self.features[7].conv[0].weight_q
            self.layer_weights[19] = self.features[7].conv[3].weight_q
            self.layer_weights[20] = self.features[7].conv[6].weight_q

            self.layer_inputs[21] = self.features[8].conv[0].input_q
            self.layer_inputs[22] = self.features[8].conv[3].input_q
            self.layer_inputs[23] = self.features[8].conv[6].input_q
            self.layer_weights[21] = self.features[8].conv[0].weight_q
            self.layer_weights[22] = self.features[8].conv[3].weight_q
            self.layer_weights[23] = self.features[8].conv[6].weight_q        

            self.layer_inputs[24] = self.features[9].conv[0].input_q
            self.layer_inputs[25] = self.features[9].conv[3].input_q
            self.layer_inputs[26] = self.features[9].conv[6].input_q
            self.layer_weights[24] = self.features[9].conv[0].weight_q
            self.layer_weights[25] = self.features[9].conv[3].weight_q
            self.layer_weights[26] = self.features[9].conv[6].weight_q

            self.layer_inputs[27] = self.features[10].conv[0].input_q
            self.layer_inputs[28] = self.features[10].conv[3].input_q
            self.layer_inputs[29] = self.features[10].conv[6].input_q
            self.layer_weights[27] = self.features[10].conv[0].weight_q
            self.layer_weights[28] = self.features[10].conv[3].weight_q
            self.layer_weights[29] = self.features[10].conv[6].weight_q        


            self.layer_inputs[30] = self.features[11].conv[0].input_q
            self.layer_inputs[31] = self.features[11].conv[3].input_q
            self.layer_inputs[32] = self.features[11].conv[6].input_q
            self.layer_weights[30] = self.features[11].conv[0].weight_q
            self.layer_weights[31] = self.features[11].conv[3].weight_q
            self.layer_weights[32] = self.features[11].conv[6].weight_q        


            self.layer_inputs[33] = self.features[12].conv[0].input_q
            self.layer_inputs[34] = self.features[12].conv[3].input_q
            self.layer_inputs[35] = self.features[12].conv[6].input_q
            self.layer_weights[33] = self.features[12].conv[0].weight_q
            self.layer_weights[34] = self.features[12].conv[3].weight_q
            self.layer_weights[35] = self.features[12].conv[6].weight_q


            self.layer_inputs[36] = self.features[13].conv[0].input_q
            self.layer_inputs[37] = self.features[13].conv[3].input_q
            self.layer_inputs[38] = self.features[13].conv[6].input_q
            self.layer_weights[36] = self.features[13].conv[0].weight_q
            self.layer_weights[37] = self.features[13].conv[3].weight_q
            self.layer_weights[38] = self.features[13].conv[6].weight_q


            self.layer_inputs[39] = self.features[14].conv[0].input_q
            self.layer_inputs[40] = self.features[14].conv[3].input_q
            self.layer_inputs[41] = self.features[14].conv[6].input_q
            self.layer_weights[39] = self.features[14].conv[0].weight_q
            self.layer_weights[40] = self.features[14].conv[3].weight_q
            self.layer_weights[41] = self.features[14].conv[6].weight_q


            self.layer_inputs[42] = self.features[15].conv[0].input_q
            self.layer_inputs[43] = self.features[15].conv[3].input_q
            self.layer_inputs[44] = self.features[15].conv[6].input_q
            self.layer_weights[42] = self.features[15].conv[0].weight_q
            self.layer_weights[43] = self.features[15].conv[3].weight_q
            self.layer_weights[44] = self.features[15].conv[6].weight_q

            self.layer_inputs[45] = self.features[16].conv[0].input_q
            self.layer_inputs[46] = self.features[16].conv[3].input_q
            self.layer_inputs[47] = self.features[16].conv[6].input_q
            self.layer_weights[45] = self.features[16].conv[0].weight_q
            self.layer_weights[46] = self.features[16].conv[3].weight_q
            self.layer_weights[47] = self.features[16].conv[6].weight_q


            self.layer_inputs[48] = self.features[17].conv[0].input_q
            self.layer_inputs[49] = self.features[17].conv[3].input_q
            self.layer_inputs[50] = self.features[17].conv[6].input_q
            self.layer_weights[48] = self.features[17].conv[0].weight_q
            self.layer_weights[49] = self.features[17].conv[3].weight_q
            self.layer_weights[50] = self.features[17].conv[6].weight_q

            self.layer_inputs[51] = self.conv[0].input_q
            self.layer_weights[51] = self.conv[0].weight_q

            self.layer_inputs[52] = self.classifier.input_q
            self.layer_weights[52] = self.classifier.weight_q


        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)


if __name__=='__main__':
    # model check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2(qbit = 32).to(device)
    print(model)

