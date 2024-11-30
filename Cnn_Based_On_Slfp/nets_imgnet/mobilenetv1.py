import sys
import torch
import torch.nn as nn
from torchsummary import summary

sys.path.append('..')
from utils.sfp_quant import *
from utils.conv2d_func import *
from utils.scale_factor import *
from utils.activation_func import *
from utils.scale_bitch_size_conv2d_func import *


class MobileNetV1_Q(nn.Module):
    def __init__(self, ch_in, qbit, pre_reference, activation_function):
        super(MobileNetV1_Q, self).__init__()

        """--------------- acquired scaling factor ----------------"""
        self.pre_reference = pre_reference

        """--------------- choose activation function ----------------"""
        if activation_function == 'relu':
               act_func = nn.ReLU
        elif activation_function == 'swish':
               act_func = Swish
        elif activation_function == "mish":
               act_func = nn.Mish
        elif activation_function == "gelu":
               act_func = nn.GELU
        elif activation_function == "prelu":
               act_func = nn.PReLU

        """--------------- choose bitch size scale----------------"""
        if self.pre_reference == True or qbit == 32:
                Ka = np.array([1]*50) 
                Kw = np.array([1]*50)

                Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
                Linear = linear_Q(q_bit = qbit, Kw = Kw[27], Ka = Ka[27])
        else:
                """------- 1. Standard convolution is not handled -------"""
                MAXIMUM_MAGNITUDE = 15.5
                
                ka = acquire_input_layer_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/layer_input_max.txt')   
                Ka = np.array(ka)/MAXIMUM_MAGNITUDE
         
                kw = acquire_weight_bitch_size_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/bitch_size_weight_max.txt')   
                Kw = [[x / MAXIMUM_MAGNITUDE for x in sub_list] if isinstance(sub_list, list) else sub_list / MAXIMUM_MAGNITUDE for sub_list in kw]
                

                Conv2d = conv2d_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw, Ka = Ka)
                Linear = linear_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw[27], Ka = Ka[27])     
                """------- 2. Standard convolution is FP32 -------"""
                """ 
                MAXIMUM_MAGNITUDE = 15.5
                ka = acquire_input_scale_factor_txt('/root/autodl-tmp/Cnn_Based_On_Slfp/max_act_wgt/imgnet/layer_input_max.txt')   
                Ka = np.array(ka)/MAXIMUM_MAGNITUDE
                
                # Ka = np.array([1]*50) 
         
                kw = acquire_weight_scale_factor_txt('/root/autodl-tmp/Cnn_Based_On_Slfp/max_act_wgt/imgnet/bitch_size_weight_max.txt')   
                Kw = [[x / MAXIMUM_MAGNITUDE for x in sub_list] if isinstance(sub_list, list) else sub_list / MAXIMUM_MAGNITUDE for sub_list in kw]
                
                Conv2d = conv2d_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw, Ka = Ka)
                Linear = linear_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw[27], Ka = Ka[27])    
                """

        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}
        

        def conv_dw(inp, oup, stride, Kw, Ka):
                return nn.Sequential(
                        # dw
                        Conv2d(inp, inp, 3, Kw[0], Ka[0], stride, 1, groups=inp, bias=False),
                        nn.BatchNorm2d(inp),
                        act_func(),
                 
                        
                        # pw
                        Conv2d(inp, oup, 1, Kw[1], Ka[1], 1, 0, bias=False),
                        nn.BatchNorm2d(oup),
                        act_func()
                        )

        def conv_bn(inp, oup, stride, Kw, Ka):
                return nn.Sequential(
                        Conv2d(inp, oup, 3, Kw, Ka, stride, 1,  bias=False), 
                        nn.BatchNorm2d(oup),
                        act_func()
                    )
        
        self.model = nn.Sequential(
                conv_bn(ch_in, 32, 2, Kw[0], Ka[0]),                # outout : 112 * 112 * 32
                conv_dw(32,  64, 1, Kw[1:], Ka[1:]),                # outout : 112 * 112 * 64
                conv_dw(64,  128, 2, Kw[3:], Ka[3:]),               # outout : 56 * 56 * 128
                conv_dw(128, 128, 1, Kw[5:], Ka[5:]),               # outout : 56 * 56 * 128
                conv_dw(128, 256, 2, Kw[7:], Ka[7:]),               # outout : 28 * 28 * 256
                conv_dw(256, 256, 1, Kw[9:], Ka[9:]),               # outout : 28 * 28 * 256
                conv_dw(256, 512, 2, Kw[11:], Ka[11:]),             # outout : 14 * 14 * 512
                conv_dw(512, 512, 1, Kw[13:], Ka[13:]),             # outout : 14 * 14 * 512
                conv_dw(512, 512, 1, Kw[15:], Ka[15:]),             # outout : 14 * 14 * 512
                conv_dw(512, 512, 1, Kw[17:], Ka[17:]),             # outout : 14 * 14 * 512
                conv_dw(512, 512, 1, Kw[19:], Ka[19:]),             # outout : 14 * 14 * 512
                conv_dw(512, 512, 1, Kw[21:], Ka[21:]),             # outout :  7 *  7 * 512
                conv_dw(512, 1024, 2, Kw[23:], Ka[23:]),            # outout :  7 *  7 * 1024
                conv_dw(1024, 1024, 1, Kw[25:], Ka[25:]),           # outout :  7 *  7 * 1024
                nn.AvgPool2d(7)
        )
        self.fc = Linear(1024, 1000)

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
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        
        if self.pre_reference == True:
                self.layer_inputs[0] = self.model[0][0].input_q
                self.layer_weights[0] = self.model[0][0].weight_q

                self.layer_inputs[1] = self.model[1][0].input_q
                self.layer_weights[1] = self.model[1][0].weight_q

                self.layer_inputs[2] = self.model[1][3].input_q
                self.layer_weights[2] = self.model[1][3].weight_q

                self.layer_inputs[3] = self.model[2][0].input_q
                self.layer_weights[3] = self.model[2][0].weight_q

                self.layer_inputs[4] = self.model[2][3].input_q
                self.layer_weights[4] = self.model[2][3].weight_q

                self.layer_inputs[5] = self.model[3][0].input_q
                self.layer_weights[5] = self.model[3][0].weight_q

                self.layer_inputs[6] = self.model[3][3].input_q
                self.layer_weights[6] = self.model[3][3].weight_q

                self.layer_inputs[7] = self.model[4][0].input_q
                self.layer_weights[7] = self.model[4][0].weight_q

                self.layer_inputs[8] = self.model[4][3].input_q
                self.layer_weights[8] = self.model[4][3].weight_q

                self.layer_inputs[9] = self.model[5][0].input_q
                self.layer_weights[9] = self.model[5][0].weight_q

                self.layer_inputs[10] = self.model[5][3].input_q
                self.layer_weights[10] = self.model[5][3].weight_q

                self.layer_inputs[11] = self.model[6][0].input_q
                self.layer_weights[11] = self.model[6][0].weight_q

                self.layer_inputs[12] = self.model[6][3].input_q
                self.layer_weights[12] = self.model[6][3].weight_q
                
                self.layer_inputs[13] = self.model[7][0].input_q
                self.layer_weights[13] = self.model[7][0].weight_q

                self.layer_inputs[14] = self.model[7][3].input_q
                self.layer_weights[14] = self.model[7][3].weight_q

                self.layer_inputs[15] = self.model[8][0].input_q
                self.layer_weights[15] = self.model[8][0].weight_q

                self.layer_inputs[16] = self.model[8][3].input_q
                self.layer_weights[16] = self.model[8][3].weight_q

                self.layer_inputs[17] = self.model[9][0].input_q
                self.layer_weights[17] = self.model[9][0].weight_q

                self.layer_inputs[18] = self.model[9][3].input_q
                self.layer_weights[18] = self.model[9][3].weight_q

                self.layer_inputs[19] = self.model[10][0].input_q
                self.layer_weights[19] = self.model[10][0].weight_q

                self.layer_inputs[20] = self.model[10][3].input_q
                self.layer_weights[20] = self.model[10][3].weight_q

                self.layer_inputs[21] = self.model[11][0].input_q
                self.layer_weights[21] = self.model[11][0].weight_q

                self.layer_inputs[22] = self.model[11][3].input_q
                self.layer_weights[22] = self.model[11][3].weight_q

                self.layer_inputs[23] = self.model[12][0].input_q
                self.layer_weights[23] = self.model[12][0].weight_q

                self.layer_inputs[24] = self.model[12][3].input_q
                self.layer_weights[24] = self.model[12][3].weight_q

                self.layer_inputs[25] = self.model[13][0].input_q
                self.layer_weights[25] = self.model[13][0].weight_q

                self.layer_inputs[26] = self.model[13][3].input_q
                self.layer_weights[26] = self.model[13][3].weight_q

                self.layer_inputs[27] = self.fc.input_q
                self.layer_weights[27] = self.fc.weight_q

                self.layer_outputs[27] = x
        return x

if __name__=='__main__':
    # model check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV1_Q(ch_in = 3, qbit = 32, pre_reference = False, activation_function = "relu").to(device)
    print(model)
