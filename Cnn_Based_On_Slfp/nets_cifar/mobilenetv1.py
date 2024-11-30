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
    def __init__(self, ch_in, qbit, pre_reference):
        super(MobileNetV1_Q, self).__init__()

        self.pre_reference =    pre_reference
 
        """--------------- choose bitch size scale----------------"""
        if self.pre_reference == True or qbit == 32:
                Ka = np.array([1]*50) 
                Kw = np.array([1]*50)

                Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
                Linear = linear_Q(q_bit = qbit, Kw = Kw[27], Ka = Ka[27])
        
        else:
                """------- 1. Standard convolution is not handled -------"""
                MAXIMUM_MAGNITUDE = 15.5
                ka = [2.7537312507629395, 5.917684078216553, 10.04549789428711, 8.252275466918945, 7.9321160316467285, 4.437595844268799, 7.5731892585754395, 4.2505693435668945, 4.2302751541137695, 3.1539459228515625, 4.045300006866455, 2.4653003215789795, 4.186850070953369, 1.8589879274368286, 3.7641642093658447, 1.3155642747879028, 3.7621400356292725, 1.5117807388305664, 3.296818971633911, 1.4270546436309814, 2.336785316467285, 1.0752859115600586, 1.8721150159835815, 0.8521665930747986, 1.4605412483215332, 0.8180747032165527, 2.3721730709075928, 6.377511501312256, 18.917125701904297]
                Ka = np.array(ka)/MAXIMUM_MAGNITUDE
                
                kw = acquire_weight_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/cifar/bitch_size_weight_max.txt')   
                Kw = [[x / MAXIMUM_MAGNITUDE for x in sub_list] if isinstance(sub_list, list) else sub_list / MAXIMUM_MAGNITUDE for sub_list in kw]
                        
                Conv2d = conv2d_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw, Ka = Ka)
                Linear = linear_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw[27], Ka = Ka[27]) 

                """------- 2. Standard convolution individually scaled -------"""
                """
                MAXIMUM_MAGNITUDE = 15.5
                ka = [2.7537312507629395, 5.917684078216553, 10.04549789428711, 8.252275466918945, 7.9321160316467285, 4.437595844268799, 7.5731892585754395, 4.2505693435668945, 4.2302751541137695, 3.1539459228515625, 4.045300006866455, 2.4653003215789795, 4.186850070953369, 1.8589879274368286, 3.7641642093658447, 1.3155642747879028, 3.7621400356292725, 1.5117807388305664, 3.296818971633911, 1.4270546436309814, 2.336785316467285, 1.0752859115600586, 1.8721150159835815, 0.8521665930747986, 1.4605412483215332, 0.8180747032165527, 2.3721730709075928, 6.377511501312256, 18.917125701904297]
                Ka = np.array(ka)/MAXIMUM_MAGNITUDE
                
                kw = acquire_weight_scale_factor_txt_improved('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/cifar/bitch_size_weight_max_improved.txt')   
                kw_first_layer = kw[0]
                kw_other       = kw[1:]

                Kw_first_layer = [[x / MAXIMUM_MAGNITUDE for x in sub_list] if isinstance(sub_list, list) else sub_list / MAXIMUM_MAGNITUDE for sub_list in kw_first_layer]
                Kw_other       = [[x / MAXIMUM_MAGNITUDE for x in sub_list] if isinstance(sub_list, list) else sub_list / MAXIMUM_MAGNITUDE for sub_list in kw_other]
                Kw = []
                Kw.append(Kw_first_layer)
                Kw[1:] = Kw_other
              
                Conv2d = conv2d_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw, Ka = Ka)
                Linear = linear_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw[27], Ka = Ka[27])  
                """
                """------- 3. Standard convolution is FP32 -------"""
                """
                MAXIMUM_MAGNITUDE = 15.5
                ka = [2.7537312507629395, 5.917684078216553, 10.04549789428711, 8.252275466918945, 7.9321160316467285, 4.437595844268799, 7.5731892585754395, 4.2505693435668945, 4.2302751541137695, 3.1539459228515625, 4.045300006866455, 2.4653003215789795, 4.186850070953369, 1.8589879274368286, 3.7641642093658447, 1.3155642747879028, 3.7621400356292725, 1.5117807388305664, 3.296818971633911, 1.4270546436309814, 2.336785316467285, 1.0752859115600586, 1.8721150159835815, 0.8521665930747986, 1.4605412483215332, 0.8180747032165527, 2.3721730709075928, 6.377511501312256, 18.917125701904297]
                Ka = np.array(ka)/MAXIMUM_MAGNITUDE
                
                kw = acquire_weight_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/cifar/bitch_size_weight_max.txt')   
                Kw = [[x / MAXIMUM_MAGNITUDE for x in sub_list] if isinstance(sub_list, list) else sub_list / MAXIMUM_MAGNITUDE for sub_list in kw]
                        
                Conv2d_Layer1 =  conv2d_Q(q_bit = 32, Kw = Kw, Ka = Ka)      
                Conv2d = conv2d_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw, Ka = Ka)
                Linear = linear_Q_bitch_size_scaling(q_bit = qbit, Kw = Kw[27], Ka = Ka[27])  
                """

        """--------------- choose layer scale----------------"""     
        Ka = [2.7537312507629395, 5.917684078216553, 10.04549789428711, 8.252275466918945, 7.9321160316467285, 4.437595844268799, 7.5731892585754395, 4.2505693435668945, 4.2302751541137695, 3.1539459228515625, 4.045300006866455, 2.4653003215789795, 4.186850070953369, 1.8589879274368286, 3.7641642093658447, 1.3155642747879028, 3.7621400356292725, 1.5117807388305664, 3.296818971633911, 1.4270546436309814, 2.336785316467285, 1.0752859115600586, 1.8721150159835815, 0.8521665930747986, 1.4605412483215332, 0.8180747032165527, 2.3721730709075928, 6.377511501312256, 18.917125701904297]
        Ka = np.array(Ka)/15.5
        # Ka = np.array([1]*50) 

        Kw = [1.4267572164535522, 1.9483833312988281, 1.1707897186279297, 0.5664068460464478, 0.7945287823677063, 0.9216379523277283, 0.6194500923156738, 0.3814568817615509, 0.5333530306816101, 0.587346613407135, 0.5604112148284912, 0.32260584831237793, 0.36938217282295227, 0.29242363572120667, 0.23009979724884033, 0.24940961599349976, 0.19502635300159454, 0.20861107110977173, 0.17908576130867004, 0.18250305950641632, 0.20463404059410095, 0.19018962979316711, 0.21646292507648468, 0.13597333431243896, 0.13064543902873993, 0.13209079205989838, 0.1518671214580536, 7.31498384475708]
        Kw = np.array(Kw)/15.5
        # Kw = np.array([1]*50) 

        Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q(q_bit = qbit, Kw = Kw[27], Ka = Ka[27]) 

        """--------------- choose scale means end----------------"""  


        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}

        def conv_dw(inp, oup, stride, Kw, Ka):
                return nn.Sequential(
                        # dw
                        Conv2d(inp, inp, 3, Kw[0], Ka[0], stride, 1, groups=inp, bias=False),
                        nn.BatchNorm2d(inp),
                        nn.ReLU(inplace=True),
                        # pw
                        Conv2d(inp, oup, 1, Kw[1], Ka[1], 1, 0, bias=False),
                        nn.BatchNorm2d(oup),
                        nn.ReLU(inplace=True),
                )

        def conv_bn(inp, oup, stride, Kw, Ka):
                return nn.Sequential(
                        Conv2d(inp, oup, 3, Kw, Ka, stride, 1,  bias=False), 
                        nn.BatchNorm2d(oup),
                        nn.ReLU(inplace=True),
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
                nn.AdaptiveAvgPool2d(1)
        )
        self.fc = Linear(1024, 100)

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
                self.layer_inputs[0]  = self.model[0][0].input_q
                self.layer_weights[0] = self.model[0][0].weight_q

                self.layer_inputs[1]  = self.model[1][0].input_q
                self.layer_weights[1] = self.model[1][0].weight_q

                self.layer_inputs[2]  = self.model[1][3].input_q
                self.layer_weights[2] = self.model[1][3].weight_q

                self.layer_inputs[3]  = self.model[2][0].input_q
                self.layer_weights[3] = self.model[2][0].weight_q

                self.layer_inputs[4]  = self.model[2][3].input_q
                self.layer_weights[4] = self.model[2][3].weight_q

                self.layer_inputs[5]  = self.model[3][0].input_q
                self.layer_weights[5] = self.model[3][0].weight_q

                self.layer_inputs[6]  = self.model[3][3].input_q
                self.layer_weights[6] = self.model[3][3].weight_q

                self.layer_inputs[7]  = self.model[4][0].input_q
                self.layer_weights[7] = self.model[4][0].weight_q

                self.layer_inputs[8]  = self.model[4][3].input_q
                self.layer_weights[8] = self.model[4][3].weight_q

                self.layer_inputs[9]  = self.model[5][0].input_q
                self.layer_weights[9] = self.model[5][0].weight_q

                self.layer_inputs[10]  = self.model[5][3].input_q
                self.layer_weights[10] = self.model[5][3].weight_q

                self.layer_inputs[11]  = self.model[6][0].input_q
                self.layer_weights[11] = self.model[6][0].weight_q

                self.layer_inputs[12]  = self.model[6][3].input_q
                self.layer_weights[12] = self.model[6][3].weight_q

                self.layer_inputs[13]  = self.model[7][0].input_q
                self.layer_weights[13] = self.model[7][0].weight_q

                self.layer_inputs[14]  = self.model[7][3].input_q
                self.layer_weights[14] = self.model[7][3].weight_q

                self.layer_inputs[15]  = self.model[8][0].input_q
                self.layer_weights[15] = self.model[8][0].weight_q

                self.layer_inputs[16]  = self.model[8][3].input_q
                self.layer_weights[16] = self.model[8][3].weight_q

                self.layer_inputs[17]  = self.model[9][0].input_q
                self.layer_weights[17] = self.model[9][0].weight_q

                self.layer_inputs[18]  = self.model[9][3].input_q
                self.layer_weights[18] = self.model[9][3].weight_q

                self.layer_inputs[19]  = self.model[10][0].input_q
                self.layer_weights[19] = self.model[10][0].weight_q

                self.layer_inputs[20]  = self.model[10][3].input_q
                self.layer_weights[20] = self.model[10][3].weight_q

                self.layer_inputs[21]  = self.model[11][0].input_q
                self.layer_weights[21] = self.model[11][0].weight_q

                self.layer_inputs[22]  = self.model[11][3].input_q
                self.layer_weights[22] = self.model[11][3].weight_q

                self.layer_inputs[23]  = self.model[12][0].input_q
                self.layer_weights[23] = self.model[12][0].weight_q

                self.layer_inputs[24]  = self.model[12][3].input_q
                self.layer_weights[24] = self.model[12][3].weight_q

                self.layer_inputs[25]  = self.model[13][0].input_q
                self.layer_weights[25] = self.model[13][0].weight_q

                self.layer_inputs[26]  = self.model[13][3].input_q
                self.layer_weights[26] = self.model[13][3].weight_q

                self.layer_inputs[27]  = self.fc.input_q
                self.layer_weights[27] = self.fc.weight_q

                self.layer_outputs[27] = x
        
        
        return x
##############

class MobileNetV1_swish(nn.Module):
    def __init__(self, ch_in, qbit):
        super(MobileNetV1_swish, self).__init__()
        
        # ka = [2.7537312507629395, 5.9545183181762695, 12.440423965454102, 7.20894193649292, 7.336813449859619, 4.235692501068115, 5.876864910125732, 4.778749465942383, 5.308851718902588, 3.2961573600769043, 4.680371284484863, 1.9677976369857788, 4.279696941375732, 1.8961942195892334, 5.168840408325195, 1.653853416442871, 4.763396263122559, 1.4445624351501465, 4.008443355560303, 1.4485292434692383, 4.650404930114746, 1.779747486114502, 3.6072027683258057, 2.8032076358795166, 2.662320137023926, 1.3386856317520142, 2.57434344291687, 9.546688079833984, 23.330665588378906]
        # ka = [15.390182495117188, 15.448236465454102, 13.605313301086426, 15.104866981506348, 14.362680435180664, 15.480584144592285, 15.308571815490723, 15.078579902648926, 15.262062072753906, 12.985400199890137, 13.853912353515625, 14.512986183166504, 15.0990571975708, 12.58373737335205, 15.129251480102539, 14.405941009521484, 12.992108345031738, 14.762649536132812, 15.33455753326416, 13.319290161132812, 15.491264343261719, 15.320398330688477, 15.494003295898438, 15.375253677368164, 14.556318283081055, 15.204405784606934, 12.60299301147461, 9.711894035339355]
        Ka = np.array([1]*50) 
        Ka = np.array(Ka)/15.5
        #Ka = np.ones_like(Ka)

        # kw = [1.277213454246521, 2.0415124893188477, 1.6111379861831665, 0.6062653660774231, 0.9518787264823914, 1.2452478408813477, 0.7040279507637024, 0.430196613073349, 0.8319571018218994, 0.6617120504379272, 0.5503885746002197, 0.3471983075141907, 0.37258848547935486, 0.37460413575172424, 0.3056546747684479, 0.3249901831150055, 0.2378476858139038, 0.3115707039833069, 0.21365958452224731, 0.2490127831697464, 0.1915145218372345, 0.23107753694057465, 0.24312478303909302, 0.32631564140319824, 0.21083834767341614, 0.10863597691059113, 0.20351974666118622, 13.45866870880127]
        # kw = [12.377761840820312, 13.478032112121582, 15.422836303710938, 14.992683410644531, 15.257452011108398, 14.823978424072266, 13.808598518371582, 14.622962951660156, 14.445660591125488, 15.398613929748535, 12.870665550231934, 14.95473861694336, 12.433236122131348, 15.099527359008789, 15.17325210571289, 14.734923362731934, 14.83267593383789, 15.484841346740723, 14.355389595031738, 15.143182754516602, 13.80544376373291, 15.052675247192383, 14.83446979522705, 15.162378311157227, 14.454427719116211, 15.111660957336426, 11.806191444396973, 0.8708469271659851]
        Kw = np.array([1]*50) 
        Kw = np.array(Kw)/15.5
        #Kw = np.ones_like(Kw)

        Conv2d = conv2d_Q(q_bit=qbit, Kw = Kw, Ka = Ka)
        Conv2d_with_swish = conv2d_Q_with_swish(q_bit=qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q(q_bit=qbit, Kw = Kw[27], Ka = Ka[27])
        Linear_with_swish = linear_Q_with_swish(q_bit=qbit, Kw = Kw[27], Ka = Ka[27])
        #self.act_q = activation_quantize_fn(a_bit=abit)
        layerout_quantize_func(q_bit=qbit)
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}

        def conv_dw_swish(inp, oup, stride, Kw, Ka):
            return nn.Sequential(
                # dw
                Conv2d(inp, inp, 3, Kw[0], Ka[0], stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                layerout_quantize_func(q_bit=qbit),
                Swish(),
                # pw
                Conv2d(inp, oup, 1, Kw[1], Ka[1], 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                layerout_quantize_func(q_bit=qbit),
                Swish(),
                )

        def conv_dw_with_swish0(inp, oup, stride, Kw, Ka):
            return nn.Sequential(
                # dw
                Conv2d(inp, inp, 3, Kw[0], Ka[0], stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                layerout_quantize_func(q_bit=qbit),
                # pw
                Conv2d_with_swish(inp, oup, 1, Kw[1], Ka[1], 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                layerout_quantize_func(q_bit=qbit),
                )

        def conv_dw_with_swish(inp, oup, stride, Kw, Ka):
            return nn.Sequential(
                # dw
                Conv2d_with_swish(inp, inp, 3, Kw[0], Ka[0], stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                layerout_quantize_func(q_bit=qbit),
                # pw
                Conv2d_with_swish(inp, oup, 1, Kw[1], Ka[1], 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                layerout_quantize_func(q_bit=qbit),
                )

        def conv_dw(inp, oup, stride, Kw, Ka):
            return nn.Sequential(
                # dw
                Conv2d(inp, inp, 3, Kw[0], Ka[0], stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                layerout_quantize_func(q_bit=qbit),
                nn.ReLU(),
                # pw
                Conv2d(inp, oup, 1, Kw[1], Ka[1], 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                layerout_quantize_func(q_bit=qbit),
                nn.ReLU(),
                )

        def conv_bn(inp, oup, stride, Kw, Ka):
                return nn.Sequential(
                    Conv2d(inp, oup, 3, Kw, Ka, stride, 1,  bias=False), 
                    nn.BatchNorm2d(oup),
                    layerout_quantize_func(q_bit=qbit),
                    nn.ReLU(),
                    )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2, Kw[0], Ka[0]),
            conv_dw(32, 64, 1, Kw[1:], Ka[1:]),
            conv_dw(64, 128, 2, Kw[3:], Ka[3:]),
            conv_dw(128, 128, 1, Kw[5:], Ka[5:]),
            conv_dw(128, 256, 2, Kw[7:], Ka[7:]),
            conv_dw(256, 256, 1, Kw[9:], Ka[9:]),
            conv_dw(256, 512, 2, Kw[11:], Ka[11:]),
            conv_dw(512, 512, 1, Kw[13:], Ka[13:]),
            conv_dw(512, 512, 1, Kw[15:], Ka[15:]),
            conv_dw(512, 512, 1, Kw[17:], Ka[17:]),
            conv_dw_swish(512, 512, 1, Kw[19:], Ka[19:]),
            conv_dw_swish(512, 512, 1, Kw[21:], Ka[21:]),
            conv_dw_swish(512, 1024, 2, Kw[23:], Ka[23:]),
            conv_dw_swish(1024, 1024, 1, Kw[25:], Ka[25:]),
            #nn.AvgPool2d(7)
            nn.AdaptiveAvgPool2d(1) #27 14
        )
        self.fc = Linear_with_swish(1024, 100)

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
        
        self.layer_inputs[0] = self.model[0][0].input_q
        self.layer_weights[0] = self.model[0][0].weight_q
        self.layer_inputs[1] = self.model[1][0].input_q
        self.layer_weights[1] = self.model[1][0].weight_q
        self.layer_inputs[2] = self.model[1][4].input_q
        self.layer_weights[2] = self.model[1][4].weight_q
        self.layer_inputs[3] = self.model[2][0].input_q
        self.layer_weights[3] = self.model[2][0].weight_q
        self.layer_inputs[4] = self.model[2][4].input_q
        self.layer_weights[4] = self.model[2][4].weight_q
        self.layer_inputs[5] = self.model[3][0].input_q
        self.layer_weights[5] = self.model[3][0].weight_q
        self.layer_inputs[6] = self.model[3][4].input_q
        self.layer_weights[6] = self.model[3][4].weight_q
        self.layer_inputs[7] = self.model[4][0].input_q
        self.layer_weights[7] = self.model[4][0].weight_q
        self.layer_inputs[8] = self.model[4][4].input_q
        self.layer_weights[8] = self.model[4][4].weight_q
        self.layer_inputs[9] = self.model[5][0].input_q
        self.layer_weights[9] = self.model[5][0].weight_q
        self.layer_inputs[10] = self.model[5][4].input_q
        self.layer_weights[10] = self.model[5][4].weight_q
        self.layer_inputs[11] = self.model[6][0].input_q
        self.layer_weights[11] = self.model[6][0].weight_q
        self.layer_inputs[12] = self.model[6][4].input_q
        self.layer_weights[12] = self.model[6][4].weight_q
        self.layer_inputs[13] = self.model[7][0].input_q
        self.layer_weights[13] = self.model[7][0].weight_q
        self.layer_inputs[14] = self.model[7][4].input_q
        self.layer_weights[14] = self.model[7][4].weight_q
        self.layer_inputs[15] = self.model[8][0].input_q
        self.layer_weights[15] = self.model[8][0].weight_q
        self.layer_inputs[16] = self.model[8][4].input_q
        self.layer_weights[16] = self.model[8][4].weight_q
        self.layer_inputs[17] = self.model[9][0].input_q
        self.layer_weights[17] = self.model[9][0].weight_q
        self.layer_inputs[18] = self.model[9][4].input_q
        self.layer_weights[18] = self.model[9][4].weight_q
        self.layer_inputs[19] = self.model[10][0].input_q
        self.layer_weights[19] = self.model[10][0].weight_q
        self.layer_inputs[20] = self.model[10][4].input_q
        self.layer_weights[20] = self.model[10][4].weight_q
        self.layer_inputs[21] = self.model[11][0].input_q
        self.layer_weights[21] = self.model[11][0].weight_q
        self.layer_inputs[22] = self.model[11][4].input_q
        self.layer_weights[22] = self.model[11][4].weight_q
        self.layer_inputs[23] = self.model[12][0].input_q
        self.layer_weights[23] = self.model[12][0].weight_q
        self.layer_inputs[24] = self.model[12][4].input_q
        self.layer_weights[24] = self.model[12][4].weight_q
        self.layer_inputs[25] = self.model[13][0].input_q
        self.layer_weights[25] = self.model[13][0].weight_q
        self.layer_inputs[26] = self.model[13][4].input_q
        self.layer_weights[26] = self.model[13][4].weight_q
        self.layer_inputs[27] = self.fc.input_q
        self.layer_weights[27] = self.fc.weight_q
        self.layer_outputs[27] = x
        return x

if __name__=='__main__':
    # model check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV1_Q(ch_in=3, qbit=32).to(device)
    #model = MobileNetV1(ch_in=3, n_classes=10).to(device)
    print(model)
    summary(model, input_size=(3, 224, 224), device='cuda')
