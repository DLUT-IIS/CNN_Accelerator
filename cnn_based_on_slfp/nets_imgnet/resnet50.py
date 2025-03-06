import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.sfp_quant import *
from torchsummary import summary
from torch.nn import functional as F
from utils.sfp_quant import *
from utils.activation_func import *
from utils.conv2d_func import *
from utils.scale_factor import *
from utils.scale_batch_size_conv2d_func import *
 
from typing import Type, Any, Callable, Union, List, Optional
 
import torch
import torch.nn as nn
from torch import Tensor
 
#from nets._internally_replaced_utils import load_state_dict_from_url
from nets_imgnet.util_resnet import _log_api_usage_once
 
__all__ = [
    "ResNet50",
]
 
class Bottleneck(nn.Module):
    expansion: int = 4
 
    def __init__(
        self,
        qbit,
        pre_reference,
        Kw,
        Ka,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.pre_reference = pre_reference
        if self.pre_reference == True or qbit == 32:
            Conv2d_1 = conv2d_Q(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])
            Conv2d_2 = conv2d_Q(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])
            Conv2d_3 = conv2d_Q(q_bit = qbit, Kw = Kw[3], Ka = Ka[3])
        else :
            """--------------- choose batch size scale----------------"""
            # Conv2d_1 = conv2d_Q_batch_size_scaling(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])
            # Conv2d_2 = conv2d_Q_batch_size_scaling(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])
            # Conv2d_3 = conv2d_Q_batch_size_scaling(q_bit = qbit, Kw = Kw[3], Ka = Ka[3])
            """--------------- choose layer scale----------------"""
            Conv2d_1 = conv2d_Q(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])
            Conv2d_2 = conv2d_Q(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])
            Conv2d_3 = conv2d_Q(q_bit = qbit, Kw = Kw[3], Ka = Ka[3])

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2d_1(in_channels = inplanes, out_channels = width, stride = 1, kernel_size = 1)   #conv1x1_1(inplanes, width)
    
        self.bn1 = norm_layer(width)
        self.conv2 = Conv2d_2(width, width, groups = 1, padding=dilation, dilation=dilation, stride = stride, kernel_size = 3)    #conv3x3_2(width, width, stride, groups, dilation)

        self.bn2 = norm_layer(width)
        self.conv3 = Conv2d_3(width, planes * self.expansion, stride = 1, kernel_size = 1) #conv1x1_3(width, planes * self.expansion)
        
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        # self.mish = nn.Mish()
        # self.swish = Swish()
        # self.gelu = nn.GELU()
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x: Tensor) -> Tensor:
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
 
 
class ResNet50(nn.Module):  
    def __init__(
        self,
        qbit, 
        pre_reference,
        #block = Bottleneck(qbit = qbit, Kw = Kw, Ka = Ka) ,  #type[Union xxx] 表示可以接受块类型为basicblock或bottleneck :Type[Union[BasicBlock, Bottleneck]]
        layers = [3,4,6,3],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.pre_reference = pre_reference

        
        if self.pre_reference == True or qbit == 32:
            Ka = np.array([1]*100) 
            Kw = np.array([1]*100)
            Conv2d_first = conv2d_Q(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
            Linear = linear_Q(q_bit = qbit, Kw = Kw[53], Ka = Ka[53])

        else:
            """--------------- choose batch size scale----------------"""
            # MAXIMUM_MAGNITUDE = 15.5
            # # ka = [2.7537312507629395, 5.917684078216553, 10.04549789428711, 8.252275466918945, 7.9321160316467285, 4.437595844268799, 7.5731892585754395, 4.2505693435668945, 4.2302751541137695, 3.1539459228515625, 4.045300006866455, 2.4653003215789795, 4.186850070953369, 1.8589879274368286, 3.7641642093658447, 1.3155642747879028, 3.7621400356292725, 1.5117807388305664, 3.296818971633911, 1.4270546436309814, 2.336785316467285, 1.0752859115600586, 1.8721150159835815, 0.8521665930747986, 1.4605412483215332, 0.8180747032165527, 2.3721730709075928, 6.377511501312256, 18.917125701904297]
            # ka = acquire_input_layer_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/resnet_layer_input_max.txt')   
            # Ka = np.array(ka)/MAXIMUM_MAGNITUDE
        
            # kw = acquire_weight_batch_size_scale_factor_txt('/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/resnet_layer_weight_max.txt')   
            # Kw = [[x / MAXIMUM_MAGNITUDE for x in sub_list] if isinstance(sub_list, list) else sub_list / MAXIMUM_MAGNITUDE for sub_list in kw]

            # Conv2d_first = conv2d_Q_batch_size_scaling(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
            # Linear = linear_Q_batch_size_scaling(q_bit = qbit, Kw = Kw[53], Ka = Ka[53])

            """--------------- choose layer scale----------------"""
            inout = [2.640000104904175, 6.8493123054504395, 6.8493123054504395, 2.0926101207733154, 2.3465774059295654, 4.545978546142578, 2.3446199893951416, 2.8475520610809326, 4.749269485473633, 2.107717990875244, 3.3412084579467773, 4.412791728973389, 4.412791728973389, 3.8282792568206787, 2.9802281856536865, 5.069820404052734, 1.4619481563568115, 2.186246395111084, 5.0605292320251465, 2.0890896320343018, 2.204008102416992, 5.053404808044434, 2.407410144805908, 3.188458204269409, 4.624925136566162, 4.624925136566162, 3.9921064376831055, 2.503716230392456, 3.886512041091919, 3.0490880012512207, 1.9895399808883667, 4.729367256164551, 1.8484134674072266, 1.7739477157592773, 4.359723091125488, 2.481842279434204, 2.022366762161255, 5.081398963928223, 3.197451591491699, 1.9158319234848022, 5.182647705078125, 2.850689172744751, 3.7739882469177246, 4.207239627838135, 4.207239627838135, 2.8491551876068115, 3.0215585231781006, 15.216011047363281, 3.1868929862976074, 1.979512095451355, 16.78635597229004, 2.9933321475982666, 2.6009302139282227, 7.67310094833374]
            weight = [0.7817208766937256, 0.987881064414978, 0.7266281247138977, 0.46786433458328247, 0.3936349153518677, 0.2617597281932831, 0.5201045870780945, 0.29462704062461853, 0.19206704199314117, 0.2855665683746338, 0.2751551568508148, 0.5662445425987244, 0.3531537353992462, 0.29927510023117065, 0.3916732370853424, 0.25216183066368103, 0.2997848093509674, 0.30379050970077515, 0.23830968141555786, 0.2555960714817047, 0.35215842723846436, 0.28143224120140076, 0.2209654152393341, 0.2956201732158661, 0.34601572155952454, 0.3425379693508148, 0.2007666528224945, 0.32124170660972595, 0.29417240619659424, 0.2634257674217224, 0.4968879222869873, 0.2714691460132599, 0.21002456545829773, 0.3537616431713104, 0.2390037477016449, 0.27921295166015625, 0.3126426041126251, 0.2721982002258301, 0.19188867509365082, 0.316133052110672, 0.39949774742126465, 0.2235630750656128, 0.32883593440055847, 0.6412832736968994, 0.3415152430534363, 0.3992723524570465, 0.3546474874019623, 0.700333833694458, 0.22574764490127563, 0.24268335103988647, 0.4540838599205017, 0.14155906438827515, 0.279774934053421, 0.7371371984481812]
        
            Ka = np.array(inout)/15.5
            Kw = np.array(weight)/15.5    

            Conv2d_first = conv2d_Q(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
            Linear = linear_Q(q_bit = qbit, Kw = Kw[53], Ka = Ka[53])

        """--------------- choose scale means end----------------"""
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}
        
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
 
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        self.qbit = qbit
        self.Kw = Kw
        self.Ka = Ka
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d_first(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  # 替换conv_Q，第一层
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.mish = nn.Mish(inplace=True)
        # self.swish = Swish()
        # self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], Kw = Kw[1:], Ka = Ka[1:])
        self.layer2 = self._make_layer(128, layers[1], Kw = Kw[11:], Ka = Ka[11:], stride=2)
        self.layer3 = self._make_layer(256, layers[2], Kw = Kw[24:], Ka = Ka[24:], stride=2)
        self.layer4 = self._make_layer(512, layers[3], Kw = Kw[43:], Ka = Ka[43:], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * Bottleneck.expansion, num_classes)  #换linear_Q, 最后一层
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
 
    def _make_layer(
        self,
        #block,  #: Type[Union[BasicBlock, Bottleneck]]
        planes: int,
        blocks: int,
        Kw,
        Ka,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if self.pre_reference == True or self.qbit == 32:
            conv1x1_downsample = conv2d_Q(q_bit = self.qbit, Kw = Kw[0], Ka = Ka[0])
        else:
            """--------------- choose batch size scale----------------"""
            # conv1x1_downsample = conv2d_Q_batch_size_scaling(q_bit = self.qbit, Kw = Kw[0], Ka = Ka[0])
            """--------------- choose layer scale----------------"""
            conv1x1_downsample = conv2d_Q(q_bit = self.qbit, Kw = Kw[0], Ka = Ka[0])           

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1_downsample(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride = stride),
                norm_layer(planes * Bottleneck.expansion),
            )
 
        layers = []
        layers.append(
            Bottleneck(
                self.qbit, self.pre_reference, Kw, Ka,   self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * Bottleneck.expansion
        for counter in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.qbit,
                    self.pre_reference,
                    Kw[3*counter:],
                    Ka[3*counter:],
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
 
        return nn.Sequential(*layers)

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

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.mish(x)
        # x = self.swish(x)
        # x = self.gelu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.pre_reference == True:
                self.layer_inputs[0] = self.conv1.input_q
                self.layer_weights[0] = self.conv1.weight_q
                self.layer_inputs[1] = self.layer1[0].downsample[0].input_q
                self.layer_weights[1] = self.layer1[0].downsample[0].weight_q
                self.layer_inputs[2] = self.layer1[0].conv1.input_q
                self.layer_weights[2] = self.layer1[0].conv1.weight_q
                self.layer_inputs[3] = self.layer1[0].conv2.input_q
                self.layer_weights[3] = self.layer1[0].conv2.weight_q
                self.layer_inputs[4] = self.layer1[0].conv3.input_q
                self.layer_weights[4] = self.layer1[0].conv3.weight_q
                self.layer_inputs[5] = self.layer1[1].conv1.input_q
                self.layer_weights[5] = self.layer1[1].conv1.weight_q
                self.layer_inputs[6] = self.layer1[1].conv2.input_q
                self.layer_weights[6] = self.layer1[1].conv2.weight_q
                self.layer_inputs[7] = self.layer1[1].conv3.input_q
                self.layer_weights[7] = self.layer1[1].conv3.weight_q
                self.layer_inputs[8] = self.layer1[2].conv1.input_q
                self.layer_weights[8] = self.layer1[2].conv1.weight_q
                self.layer_inputs[9] = self.layer1[2].conv2.input_q
                self.layer_weights[9] = self.layer1[2].conv2.weight_q
                self.layer_inputs[10] = self.layer1[2].conv3.input_q
                self.layer_weights[10] = self.layer1[2].conv3.weight_q
                self.layer_inputs[11] = self.layer2[0].downsample[0].input_q
                self.layer_weights[11] = self.layer2[0].downsample[0].weight_q
                self.layer_inputs[12] = self.layer2[0].conv1.input_q
                self.layer_weights[12] = self.layer2[0].conv1.weight_q
                self.layer_inputs[13] = self.layer2[0].conv2.input_q
                self.layer_weights[13] = self.layer2[0].conv2.weight_q
                self.layer_inputs[14] = self.layer2[0].conv3.input_q
                self.layer_weights[14] = self.layer2[0].conv3.weight_q
                self.layer_inputs[15] = self.layer2[1].conv1.input_q
                self.layer_weights[15] = self.layer2[1].conv1.weight_q
                self.layer_inputs[16] = self.layer2[1].conv2.input_q
                self.layer_weights[16] = self.layer2[1].conv2.weight_q
                self.layer_inputs[17] = self.layer2[1].conv3.input_q
                self.layer_weights[17] = self.layer2[1].conv3.weight_q
                self.layer_inputs[18] = self.layer2[2].conv1.input_q
                self.layer_weights[18] = self.layer2[2].conv1.weight_q
                self.layer_inputs[19] = self.layer2[2].conv2.input_q
                self.layer_weights[19] = self.layer2[2].conv2.weight_q
                self.layer_inputs[20] = self.layer2[2].conv3.input_q
                self.layer_weights[20] = self.layer2[2].conv3.weight_q
                self.layer_inputs[21] = self.layer2[3].conv1.input_q
                self.layer_weights[21] = self.layer2[3].conv1.weight_q
                self.layer_inputs[22] = self.layer2[3].conv2.input_q
                self.layer_weights[22] = self.layer2[3].conv2.weight_q
                self.layer_inputs[23] = self.layer2[3].conv3.input_q
                self.layer_weights[23] = self.layer2[3].conv3.weight_q
                self.layer_inputs[24] = self.layer3[0].downsample[0].input_q
                self.layer_weights[24] = self.layer3[0].downsample[0].weight_q
                self.layer_inputs[25] = self.layer3[0].conv1.input_q
                self.layer_weights[25] = self.layer3[0].conv1.weight_q
                self.layer_inputs[26] = self.layer3[0].conv2.input_q
                self.layer_weights[26] = self.layer3[0].conv2.weight_q
                self.layer_inputs[27] = self.layer3[0].conv3.input_q
                self.layer_weights[27] = self.layer3[0].conv3.weight_q
                self.layer_inputs[28] = self.layer3[1].conv1.input_q
                self.layer_weights[28] = self.layer3[1].conv1.weight_q
                self.layer_inputs[29] = self.layer3[1].conv2.input_q
                self.layer_weights[29] = self.layer3[1].conv2.weight_q
                self.layer_inputs[30] = self.layer3[1].conv3.input_q
                self.layer_weights[30] = self.layer3[1].conv3.weight_q
                self.layer_inputs[31] = self.layer3[2].conv1.input_q
                self.layer_weights[31] = self.layer3[2].conv1.weight_q
                self.layer_inputs[32] = self.layer3[2].conv2.input_q
                self.layer_weights[32] = self.layer3[2].conv2.weight_q
                self.layer_inputs[33] = self.layer3[2].conv3.input_q
                self.layer_weights[33] = self.layer3[2].conv3.weight_q
                self.layer_inputs[34] = self.layer3[3].conv1.input_q
                self.layer_weights[34] = self.layer3[3].conv1.weight_q
                self.layer_inputs[35] = self.layer3[3].conv2.input_q
                self.layer_weights[35] = self.layer3[3].conv2.weight_q
                self.layer_inputs[36] = self.layer3[3].conv3.input_q
                self.layer_weights[36] = self.layer3[3].conv3.weight_q
                self.layer_inputs[37] = self.layer3[4].conv1.input_q
                self.layer_weights[37] = self.layer3[4].conv1.weight_q
                self.layer_inputs[38] = self.layer3[4].conv2.input_q
                self.layer_weights[38] = self.layer3[4].conv2.weight_q
                self.layer_inputs[39] = self.layer3[4].conv3.input_q
                self.layer_weights[39] = self.layer3[4].conv3.weight_q
                self.layer_inputs[40] = self.layer3[5].conv1.input_q
                self.layer_weights[40] = self.layer3[5].conv1.weight_q
                self.layer_inputs[41] = self.layer3[5].conv2.input_q
                self.layer_weights[41] = self.layer3[5].conv2.weight_q
                self.layer_inputs[42] = self.layer3[5].conv3.input_q
                self.layer_weights[42] = self.layer3[5].conv3.weight_q
                self.layer_inputs[43] = self.layer4[0].downsample[0].input_q
                self.layer_weights[43] = self.layer4[0].downsample[0].weight_q
                self.layer_inputs[44] = self.layer4[0].conv1.input_q
                self.layer_weights[44] = self.layer4[0].conv1.weight_q
                self.layer_inputs[45] = self.layer4[0].conv2.input_q
                self.layer_weights[45] = self.layer4[0].conv2.weight_q
                self.layer_inputs[46] = self.layer4[0].conv3.input_q
                self.layer_weights[46] = self.layer4[0].conv3.weight_q
                self.layer_inputs[47] = self.layer4[1].conv1.input_q
                self.layer_weights[47] = self.layer4[1].conv1.weight_q
                self.layer_inputs[48] = self.layer4[1].conv2.input_q
                self.layer_weights[48] = self.layer4[1].conv2.weight_q
                self.layer_inputs[49] = self.layer4[1].conv3.input_q
                self.layer_weights[49] = self.layer4[1].conv3.weight_q
                self.layer_inputs[50] = self.layer4[2].conv1.input_q
                self.layer_weights[50] = self.layer4[2].conv1.weight_q
                self.layer_inputs[51] = self.layer4[2].conv2.input_q
                self.layer_weights[51] = self.layer4[2].conv2.weight_q
                self.layer_inputs[52] = self.layer4[2].conv3.input_q
                self.layer_weights[52] = self.layer4[2].conv3.weight_q
                self.layer_inputs[53] = self.fc.input_q
                self.layer_weights[53] = self.fc.weight_q
        
        return x
 
 

if __name__=='__main__':
    # model check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(qbit = 32).to(device)
    print(model)

