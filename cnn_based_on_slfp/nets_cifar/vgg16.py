
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys
sys.path.append('..')
from utils.sfp_quant import *   
from utils.activation_func import *
from utils.conv2d_func import *
from torch.nn import init


class VGG16_Q(nn.Module):
    def __init__(self, qbit):
        super(VGG16_Q, self).__init__()

        ka = [2.7537312507629395, 6.677286624908447, 8.463759422302246, 3.207906723022461, 4.464205265045166, 3.4566824436187744, 3.0526161193847656, 4.22322940826416, 3.0152904987335205, 3.672210454940796, 2.209512948989868, 1.7422523498535156, 1.0218650102615356, 4.529740810394287, 4.926044940948486, 7.873326301574707]
        Ka = np.array(ka)/15.5

        kw = [1.3687726259231567, 0.4951185882091522, 0.266398161649704, 0.24859268963336945, 0.2601229250431061, 0.1888742297887802, 0.2115396410226822, 0.14919601380825043, 0.10154223442077637, 0.07817015796899796, 0.06413832306861877, 0.08031665533781052, 0.08906633406877518, 0.14742621779441833, 0.15958546102046967, 0.17451441287994385]
        Kw = np.array(kw)/15.5

        Conv2d = conv2d_Q_bias(q_bit = qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q(q_bit=qbit, Kw = Kw, Ka = Ka)

        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}

        self.layer1 = nn.Sequential(
            Conv2d(3, 64, 3, Kw[0], Ka[0], 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            Conv2d(64, 64, 3, Kw[1], Ka[1], 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            Conv2d(64, 128, 3, Kw[2], Ka[2], 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            Conv2d(128, 128, 3, Kw[3], Ka[3], 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            Conv2d(128, 256, 3, Kw[4], Ka[4], 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            Conv2d(256, 256, 3, Kw[5], Ka[5], 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            Conv2d(256, 256, 3, Kw[6], Ka[6], 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            Conv2d(256, 512, 3, Kw[7], Ka[7], 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            Conv2d(512, 512, 3, Kw[8], Ka[8], 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            Conv2d(512, 512, 3, Kw[9], Ka[9], 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            Conv2d(512, 512, 3, Kw[10], Ka[10], 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            Conv2d(512, 512, 3, Kw[11], Ka[11], 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            Conv2d(512, 512, 3, Kw[12], Ka[12], 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # reshape to: (batch_size, channels, 1, 1)
            nn.Flatten(),
            Linear(512, 512, Kw[13], Ka[13]),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            Linear(512, 256, Kw[13], Ka[13]),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc3 = Linear(256, 100, Kw[13], Ka[13])

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        self.layer_inputs[0] = self.layer1[0].input_q
        self.layer_weights[0] = self.layer1[0].weight_q

        self.layer_inputs[1] = self.layer1[3].input_q
        self.layer_weights[1] = self.layer1[3].weight_q

        self.layer_inputs[2] = self.layer2[0].input_q
        self.layer_weights[2] = self.layer2[0].weight_q

        self.layer_inputs[3] = self.layer2[3].input_q
        self.layer_weights[3] = self.layer2[3].weight_q

        self.layer_inputs[4] = self.layer3[0].input_q
        self.layer_weights[4] = self.layer3[0].weight_q

        self.layer_inputs[5] = self.layer3[3].input_q
        self.layer_weights[5] = self.layer3[3].weight_q

        self.layer_inputs[6] = self.layer3[6].input_q
        self.layer_weights[6] = self.layer3[6].weight_q

        self.layer_inputs[7] = self.layer4[0].input_q
        self.layer_weights[7] = self.layer4[0].weight_q

        self.layer_inputs[8] = self.layer4[3].input_q
        self.layer_weights[8] = self.layer4[3].weight_q

        self.layer_inputs[9] = self.layer4[6].input_q
        self.layer_weights[9] = self.layer4[6].weight_q

        self.layer_inputs[10] = self.layer5[0].input_q
        self.layer_weights[10] = self.layer5[0].weight_q

        self.layer_inputs[11] = self.layer5[3].input_q
        self.layer_weights[11] = self.layer5[3].weight_q

        self.layer_inputs[12] = self.layer5[6].input_q
        self.layer_weights[12] = self.layer5[6].weight_q

        self.layer_inputs[13] = self.fc1[2].input_q
        self.layer_weights[13] = self.fc1[2].weight_q

        self.layer_inputs[14] = self.fc2[0].input_q
        self.layer_weights[14] = self.fc2[0].weight_q

        self.layer_inputs[15] = self.fc3.input_q
        self.layer_weights[15] = self.fc3.weight_q
        return x


class VGG16_gelu(nn.Module):
    def __init__(self, qbit ):
        super(VGG16_gelu, self).__init__()

        ka = [2.7537312507629395, 11.711949348449707, 12.62769603729248, 4.791536808013916, 5.771259307861328, 4.079975128173828, 5.4690165519714355, 5.7750959396362305, 9.395009994506836, 8.310335159301758, 4.690559387207031, 2.743001937866211, 2.2729594707489014, 5.308618068695068, 4.6886982917785645, 6.451591491699219]
        Ka = np.array(ka)/15.5
        #Ka = np.ones_like(Ka)

        kw = [1.570762276649475, 0.7567926645278931, 0.6021664142608643, 0.34334003925323486, 0.3607594072818756, 0.21708044409751892, 0.2369239330291748, 0.26995745301246643, 0.2696983218193054, 0.21481803059577942, 0.15883386135101318, 0.18896079063415527, 0.17712126672267914, 0.16824819147586823, 0.14480118453502655, 0.24329078197479248]
        Kw = np.array(kw)/15.5
        #Kw = np.ones_like(Kw)

        Conv2d_0 = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
        Conv2d = conv2d_Q_with_gelu(q_bit = qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q_with_gelu(q_bit=qbit, Kw = Kw, Ka = Ka)
        layerout_quantize_func(q_bit=qbit)

        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_weights = {}

        self.layer1 = nn.Sequential(
            Conv2d_0(3, 64, 3, Kw[0], Ka[0], 1, 1),
            nn.BatchNorm2d(64),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(64, 64, 3, Kw[1], Ka[1], 1, 1),
            nn.BatchNorm2d(64),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            Conv2d(64, 128, 3, Kw[2], Ka[2], 1, 1),
            nn.BatchNorm2d(128),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(128, 128, 3, Kw[3], Ka[3], 1, 1),
            nn.BatchNorm2d(128),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            Conv2d(128, 256, 3, Kw[4], Ka[4], 1, 1),
            nn.BatchNorm2d(256),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(256, 256, 3, Kw[5], Ka[5], 1, 1),
            nn.BatchNorm2d(256),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(256, 256, 3, Kw[6], Ka[6], 1, 1),
            nn.BatchNorm2d(256),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            Conv2d(256, 512, 3, Kw[7], Ka[7], 1, 1),
            nn.BatchNorm2d(512),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(512, 512, 3, Kw[8], Ka[8], 1, 1),
            nn.BatchNorm2d(512),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(512, 512, 3, Kw[9], Ka[9], 1, 1),
            nn.BatchNorm2d(512),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            Conv2d(512, 512, 3, Kw[10], Ka[10], 1, 1),
            nn.BatchNorm2d(512),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(512, 512, 3, Kw[11], Ka[11], 1, 1),
            nn.BatchNorm2d(512),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),

            Conv2d(512, 512, 3, Kw[12], Ka[12], 1, 1),
            nn.BatchNorm2d(512),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 将特征图尺寸转换为 (batch_size, channels, 1, 1)
            nn.Flatten(),
            Linear(512, 512, Kw[13], Ka[13]),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Sequential(
            Linear(512, 256, Kw[14], Ka[14]),
            layerout_quantize_func(q_bit=qbit),
            #nn.GELU(),
            Identity(),
            nn.Dropout(0.4)
        )
        self.fc3 = Linear(256, 100, Kw[15], Ka[15])

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        self.layer_inputs[0] = self.layer1[0].input_q
        self.layer_weights[0] = self.layer1[0].weight_q

        self.layer_inputs[1] = self.layer1[4].input_q
        self.layer_weights[1] = self.layer1[4].weight_q

        self.layer_inputs[2] = self.layer2[0].input_q
        self.layer_weights[2] = self.layer2[0].weight_q

        self.layer_inputs[3] = self.layer2[4].input_q
        self.layer_weights[3] = self.layer2[4].weight_q

        self.layer_inputs[4] = self.layer3[0].input_q
        self.layer_weights[4] = self.layer3[0].weight_q

        self.layer_inputs[5] = self.layer3[4].input_q
        self.layer_weights[5] = self.layer3[4].weight_q

        self.layer_inputs[6] = self.layer3[8].input_q
        self.layer_weights[6] = self.layer3[8].weight_q

        self.layer_inputs[7] = self.layer4[0].input_q
        self.layer_weights[7] = self.layer4[0].weight_q

        self.layer_inputs[8] = self.layer4[4].input_q
        self.layer_weights[8] = self.layer4[4].weight_q

        self.layer_inputs[9] = self.layer4[8].input_q
        self.layer_weights[9] = self.layer4[8].weight_q

        self.layer_inputs[10] = self.layer5[0].input_q
        self.layer_weights[10] = self.layer5[0].weight_q

        self.layer_inputs[11] = self.layer5[4].input_q
        self.layer_weights[11] = self.layer5[4].weight_q

        self.layer_inputs[12] = self.layer5[8].input_q
        self.layer_weights[12] = self.layer5[8].weight_q

        self.layer_inputs[13] = self.fc1[2].input_q
        self.layer_weights[13] = self.fc1[2].weight_q

        self.layer_inputs[14] = self.fc2[0].input_q
        self.layer_weights[14] = self.fc2[0].weight_q

        self.layer_inputs[15] = self.fc3.input_q
        self.layer_weights[15] = self.fc3.weight_q

        return x


if __name__ == '__main__':
  features = []

  def hook(self, input, output):
    #print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = VGG16_gelu(32).to(device)
  print(model)
  # torchsummary.summary(model, (3,224,224) ,device="cuda")