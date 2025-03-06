import torch
from torch import nn
from torchsummary import summary
import sys
sys.path.append('..')
from utils.sfp_quant import *
from utils.conv2d_func import *

class MyLeNet5(nn.Module):

    def __init__(self,qbit):
        super(MyLeNet5, self).__init__()
        ''' 
        kw = [1.3840779066085815, 0.8324679732322693, 0.32982638478279114, 0.335301011800766, 0.8103049397468567]
        Kw = np.array(kw)/15
        ka = [1.0, 0.9996421933174133, 0.9999679327011108, 1.5531563758850098, 4.819594383239746]
        Ka = np.array(ka)/15
        '''
        Ka = [1 , 1 , 1, 1, 1]
        Kw = [1 , 1 , 1, 1, 1]

        Conv2d = conv2d_Q_bias(q_bit = qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q(q_bit=qbit, Kw = Kw, Ka = Ka)

        self.layer_inputs =  {}
        self.layer_outputs = {}
        self.layer_weights = {}        
        
        
        self.c1 = Conv2d(1,6,5,Kw[0],Ka[0],stride=1,padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = Conv2d(6,16,5,Kw[1],Ka[1],stride=1)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = Conv2d(16,120,5,Kw[2],Ka[2])
        self.flatten = nn.Flatten()
        self.f6 = Linear(120,84,Kw[3],Ka[3])
        self.output = Linear(84,10,Kw[4],Ka[4])
    
    
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
        
    def forward(self,x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        
        self.layer_inputs[0] = self.c1.input_q
        self.layer_weights[0] = self.c1.weight_q
        self.layer_inputs[1] = self.c3.input_q
        self.layer_weights[1] = self.c3.weight_q
        self.layer_inputs[2] = self.c5.input_q
        self.layer_weights[2] = self.c5.weight_q
        self.layer_inputs[3] = self.f6.input_q
        self.layer_weights[3] = self.f6.weight_q
        self.layer_inputs[4] = self.output.input_q
        self.layer_weights[4] = self.output.weight_q
        
        return x
    
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyLeNet5(qbit = 32).to(device)
    print(model)
    summary(model,input_size=(1,28,28),device='cuda')
