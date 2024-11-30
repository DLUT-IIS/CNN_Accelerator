"""
    File function: The convolutional layer and the linear are improved to scale each set of weights.
    Author: Zhangshize
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.sfp_quant import *
import time

# ------------------------------------ layer_scale---------------------------------------- #
def conv2d_Q(q_bit, Kw, Ka):
  class Conv2d_Q(nn.Conv2d):  
    def __init__(self, in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, 
    stride=1, padding=0, dilation=1, groups=1, bias=False):
      '''
          in_channels:  Input channel number
          out_channels: The number of output channels, which determines the number of convolution cores, is also the number of channels for output data
          kernel_size:  Size of the convolution kernel
          Kw:           Scale factor of weight, scale factor of activation value
          Ka:           Represents an argument to the activation function
          stride:       The step size of the convolution
          padding:      The number of pixels filled around the input data to control the shape of the output
          dilation:     The spacing between pixels in the convolution kernel
          groups:       Controls how input and output channels are connected
          bias:         Whether to use an offset item
      '''
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.q_bit = q_bit                                       # Is it quantified and in what format 
      self.quantize_weight = weight_quantize_func(q_bit=q_bit) # Quantization weight
      self.quantize_act = act_quantize_func(q_bit=q_bit)       # Quantization act
      self.Kw = torch.tensor(Kw)                               # Convert data Kw to a tensor object for PyTorch
      self.Ka = torch.tensor(Ka)

    def forward(self, input, order=None):
      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.weight/self.Kw) #
      self.output = F.conv2d(self.input_q, self.weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)*self.Ka*self.Kw
      return self.output
  return Conv2d_Q

def conv2d_Q_with_swish(q_bit, Kw, Ka):
  class Conv2d_Q_with_swish(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, 
    stride=1, padding=0, dilation=1, groups=1, bias=False):
      super(Conv2d_Q_with_swish, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_sfp34 = quantize_sfp34(k=q_bit) 
      self.quantize_slfp34 = quantize_slfp34(k=q_bit)
      self.Kw = torch.tensor(Kw)
      self.Ka = torch.tensor(Ka)

    def forward(self, input, order=None):
      self.input_sfp = self.quantize_sfp34(input/self.Ka)
      input_swish = self.input_sfp * torch.sigmoid(self.input_sfp)
      self.input_q = self.quantize_slfp34(input_swish)
      self.weight_q = self.quantize_weight(self.weight/self.Kw) 
      self.output = F.conv2d(self.input_q, self.weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)*self.Ka*self.Kw
      return self.output
  return Conv2d_Q_with_swish


def linear_Q(q_bit, Kw, Ka):   
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, Kw=Kw, Ka=Ka, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)
      self.Kw = torch.tensor(Kw)
      self.Ka = torch.tensor(Ka)

    def forward(self, input):
      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.weight/self.Kw)
      self.bias_q = self.bias /self.Kw /self.Ka
      out =  F.linear(self.input_q, self.weight_q, self.bias_q)*self.Kw*self.Ka
      return out
  return Linear_Q

# ------------------------------------ bitch_size_scale---------------------------------------- #
"""  
    Function: Bitch_size_scaling, Each convolution kernel uses a scaling factor.
    Author  : Zhangshize
    Bias    : False
"""
def conv2d_Q_bitch_size_scaling(q_bit, Kw, Ka):
  class conv2d_Q_bitch_size_scaling(nn.Conv2d):  
    def __init__(self,in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, stride=1, padding=0, dilation=1, groups=1, bias=False):
      super(conv2d_Q_bitch_size_scaling, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)
      
      self.Kw =  torch.tensor(Kw).cuda()
      self.Kw =  self.Kw.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

      self.Ka = Ka
 
    def forward(self, input, order=None):
      self.layer_weight_q = torch.div(self.weight , self.Kw)
      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.layer_weight_q)

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.weight_q = self.weight_q.to(device)
      Ka_tentor = torch.tensor(self.Ka, device = 'cuda:0', requires_grad = True)

      self.output = F.conv2d(self.input_q, self.weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
      final_output = self.output * self.Kw.permute(1, 0, 2, 3) * Ka_tentor
           
      return final_output
      
  return conv2d_Q_bitch_size_scaling

"""  
  Function: Bitch_size_scaling, Each Fully connected group uses a scaling factor.
  Author  : Zhangshize
  Bias    : False
"""
def linear_Q_bitch_size_scaling(q_bit, Kw, Ka):   
  class linear_Q_bitch_size_scaling(nn.Linear):
    def __init__(self, in_features, out_features, Kw=Kw, Ka=Ka, bias = False):
      super(linear_Q_bitch_size_scaling, self).__init__(in_features, out_features, bias )
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)

      self.Kw =  torch.tensor(Kw).cuda()
      self.Kw =  self.Kw.unsqueeze(-1)

      self.Ka = Ka

    def forward(self, input):
      """ Each bitch size is scaled """
      self.layer_weight_q = torch.div(self.weight , self.Kw)

      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.layer_weight_q)

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.weight_q = self.weight_q.to(device)

      Ka_tentor = torch.tensor(self.Ka, device = 'cuda:0', requires_grad = True)

      out =  F.linear(self.input_q, self.weight_q, self.bias)

      final_output = out * self.Kw.permute(1, 0) * Ka_tentor

      return final_output
  return linear_Q_bitch_size_scaling

"""  
    Function: Bitch_size_scaling, Each convolution kernel uses a scaling factor.
    Author  : Zhangshize
    Bias    : True
"""
def conv2d_Q_bias_bitch_size_scaling(q_bit, Kw, Ka):
  class conv2d_Q_bias_bitch_size_scaling(nn.Conv2d):  
    def __init__(self,in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, stride=1, padding=0, dilation=1, groups=1, bias=True):
      super(conv2d_Q_bias_bitch_size_scaling, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)

      self.Kw = Kw
      self.Ka = Ka
 
    def forward(self, input, order=None):
      """ Each bitch size is scaled """
      for index, bitch_size_weight_max in enumerate(self.Kw):
          bitch_size_weight_q = torch.div(self.weight[index] , torch.tensor(bitch_size_weight_max)) # tentor division
          if index == 0:
            layer_weight_q = bitch_size_weight_q
            layer_weight_q = layer_weight_q.unsqueeze(0)
          else:
            bitch_size_weight_q = bitch_size_weight_q.unsqueeze(0)
            layer_weight_q = torch.cat((layer_weight_q, bitch_size_weight_q), dim=0)

      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(layer_weight_q) 

      Kw_tentor = torch.tensor(self.Kw, device = 'cuda:0',requires_grad=True)
      Ka_tentor = torch.tensor(self.Ka, device = 'cuda:0',requires_grad=True)

      self.bias_q =  self.bias / Kw_tentor / Ka_tentor

      self.output = F.conv2d(self.input_q, self.weight_q, self.bias_q, self.stride,
                    self.padding, self.dilation, self.groups)
      Kw_tentor = Kw_tentor.unsqueeze(0)
      dim_to_add = len(self.output.size()) - len(Kw_tentor.size())    # Determine the number of dimension extensions
      for _ in range(dim_to_add):                                     # Add dimensions to the tail of the Kw_tentor tensor
        Kw_tentor = Kw_tentor.unsqueeze(-1)                           # Add dimensions to the last dimension
      final_output = self.output * Kw_tentor * Ka_tentor              # ka * kw * [[ (a / ka) * (w / kw)] + B /(ka * kw)]
           
      return final_output
      
  return conv2d_Q_bias_bitch_size_scaling

"""  
  Function: Bitch_size_scaling, Each Fully connected group uses a scaling factor.
  Author  : Zhangshize
  Bias    : True
"""
def linear_Q_Bise_bitch_size_scaling(q_bit, Kw, Ka):   
  class linear_Q_Bise_bitch_size_scaling(nn.Linear):
    def __init__(self, in_features, out_features, Kw=Kw, Ka=Ka, bias=True):
      super(linear_Q_Bise_bitch_size_scaling, self).__init__(in_features, out_features, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)

      self.Kw = Kw
      self.Ka = Ka

    def forward(self, input):
      """ Each bitch size is scaled """
      for index, bitch_size_weight_max in enumerate(self.Kw):
          print('Bitch size index of the layer ',index)

          bitch_size_weight_q = torch.div(self.weight[index] , torch.tensor(bitch_size_weight_max)) # tentor division 

          if index == 0:
            layer_weight_q = bitch_size_weight_q
            layer_weight_q = layer_weight_q.unsqueeze(0)
          else:
            bitch_size_weight_q = bitch_size_weight_q.unsqueeze(0)
            layer_weight_q = torch.cat((layer_weight_q, bitch_size_weight_q), dim=0)

      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(layer_weight_q) 

      Kw_tentor = torch.tensor(self.Kw, device = 'cuda:0',requires_grad=True)
      Ka_tentor = torch.tensor(self.Ka, device = 'cuda:0',requires_grad=True)

      self.bias_q =  self.bias / Kw_tentor / Ka_tentor

      out =  F.linear(self.input_q, self.weight_q, self.bias_q)
      Kw_tentor = Kw_tentor.unsqueeze(0)
      dim_to_add = len(out.size()) - len(Kw_tentor.size())            # Determine the number of dimension extensions
      for _ in range(dim_to_add):                                     # Add dimensions to the tail of the Kw_tentor tensor
        Kw_tentor = Kw_tentor.unsqueeze(-1)                           # Add dimensions to the last dimension
      print(out.size())
      print(Kw_tentor.size())

      final_output = out * Kw_tentor * Ka_tentor                      # ka * kw * [[ (a / ka) * (w / kw)] + B /(ka * kw)]

      return final_output
  return linear_Q_Bise_bitch_size_scaling

def conv2d_Q_Stride(q_bit, Kw, Ka, stride):
  class conv2d_Q_Stride(nn.Conv2d):  
    def __init__(self, in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, 
    stride = stride, padding = 1, dilation = 1, groups = 1, bias = False):

      super(conv2d_Q_Stride, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.q_bit = q_bit                                       
      self.quantize_weight = weight_quantize_func(q_bit=q_bit) 
      self.quantize_act = act_quantize_func(q_bit=q_bit)       
      self.Kw = torch.tensor(Kw)                               
      self.Ka = torch.tensor(Ka)

    def forward(self, input, order=None):
      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.weight/self.Kw)
      self.output = F.conv2d(self.input_q, self.weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)*self.Ka*self.Kw
      return self.output
  return conv2d_Q_Stride
