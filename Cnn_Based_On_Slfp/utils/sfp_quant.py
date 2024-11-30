"""
Functional description: SLFP Quantization
Author: Xintong He
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def quantize_weight(k):  
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input

        elif k == 7:
            #out = input.clone()
            N = 8   #SFP<3,3>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs)
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)         
            mantissa_q = torch.round(mantissa * N) / N
            output_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs >= 15] = 15
            out = torch.mul(sign, output_abs)

        elif k == 8:
            N = 16  # SLFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_log = torch.round(torch.log2(mantissa)*N)/N # ???
            output_abs = pow(2, (exponent + mantissa_log))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10 # to avoid inf
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs > 15.32165] = 15.32165
            out = torch.mul(sign, output_abs) 
        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE, do nothing in backward
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

def quantize_sfp34(k):  
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input

        if k == 8:
            #out = input.clone()
            N = 16   #SFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            output_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs >= 15.5] = 15.5
            out = torch.mul(sign, output_abs)

        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE, do nothing in backward
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

def quantize_slfp34(k):  
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input

        if k == 8:   # SLFP<3,4>
            N = 16   # SFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            mantissa_log = torch.round(torch.log2(mantissa_q)*N)/N  # SLFP<3,4>, log converter
            output_abs = pow(2, (exponent + mantissa_log))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs > 15.32165] = 15.32165 # 0 111 1111
            out = torch.mul(sign, output_abs)

        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE, do nothing in backward
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

def quantize_act(k): 
  class qfn(torch.autograd.Function): 
    @staticmethod             
    def forward(ctx, input):
        if k == 32:
            out = input

        elif k == 7:  # SFP<3,3>
            N = 8 
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            output_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs >= 15] = 15
            out = torch.mul(sign, output_abs)

        elif k == 8: # SLFP<3,4>
            N = 16   # SFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            mantissa_log = torch.round(torch.log2(mantissa_q)*N)/N  # SLFP<3,4>, log converter
            output_abs = pow(2, (exponent + mantissa_log))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs > 15.32165] = 15.32165 # 0 111 1111
            out = torch.mul(sign, output_abs)
        return out

    @staticmethod
    def backward(ctx, grad_output): 
      grad_input = grad_output.clone()
      return grad_input
  return qfn().apply


def quantize_act_with_swish(k): 
  class qfn(torch.autograd.Function): 
    @staticmethod             
    def forward(ctx, input):
        if k == 32:
            out = input * torch.sigmoid(input)

        elif k == 8: # SLFP<3,4>
            N = 16   
            # SFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            out_sfp_abs = torch.clone(input_abs) #init out_sfp_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            out_sfp_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            out_sfp_abs[input_abs < 0.0625] = 1e-10
            out_sfp_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            out_sfp_abs[input_abs >= 15] = 15
            out_sfp = torch.mul(sign, out_sfp_abs)
            # swish
            out_swish = out_sfp * torch.sigmoid(out_sfp)
            # SLFP<3,4>
            sign = torch.sign(out_swish)
            out_swish_abs = torch.abs(out_swish)
            output_abs = torch.clone(out_swish_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(out_swish_abs))
            mantissa = out_swish_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            mantissa_log = torch.round(torch.log2(mantissa_q)*N)/N  # SLFP<3,4>, log converter
            output_abs = pow(2, (exponent + mantissa_log))
            # subnormal
            output_abs[out_swish_abs < 0.0625] = 1e-10
            output_abs[(out_swish_abs >= 0.0625) & (out_swish_abs < 0.125)] = 0.125
            # large number
            output_abs[out_swish_abs > 15.32165] = 15.32165 # 0 111 1111
            out = torch.mul(sign, output_abs)

        return out

    @staticmethod
    def backward(ctx, grad_output): 
      grad_input = grad_output.clone()
      return grad_input
  return qfn().apply

def quantize_layerout(k):  
  class qfn(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input
        elif k <= 8:
            N = 16       #SFP<4,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            output_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            output_abs[input_abs < 2^(-8)] = 1e-10
            output_abs[(input_abs >= 2^(-8)) & (input_abs < 2^(-7))] = 2^(-7)
            # large number
            output_abs[input_abs >= 248] = 248 # 0 1111 1111 = 2^(15-8) * (1 + 15/16) = 248
            out = torch.mul(sign, output_abs)
        return out

    @staticmethod
    def backward(ctx, grad_output):     
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

class weight_quantize_func(nn.Module):    
  def __init__(self, q_bit):
    super(weight_quantize_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit      
    self.quantize = quantize_weight(k=q_bit) 

  def forward(self, x):

    if self.q_bit == 32:
      weight_q = x
    elif self.q_bit == 8 or self.q_bit == 7:
      weight_q = self.quantize(x)  
    return weight_q

class act_quantize_func(nn.Module): 
  def __init__(self, q_bit):
    super(act_quantize_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit                  
    self.quantize = quantize_act(k=q_bit) 

  def forward(self, x):
    if self.q_bit == 32:
      act_q = x
    elif self.q_bit == 8 or self.q_bit == 7:
      act_q = self.quantize(x)  
    return act_q

class act_quantize_with_swish_func(nn.Module): 
  def __init__(self, q_bit):
    super(act_quantize_with_swish_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit                  
    self.quantize = quantize_act_with_swish(k=q_bit) 

  def forward(self, x):
    if self.q_bit == 32:
      act_q = x * torch.sigmoid(x)
    elif self.q_bit == 8 or self.q_bit == 7:
      act_q = self.quantize(x)  
    return act_q

class act_quantize_with_gelu_func(nn.Module): 
  def __init__(self, q_bit):
    super(act_quantize_with_gelu_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit                  
    self.quantize_sfp34 = quantize_sfp34(k=q_bit) 
    self.quantize_slfp34 = quantize_slfp34(k=q_bit)

  def forward(self, x):
    if self.q_bit == 32:
      act_q =  0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))
    elif self.q_bit == 8 :
      act_sfp = self.quantize_sfp34(x)
      act_gelu = 0.5 * act_sfp * (1 + torch.tanh(0.7978845608 * (act_sfp + 0.044715 * torch.pow(act_sfp, 3))))
      act_q = self.quantize_slfp34(act_gelu)

    return act_q

class layerout_quantize_func(nn.Module): 
  def __init__(self, q_bit):
    super(layerout_quantize_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit 
    self.quantize = quantize_layerout(k = q_bit) 

  def forward(self, x):
    if self.q_bit == 32:
      out_q = x
    elif self.q_bit == 8 or self.q_bit == 7:
      out_q = self.quantize(x)  
    return out_q

if __name__ == '__main__':
  x = torch.tensor([-10, 0.01, 0.06251, 0.125, 0.1, 0.2, 1, 15])
  z = quantize_act_with_swish(8)(x)
  print (x)
  print (z)
