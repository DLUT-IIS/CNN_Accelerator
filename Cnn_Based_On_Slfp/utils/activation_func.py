import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def STLFunction():  
  class stl(torch.autograd.Function):  
    @staticmethod             
    def forward(ctx, x):          
        out = torch.where(torch.abs(x) <= 1, x, torch.sign(x) * (torch.log(torch.abs(x)) + 1))
        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE,do nothing in backward (quantized)
      grad_input = grad_output.clone()  
      grad_input = torch.where(torch.abs(grad_output) <= 1, 1, 1/torch.abs(grad_output)) * grad_output
      return grad_input
  return stl().apply

class Identity(nn.Module):
    def forward(self, x):
        return x

class STL(nn.Module):    
  def __init__(self):
    super(STL, self).__init__()
    self.stl = STLFunction()

  def forward(self, x):
    stlout = self.stl(x)   
    return stlout

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

