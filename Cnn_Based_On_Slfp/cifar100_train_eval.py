"""
Project Name: Cifar100 Training and Evaluation
First generation author: Xintong He
Second generation author: Shize Zhang
Perfect and improve: Shize Zhang

Project Description:
This is a PyTorch implementation for training and evaluating on the Imagenet dataset.
It includes implementations of quantized MobileNet V1, SqueezeNet, AlexNet, and ResNet-50.
and their revised version by re-selecting the non-linear activation function.

8-bit SLFP and 7-bit SFP quantization based on max-scaling are implemented.

Dependencies:
- Python 3.6+
- PyTorch 1.0+
- torchvision 0.2.2+

Installation and Running:
1. Clone this repository
2. Run the code: python ./imgnet_train_eval.py --Qbits <bit width> --net <net name> ...
Arguments are optional, please refer to the argparse settings in the code.
The default setting is 32-bit floating point reference of mobilenetv1 on Imagenet.
"""

import os
import time
import argparse
import torch.optim as optim
from datetime import datetime
import torch
import torchvision
import torch.backends.cudnn as cudnn
# import globals
from tensorboardX import SummaryWriter
from torch.optim.optimizer import required
from utils.sfp_quant import *
from utils.optimizer import *
from utils.scale_factor import *
from utils.preprocessing import *
from nets_cifar.shufflenet_v2 import *
from nets_cifar.mobilenetv1 import *
from nets_cifar.vgg16 import *

cudnn.benchmark = True
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=8, suppress=True)

# Training settings
parser = argparse.ArgumentParser(description='SLFP reference and retrain, pytorch implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')           # Data set location
parser.add_argument('--log_name', type=str, default='cifar-100')
parser.add_argument('--cifar', type=int, default=100)

# running mode
parser.add_argument('--retrain', action='store_true', default=False)    # retraining
parser.add_argument('--save_model', action='store_true', default=False)
'''
    Pre-reasoning, in order to quantify, the scaling factor needs to be obtained before quantization, 
    the maximum value of input and output of each layer is calculated, the output of two files, the maximum 
    value of input and output of each layer and the maximum weight.

    The following k is obtained according to the maximum weight, and k is the scaling factor. Before quantizing
    the convolutional output value of each layer, it is scaled to the range that slfp can represent -15~+15 and then quantized.
'''
parser.add_argument('--pre_reference', action='store_true',            # 
                    default=False)  
parser.add_argument('--pretrain', action='store_true', default=False)  # True：use pretrained parameters  False：random seed
parser.add_argument('--optimizer', type=str, default='SSGD')           # Selection optimizer
parser.add_argument('--net', type=str, default='mobilenet')            # Selection net

# training hyper-parameters - Parameter selection for quantization and training
parser.add_argument('--Qbits', type=int, default=32)                   # 7:SFP<3,3>, 8:SLFP<3,4>, 32:FP32
parser.add_argument('--lr', type=float, default=0.00001)               # learning rate,  0.0001
parser.add_argument('--wd', type=float, default=0)                     # weight decay ,  1e-4

parser.add_argument('--num', type=int, default=0)
parser.add_argument('--train_batch_size', type=int, default=256)       # default=256
parser.add_argument('--eval_batch_size', type=int, default=128)        # default=128
parser.add_argument('--max_epochs', type=int, default=1)               # Specify the maximum number of rounds (maximum number of training cycles) of the model  
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--use_gpu', type=str, default='0')                # GPU 0/1
parser.add_argument('--num_workers', type=int, default=5)              # How many processors are parallelized to handle this task

cfg = parser.parse_args()  

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)   

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu

def Initializes_command_data_model():
  # -------------------------- #
  # ------------- Selective data set ------------- #
  if cfg.cifar == 10:
    print('training CIFAR-10 !')
    dataset = torchvision.datasets.CIFAR10
  elif cfg.cifar == 100:
    print('training CIFAR-100 !')
    dataset = torchvision.datasets.CIFAR100
  else:
    assert False, 'dataset unknown !'

  print('==> Preparing data ..')
  # ------------- Training data sets and data loaders ------------- #
  train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                          transform=cifar_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)

  eval_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                         transform=cifar_transform(is_training=False))

  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)
  

  num_samples = len(train_loader) * cfg.train_batch_size
  print("Total number of samples in train_loader:", num_samples)

  # model
  print("=> creating model", cfg.net, "..." )

    
  if cfg.net == "mobilenet":
    model = MobileNetV1_Q(ch_in=3, qbit=cfg.Qbits,pre_reference = cfg.pre_reference).cuda()
    pretrain_dir = './ckpt/cifar-100/mobnet-fp32-newmodel-60.64.pth'                     # fp32
    # pretrain_dir = './ckpt/cifar-100/mobilenet_20241019_0.00001_bitchsize_60.85.pth'   # slfp
    # pretrain_dir = './ckpt/cifar-100/mobilenet_20241021_0.00001_layer_60.57.pth'

  elif cfg.net == "mobilenet_swish":
    model = MobileNetV1_swish(ch_in=3, qbit=cfg.Qbits).cuda()
    pretrain_dir = './ckpt/cifar-100/mobnet-swish-61.8.pth'

  elif cfg.net == "vgg16":
    model = VGG16_Q(qbit=cfg.Qbits).cuda()
    pretrain_dir = './ckpt/cifar-100/vgg_relu_fp32_72.40.pth'

  elif cfg.net == "vgg16_gelu":
    model = VGG16_gelu(qbit=cfg.Qbits).cuda()
    #pretrain_dir = './ckpt/cifar-100/vgg16_gelu73.39.pth'
    pretrain_dir = './ckpt/cifar-100/vgg16_gelu0_tmp.pth'  #72.82
  
  # optimizer
  if cfg.optimizer == "SSGD" :
    print("optimizer : SSGD")
    optimizer = SSGD(model.parameters(), qbit = cfg.Qbits, lr = cfg.lr,  momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "DSGD":       # unused
    print("optimizer : DSGD")
    optimizer = DSGD(model.parameters(), qbit = cfg.Qbits, lr = cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "CustomSGD":  # unused
    print("optimizer : CustomSGD")
    optimizer = CustomSGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  elif cfg.optimizer == "RMSprop":
    optimizer = torch.optim.RMSprop(model.parameters(), cfg.lr)
  elif cfg.optimizer == "SGD":
    print("optimizer : SGD")
    optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "SGD_NW":
    print("SGD_NW")
    optimizer = SSGD(model.parameters(), qbit = cfg.Qbits, lr = cfg.lr, momentum=0.9, weight_decay=cfg.wd)


  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [75,85,100], gamma=0.1)

  criterion = torch.nn.CrossEntropyLoss().cuda()

  summary_writer = SummaryWriter(cfg.log_dir)

  return train_loader, eval_dataset, eval_loader, pretrain_dir, model, lr_schedu, criterion, optimizer, summary_writer

# Training
def train(epoch):

  print('\nEpoch: %d' % epoch)

  model.train()
  start_time = time.time()

  for batch_idx, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs.cuda())
    loss = criterion(outputs, targets.cuda())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % cfg.log_interval == 0:
      step = len(train_loader) * epoch + batch_idx
      duration = time.time() - start_time

      print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
            (datetime.now(), epoch, batch_idx, loss.item(),
              cfg.train_batch_size * cfg.log_interval / duration))

      start_time = time.time()
      summary_writer.add_scalar('cls_loss', loss.item(), step)
      summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

def test(epoch, eval_dataset):
  model.eval() 
  correct = 0
  for batch_idx, (inputs, targets) in enumerate(eval_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    correct += predicted.eq(targets.data).cpu().sum().item()

  acc = 100. * correct / len(eval_dataset)
  print('%s------------------------------------------------------ '
        'Precision@1: %.2f%% \n' % (datetime.now(), acc))
  acc_data.append(acc)

  summary_writer.add_scalar('Precision@1', acc, global_step=epoch)

def get_scale_factor(model, data_loader, total_images): # get Ka, Kw
    model.eval()
    correct = 0
    count = 0
    layer_inputs  = {}  # store inputs for each layer
    layer_outputs = {} 
    layer_weights = {} 
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
    
        # Record inputs and outputs for each layer
        model.reset_layer_inputs_outputs()
        model.reset_layer_weights()
        outputs = model(inputs)
        # Get inputs and outputs for each layer
        current_layer_inputs = model.get_layer_inputs()
        current_layer_outputs = model.get_layer_outputs()
        current_layer_weights = model.get_layer_weights()
        #print(inputimg.shape)
        
        for idx, input_tensor in current_layer_inputs.items():
            if idx not in layer_inputs:
                layer_inputs[idx] = []
            layer_inputs[idx].append(input_tensor.detach().cpu())
            
        for idx, output_tensor in current_layer_outputs.items():
            if idx not in layer_outputs:
                layer_outputs[idx] = []
            layer_outputs[idx].append(output_tensor.detach().cpu())

        for idx, output_tensor in current_layer_weights.items():
            if idx not in layer_weights:
                layer_weights[idx] = []
            layer_weights[idx].append(output_tensor.detach().cpu())

        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

        count += len(inputs)     
        print(inputs)                                     
        if count >= total_images:
            break
        
    ''' Calculate the maximum absolute values for layer inputs and outputs '''
    max_abs_layer_inputs = {}
    max_abs_layer_outputs = {}
    max_abs_layer_weights = {}

    for idx, inputs_list in layer_inputs.items():
        max_abs_input = torch.max(torch.abs(torch.cat(inputs_list, dim=0)))
        max_abs_layer_inputs[idx] = max_abs_input.item()
        
    for idx, outputs_list in layer_outputs.items():
        max_abs_output = torch.max(torch.abs(torch.cat(outputs_list, dim=0)))
        max_abs_layer_outputs[idx] = max_abs_output.item()

    for idx, weights_list in layer_weights.items():
        max_abs_output = torch.max(torch.abs(torch.cat(weights_list, dim=0)))
        max_abs_layer_weights[idx] = max_abs_output.item()

    acc = 100. * correct / total_images

    print('%s------------------------------------------------------ '
          'Precision@1: %.2f%% \n' % (datetime.now(), acc))

    return acc, max_abs_layer_inputs, max_abs_layer_outputs, max_abs_layer_weights


if __name__ == '__main__':
  print(torch.cuda.is_available())
  train_loader, eval_dataset, eval_loader, pretrain_dir, model, lr_schedu, criterion, optimizer, summary_writer = Initializes_command_data_model()

  if cfg.pretrain:
    model.load_state_dict(torch.load(pretrain_dir), False)

  if cfg.pre_reference:

    input_folder_path = '/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/cifar/bitch_size_input_max.txt'
    total_images = 1000  # Only 1000 images were used, and the entire dataset was not used to calculate the maximum and minimum values

    #------- 1. every bitch size = Set of channels , Ignore the first layer of standard convolution-------#
    weight_folder_path = '/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/cifar/bitch_size_weight_max.txt'
    accuracy, max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_layer_weights_list = get_scale_factor_moebz(model, eval_loader, total_images)
    put_scale_factor_txt(max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_layer_weights_list, weight_folder_path, input_folder_path)
    
  # main loop
  acc_data = [] 
  acc_max = 0
  print("Qbits: ", cfg.Qbits)
  print("lr: ", cfg.lr)
  print("wd: ", cfg.wd)
  for epoch in range(cfg.max_epochs):
    optimizer.step()
    lr_schedu.step()

    if (cfg.retrain == True):
      train(epoch)

    test(epoch, eval_dataset)
    print(acc_data) 
    print("max acc :", max(acc_data))
    if (cfg.save_model == True and max(acc_data)> acc_max):
      acc_max = max(acc_data)
      # torch.save(model.state_dict(), f'./ckpt/cifar-100/{cfg.net}_20241021_0.00001_layer.pth')
      torch.save(model.state_dict(), f'./ckpt/cifar-100/2.pth')

      # torch.save(model.module.state_dict(), os.path.join(cfg.ckpt_dir, '123.pth'))
      print(cfg.net,cfg.num)
      print("max acc :", acc_max)
      print("saving model....")

  summary_writer.close()


  