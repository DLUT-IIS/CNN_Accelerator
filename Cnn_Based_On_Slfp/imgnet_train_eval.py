"""
Project Name: ImageNet-1k Training and Evaluation
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
import math
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import ImageFile
import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import psutil # monitor CPU utilization
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from tqdm import tqdm
from nets_imgnet.mobilenetv1 import *
from nets_imgnet.resnet50 import *
from nets_imgnet.alexnet import *
from nets_imgnet.vgg16 import *
from nets_imgnet.mobilenetv2 import *
from nets_imgnet.inception_v3 import *
from nets_imgnet.squeezenet1_0 import *
from utils.preprocessing import *
from utils.optimizer import *
from torch.optim import optimizer
from torch.optim.optimizer import required
import matplotlib.ticker as ticker
cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True
#import globals

# Training settings
parser = argparse.ArgumentParser(description='SLFP train and finetune pytorch implementation')
parser.add_argument('--optimizer', type=str, default='SGD')  
parser.add_argument('--net', type=str, default='mobilenetv1')  
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/opt/datasets/imagenet-1k')
# parser.add_argument('--data_dir', type=str, default='/workspaces/pytorch-dev/datasets')
parser.add_argument('--log_name', type=str, default='imgnet-1k')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pre_reference', action='store_true', default=False)
parser.add_argument('--retrain', action='store_true', default=False)
parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--all_validate', action='store_true', default=False)        
parser.add_argument('--activation_function', action='store_true', default='relu')
parser.add_argument('--Qbits', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-5) # default 1e-5
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--train_batch_size', type=int, default=32)#256
parser.add_argument('--eval_batch_size', type=int, default=16) #100
parser.add_argument('--max_epochs', type=int, default=2)
parser.add_argument('--log_interval', type=int, default=500) #10, 500
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=5) #20
parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

if not cfg.cluster:
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu

def Initializes_command_data_model():
  """------------------------ Data loading -----------------------------"""
  traindir = os.path.join(cfg.data_dir, 'train')         # train data, 1000 categories, 100 images per category
  # traindir = os.path.join(cfg.data_dir, 'train_200')   # train data, 1000 categories, 200 images per category
  # traindir = os.path.join(cfg.data_dir, 'train_1k')    # train data, 1000 categories, 1300 images per category

  valdir = os.path.join(cfg.data_dir, 'val')             # test  data

  train_dataset = datasets.ImageFolder(traindir, imgnet_transform(is_training=True))

  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.train_batch_size,
                                             shuffle=True,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True)
  
  val_dataset = datasets.ImageFolder(valdir, imgnet_transform(is_training=False))
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=cfg.eval_batch_size,
                                           shuffle=False,
                                           num_workers=cfg.num_workers,
                                           pin_memory=True)

  print("=> creating model", cfg.net, "..." )
  print(" learning rate = ", cfg.lr)
  print(" weight_decay = ", cfg.wd)
  print(" precision = ", cfg.Qbits, "bits")

  """------------------------ activation_function -----------------------------"""
  if cfg.activation_function == "relu" :
    print("relu")
  elif cfg.activation_function == "swish":
    print("swish")
  elif cfg.activation_function == "mish":
    print("mish")
  elif cfg.activation_function == "gelu":
    print("gelu")  
  elif cfg.activation_function == "prelu":
    print("prelu")  

  """------------------------ create model -----------------------------"""
  if cfg.net == "inceptionv3":
    model = inception_v3().cuda()
    pretrain_dir = './ckpt/imgnet-1k/inception_v3.pth'

  if cfg.net == "mobilenetv1":
    model = MobileNetV1_Q(ch_in=3, qbit=cfg.Qbits, pre_reference = cfg.pre_reference, activation_function = cfg.activation_function).cuda()
    # pretrain_dir = './ckpt/imgnet-1k/mobnetv1_m1_base.pth'                                   # paper: 68.786%
    # pretrain_dir = './ckpt/imgnet-1k/20241023_mobilenet_imgnet_1e6_bitch_size_68.068.pth'    # val-8bits:68.068%  
    # pretrain_dir = './ckpt/imgnet-1k/20241023_mobilenet_imgnet_1e6_bitch_size_68.068.pth'    # val-8bits:68.068%
    # pretrain_dir = './ckpt/imgnet-1k/20241119_mobilenetv1_imgnet_1e5_channel_SGDNW_slfp.pth' # val-8bits:68.064% SGDNW 

  if cfg.net == "mobilenetv2":
    model = mobilenetv2( qbit=cfg.Qbits, pre_reference = cfg.pre_reference, activation_function = cfg.activation_function).cuda()
    # model = mobilenetv2().cuda()
    # ------------------------ pytorch baseline : 71.878% 72.154% ----------------------------#
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2-c5e733a8.pth'                              # 72.85% 
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2_0.1-7d1d638a.pth'                          # X
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2_0.5-eaa6f9ad.pth'                          # X
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2_1.0-0c6065bc.pth'                          # 72.186%
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2_128x128-fd66a69d.pth'                      # 68.186%
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2_160x160-64dc7fa1.pth'                      # 71.39%
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2_192x192-e423d99e.pth'                      # 72.254% 
    # pretrain_dir = './ckpt/imgnet-1k/mobilenetv2_192x192-e423d99e.pth'                      # 72.254% 
    # pretrain_dir = './ckpt/imgnet-1k/20241118_mobilenetv2_imgnet_1e5_layer_SGDNW_slfp.pth'  # SGDNW 71.872%

  elif cfg.net == "resnet":
    model = ResNet50(qbit = cfg.Qbits, pre_reference = cfg.pre_reference).cuda()
    # pretrain_dir = './ckpt/imgnet-1k/resnet-50.pth'                                         # 76.148% 
    # pretrain_dir = './ckpt/imgnet-1k/resnet_8perclass_slfp34_76.pth'                        # 75.864%
    # pretrain_dir = './ckpt/imgnet-1k/20241120_resnet_imgnet_1e5_channel_SGDNW_slfp.pth'     # 76.276%                                                                                             
    # pretrain_dir = './ckpt/imgnet-1k/20241120_resnet_imgnet_1e5_channel_SGD_WD0_slfp.pth'   # 76.352%

  elif cfg.net == "vgg16":
    model = vgg16_bn(qbit = cfg.Qbits).cuda()
    pretrain_dir = './ckpt/imgnet-1k/vgg16_bn.pth'
  
  elif cfg.net == "alexnet":
    model = alexnet(qbit = cfg.Qbits).cuda()
    pretrain_dir = './ckpt/imgnet-1k/alexnet.pth'

 
  """------------------------ optionally resume from a checkpoint -----------------------------"""
  if cfg.pretrain:
    model.load_state_dict(torch.load(pretrain_dir), False)

  """------------------------ define loss function (criterion) and optimizer -----------------------------"""
  if cfg.optimizer == "Adam" :
    print("Adam")
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  elif cfg.optimizer == "NormalSGD":
    print("NormalSGD")
    optimizer = NormalSGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "DSGD":
    print("DSGD")
    optimizer = DSGD(model.parameters(), qbit = cfg.Qbits, lr = cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "SSGD":
    print("SSGD")
    optimizer = SSGD(model.parameters(), qbit = cfg.Qbits, lr = cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "SGD":
    print("SGD")
    optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "SGD_NW":
    print("SGD_NW")
    optimizer = SGD_NW(model.parameters(), qbit = cfg.Qbits, lr = cfg.lr, momentum=0.9, weight_decay=cfg.wd)

  # lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.3)
  criterion = nn.CrossEntropyLoss().cuda()

  summary_writer = SummaryWriter(cfg.log_dir)

  return train_loader, val_dataset, val_loader, pretrain_dir, model, criterion, optimizer, summary_writer
   
def train(epoch):
  model.train()
  start_time = time.time()
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    # compute output
    output = model(inputs.cuda())
    loss = criterion(output, targets.cuda())

    # compute gradient and do SGD step
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

def validate(epoch):
  # switch to evaluate mode
  model.eval()
  top1 = 0      # The category with the highest probability of prediction
  top5 = 0      # The top five probabilities are predicted to be correct
  if (cfg.all_validate == True ):
    num_samples = len(val_dataset)
  else:
    num_samples = 100
  
  # with tqdm(total=num_samples) as pbar:
  for i, (inputs, targets) in enumerate(val_loader):
      if i * cfg.eval_batch_size >= num_samples:
        break 

      targets = targets.cuda()
      input_var = inputs.cuda()

      # compute output
      output = model(input_var)

      # measure accuracy and record loss
      _, pred = output.data.topk(5, dim=1, largest=True, sorted=True)
      pred = pred.t()
      correct = pred.eq(targets.view(1, -1).expand_as(pred))

      top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
      top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
      #pbar.update(cfg.eval_batch_size)

  top1 *= 100 / num_samples
  top5 *= 100 / num_samples
  print('%s------------------------------------------------------ '
        'Precision@1: %.2f%%  Precision@1: %.2f%%\n' % (datetime.now(), top1, top5))
  top1_all.append(top1)
  top5_all.append(top5)

  summary_writer.add_scalar('Precision@1', top1, epoch)
  summary_writer.add_scalar('Precision@5', top5, epoch)
  return top1, top5

def test(epoch): 
  # pass
  model.eval() 
  correct = 0
  for batch_idx, (inputs, targets) in enumerate(val_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    correct += predicted.eq(targets.data).cpu().sum().item()

  acc = 100. * correct / len(val_dataset)
  print('%s------------------------------------------------------ '
        'Precision@1: %.2f%% \n' % (datetime.now(), acc))
  acc_data.append(acc)

  summary_writer.add_scalar('Precision@1', acc, global_step=epoch)      
 

if __name__ == '__main__':  
    print(torch.cuda.is_available())

    train_loader, val_dataset, val_loader, pretrain_dir, model, criterion, optimizer, summary_writer = Initializes_command_data_model()

    if (cfg.pre_reference == True):

      weight_folder_path = '/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/resnet_layer_weight_max.txt'
      input_folder_path  = '/workspaces/pytorch-dev/Cnn_Based_On_Slfp/max_act_wgt/imgnet/resnet_layer_input_max.txt'

      total_images = 1000  # Only 1000 images were used, and the entire dataset was not used to calculate the maximum and minimum values

      # ------------- Bitch size ------------- #
      accuracy, max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_layer_weights_list = get_scale_factor_moebz(model, val_loader, total_images)    # batch size
      put_scale_factor_bitch_size_txt(max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_layer_weights_list, weight_folder_path, input_folder_path)

      # ------------- Layer ------------- #
      # accuracy, max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_layer_weights_list = get_scale_factor_moabz(model, val_loader, total_images)  # layer
      # put_scale_factor_layer_txt(max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_layer_weights_list, weight_folder_path, input_folder_path)

    # main loop
    acc_data = []
    top1_all = []
    top5_all = []
    
    acc_max = 0
    for epoch in range(1, cfg.max_epochs):
        print('epoch = ',epoch)

        if (cfg.retrain == True):
            train(epoch)

        validate(epoch)
        print("top1:", top1_all)
        print("top5:", top5_all) 
        print("max acc :", max(top1_all))

        if (cfg.save_model == True and max(top1_all)> acc_max):
            acc_max = max(top1_all)
            torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'best weight model.pth'))
            print("max acc :", acc_max)
            print("saving model....")




