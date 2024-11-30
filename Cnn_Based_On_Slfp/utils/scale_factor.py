"""
    File function: Get scale factor.
    Author: Zhangshize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

def get_scale_factor_moebz_improved(model, data_loader, total_images): # get Ka, Kw
    ''' Function:   maximum of every bitch sizes, 
                    The scaling factor for extracting the standard convolution layer is the maximum value of each convolution kernel!'''
    # In order to ensure the stability of the model's performance during the validation or testing phase, 
    # some specific tricks in the training process are turned off in order to get a more accurate model evaluation
    model.eval()       
    correct = 0
    count = 0
    layer_inputs  = {}  # store inputs for each layer
    layer_outputs = {} 
    layer_weights = {} 

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # print(batch_idx)
        " Gets the input, output and weight of every layer "
        inputs, targets = inputs.cuda(), targets.cuda()
        model.reset_layer_inputs_outputs()
        model.reset_layer_weights()
        
        outputs = model(inputs)
        # Get inputs and outputs for each layer
        current_layer_inputs = model.get_layer_inputs()
        current_layer_outputs = model.get_layer_outputs()
        current_layer_weights = model.get_layer_weights()
        
        # print(len(current_layer_inputs[0]))
        
        " Gets the input of every layer "
        for idx, input_tensor in current_layer_inputs.items():
            if idx not in layer_inputs:
                layer_inputs[idx] = []
            layer_inputs[idx].append(input_tensor.detach().cpu())
        
        " Gets the output of every layer "
        for idx, output_tensor in current_layer_outputs.items():
            if idx not in layer_outputs:
                layer_outputs[idx] = []
            layer_outputs[idx].append(output_tensor.detach().cpu())

        " Gets the weight of every layer "
        if batch_idx == 0: 
            for idx, output_tensor in current_layer_weights.items():
                if idx not in layer_weights:
                    layer_weights[idx] = []
                layer_weights[idx].append(output_tensor.detach().cpu())        

        # Only precision analysis is done, and scaling quantization is not involved. The number of samples predicted correctly by the model is added to the variable correct 
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

        count += len(inputs)
        if count >= total_images:
            break

    " Accuracy of calculation "
    acc = 100. * correct / total_images
    print('Precision@1: %.2f%% \n' % (acc))

    """ Extract the each layer max input and output """
    max_abs_layer_inputs  = []
    max_abs_layer_outputs = []
    """ Extract the batch size of each layer W """
    max_abs_batch_size_weights = [[] for _ in range(len(layer_weights))]


    """ Extract the maximum value of the input for each layer """
    for idx, inputs_list in layer_inputs.items():
        max_abs_input = torch.max(torch.abs(torch.cat(inputs_list, dim=0))) # torch.cat ：merge bitch size 
        max_abs_layer_inputs.append(max_abs_input.item()) 
        
    """ Extract the maximum value of the output for each layer """
    for idx, outputs_list in layer_outputs.items():
        max_abs_output = torch.max(torch.abs(torch.cat(outputs_list, dim=0)))
        max_abs_layer_outputs.append(max_abs_output.item())

    """ Extract the maximum value of the output for each layer """
    for weights_layer_idx, weights_layer_tensor in layer_weights.items():  
        # print(weights_layer_idx)
        if (weights_layer_idx == 0):                        # Standard convolution
            for  weights_layer_dict  in weights_layer_tensor:     
                for weights_batch_size_idx, weights_batch_size_list in enumerate(weights_layer_dict):   
                    max_abs_batch_size_weights[weights_layer_idx].append([])
                    for weights_convolution_kernel_idx, weights_convolution_kernel_list in enumerate(weights_batch_size_list):       
                        max_abs_weights = torch.max(torch.abs(weights_convolution_kernel_list))      
                        max_abs_batch_size_weights[weights_layer_idx][weights_batch_size_idx].append(max_abs_weights.item())              
        else:                                               # dw, pw
            for  weights_layer_dict  in weights_layer_tensor:    
                for weights_batch_size_idx, weights_batch_size_list in enumerate(weights_layer_dict):     
                    # print(weights_batch_size_idx)          
                    max_abs_weights = torch.max(torch.abs(weights_batch_size_list))      
                    max_abs_batch_size_weights[weights_layer_idx].append(max_abs_weights.item())  
    
    return acc, max_abs_layer_inputs, max_abs_layer_outputs, max_abs_batch_size_weights


def get_scale_factor_moebz(model, data_loader, total_images): # get Ka, Kw
    ''' Function: maximum of every bitch sizes'''
    # In order to ensure the stability of the model's performance during the validation or testing phase, 
    # some specific tricks in the training process are turned off in order to get a more accurate model evaluation
    model.eval()       
    correct = 0
    count = 0
    layer_inputs  = {}  # store inputs for each layer
    layer_outputs = {} 
    layer_weights = {} 

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        print(batch_idx)
        " Gets the input, output and weight of every layer "
        inputs, targets = inputs.cuda(), targets.cuda()
        model.reset_layer_inputs_outputs()
        model.reset_layer_weights()
        
        outputs = model(inputs)
        # Get inputs and outputs for each layer
        current_layer_inputs = model.get_layer_inputs()
        current_layer_outputs = model.get_layer_outputs()
        current_layer_weights = model.get_layer_weights()
        
        # print(len(current_layer_inputs[0]))
        
        " Gets the input of every layer "
        for idx, input_tensor in current_layer_inputs.items():
            if idx not in layer_inputs:
                layer_inputs[idx] = []
            layer_inputs[idx].append(input_tensor.detach().cpu())
        
        " Gets the output of every layer "
        for idx, output_tensor in current_layer_outputs.items():
            if idx not in layer_outputs:
                layer_outputs[idx] = []
            layer_outputs[idx].append(output_tensor.detach().cpu())

        " Gets the weight of every layer "
        if batch_idx == 0: 
            for idx, output_tensor in current_layer_weights.items():
                if idx not in layer_weights:
                    layer_weights[idx] = []
                layer_weights[idx].append(output_tensor.detach().cpu())        

        # Only precision analysis is done, and scaling quantization is not involved. The number of samples predicted correctly by the model is added to the variable correct 
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

        count += len(inputs)
        if count >= total_images:
            break

    " Accuracy of calculation "
    acc = 100. * correct / total_images
    print('Precision@1: %.2f%% \n' % (acc))

    """ Extract the each layer max input and output """
    max_abs_layer_inputs  = []
    max_abs_layer_outputs = []
    """ Extract the batch size of each layer W """
    max_abs_batch_size_weights = [[] for _ in range(len(layer_weights))]


    """ Extract the maximum value of the input for each layer """
    for idx, inputs_list in layer_inputs.items():
        max_abs_input = torch.max(torch.abs(torch.cat(inputs_list, dim=0))) # torch.cat ：merge bitch size 
        max_abs_layer_inputs.append(max_abs_input.item()) 
        
    """ Extract the maximum value of the output for each layer """
    for idx, outputs_list in layer_outputs.items():
        max_abs_output = torch.max(torch.abs(torch.cat(outputs_list, dim=0)))
        max_abs_layer_outputs.append(max_abs_output.item())

    """ Extract the maximum value of the output for each layer """
    for weights_layer_idx, weights_layer_tensor in layer_weights.items():   
        for  weights_layer_dict  in weights_layer_tensor:         
            for weights_batch_size_idx, weights_batch_size_list in enumerate(weights_layer_dict):     
                # print(weights_batch_size_idx)          
                max_abs_weights = torch.max(torch.abs(weights_batch_size_list))      
                max_abs_batch_size_weights[weights_layer_idx].append(max_abs_weights.item())  
    
    return acc, max_abs_layer_inputs, max_abs_layer_outputs, max_abs_batch_size_weights


def get_scale_factor_moabz(model, data_loader, total_images): # get Ka, Kw
    ''' maximum of all bitch sizes '''
    model.eval()       
    correct = 0
    count = 0
    layer_inputs  = {}  # store inputs for each layer
    layer_outputs = {} 
    layer_weights = {} 

    i = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        i = i + 1
        
        inputs, targets = inputs.cuda(), targets.cuda()

        # Record inputs and outputs for each layer
        model.reset_layer_inputs_outputs()
        model.reset_layer_weights()
        outputs = model(inputs)
        # Get inputs and outputs for each layer
        current_layer_inputs = model.get_layer_inputs()
        current_layer_outputs = model.get_layer_outputs()
        current_layer_weights = model.get_layer_weights()
        #   print(len(current_layer_inputs[0]))
        
        for idx, input_tensor in current_layer_inputs.items():
            if idx not in layer_inputs:
                layer_inputs[idx] = []
            layer_inputs[idx].append(input_tensor.detach().cpu())
            
        for idx, output_tensor in current_layer_outputs.items():
            if idx not in layer_outputs:
                layer_outputs[idx] = []
            layer_outputs[idx].append(output_tensor.detach().cpu())

        for idx, weight_tensor in current_layer_weights.items():
            if idx not in layer_weights:
                layer_weights[idx] = []
            layer_weights[idx].append(weight_tensor.detach().cpu())
        

        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

        count += len(inputs)
        if count >= total_images:
            break

    # Calculate the maximum absolute values for layer inputs and outputs

    max_abs_layer_inputs  = []
    max_abs_layer_outputs = []
    max_abs_layer_weights = []

    for idx, inputs_list in layer_inputs.items():
        max_abs_input = torch.max(torch.abs(torch.cat(inputs_list, dim=0))) # torch.cat ：合并 bitch size 
        max_abs_layer_inputs.append(max_abs_input.item()) 
        
    for idx, outputs_list in layer_outputs.items():
        max_abs_output = torch.max(torch.abs(torch.cat(outputs_list, dim=0)))
        max_abs_layer_outputs.append(max_abs_output.item())

    for idx, weights_list in layer_weights.items():
        max_abs_output = torch.max(torch.abs(torch.cat(weights_list, dim=0)))
        max_abs_layer_weights.append(max_abs_output.item())
    
    acc = 100. * correct / total_images
    print('Precision@1: %.2f%% \n' % (acc))

    return acc, max_abs_layer_inputs, max_abs_layer_outputs, max_abs_layer_weights


def put_scale_factor_txt_improved(max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_channel_weights, weight_folder_path, input_folder_path):

    """ ---------------  weights ----------------"""
    max_abs_channel_weights_layer_0 = max_abs_channel_weights[0]
    # print(len(max_abs_channel_weights_layer_0))
    max_abs_channel_weights_layer_except0 = max_abs_channel_weights[1:]

    max_abs_channel_weights_str = '\n'.join([' '.join(map(str, row)) for row in max_abs_channel_weights_layer_0])
    with open(weight_folder_path, 'w') as file:
        file.write(max_abs_channel_weights_str)

    max_abs_channel_weights_str = '\n'.join([' '.join(map(str, row)) for row in max_abs_channel_weights_layer_except0])
    with open(weight_folder_path, 'a') as file:         # add new data
        file.write("\n")  
        file.write("\n") 
        file.write(max_abs_channel_weights_str)

    """ ---------------  inputs ----------------"""   
    with open(input_folder_path, 'w') as file:
        for item in max_abs_layer_inputs_list:
            file.write(str(item) + '\n')

    return len(max_abs_channel_weights_layer_0)


def put_scale_factor_bitch_size_txt(max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_channel_weights, weight_folder_path, input_folder_path):

    """ ---------------  weights ----------------"""
    max_abs_channel_weights_str = '\n'.join([' '.join(map(str, row)) for row in max_abs_channel_weights])
    with open(weight_folder_path, 'w') as file:
        file.write(max_abs_channel_weights_str)
    
    """ ---------------  inputs ----------------"""
    with open(input_folder_path, 'w') as file:
        for item in max_abs_layer_inputs_list:
            file.write(str(item) + '\n')

def put_scale_factor_layer_txt(max_abs_layer_inputs_list, max_abs_layer_outputs_list, max_abs_layer_weights_list, weight_folder_path, input_folder_path):
    """ ---------------  weights ----------------"""
    with open(weight_folder_path, 'w') as file:
        for item in max_abs_layer_weights_list:
            file.write(str(item) + '\n')
    
    """ ---------------  inputs ----------------"""
    with open(input_folder_path, 'w') as file:
        for item in max_abs_layer_inputs_list:
            file.write(str(item) + '\n')


def acquire_weight_bitch_size_scale_factor_txt(folder_path):
    # Read the txt file and convert the data to a list form
    with open(folder_path, 'r') as file:
        lines = file.readlines()  # Read the data line by line
    scale_factor_list = []        # Store the list of results
    for line in lines:
        values = line.strip().split()               # Remove newlines and cut by space
        values = [float(value) for value in values] # Converts a string type to a floating-point number type
        scale_factor_list.append(values)            # Add the processed data to the result list

    return scale_factor_list

def acquire_weight_layer_scale_factor_txt(folder_path):
    # Read the txt file and convert the data to a list form
    with open(folder_path, 'r') as file:
        scale_factor_list = [float(line.strip()) for line in file]  

    return scale_factor_list


def acquire_input_layer_scale_factor_txt(folder_path):
    # Read the txt file and convert the data to a list form
    with open(folder_path, 'r') as file:
        scale_factor_list = [float(line.strip()) for line in file]  

    return scale_factor_list