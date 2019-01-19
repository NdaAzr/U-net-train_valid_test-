# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:00:36 2018

@author: Neda
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



print("Torch version: ", torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("If torch.cuda.is_available print cuda:0")
print(device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
    print(torch.cuda.get_device_name(1))
    print(torch.cuda.get_device_capability(1))


torch.cuda.set_device(0)
torch.cuda.current_device()

 
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(BaseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x
    
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DownConv, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size, padding, stride)
    
    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__ (self, in_channels, in_channels_skip, out_channels, kernel_size, padding, stride):
        super(UpConv, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.conv_block = BaseConv(in_channels=in_channels + in_channels_skip, out_channels= out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        
    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x= self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size, padding, stride):
        super(UNet, self).__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size, padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size, padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size, padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size, padding, stride)

        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels, kernel_size, padding, stride)

        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels, kernel_size, padding, stride)

        self.up1 = UpConv(2 * out_channels, out_channels, out_channels, kernel_size, padding, stride)

        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Decoder
        x_up1 = self.up3(x3, x2)
        x_up2 = self.up2(x_up1, x1)
        x_up3 = self.up1(x_up2, x)
        x_out = F.log_softmax(self.out(x_up3), 1)
        #print(x_out.size())
        return x_out
 

    
model = UNet(in_channels=1,
             out_channels=60,
             n_class=2,
             kernel_size=3,
             padding=1,
             stride=1)

model = model.to(device)
#print(model)
print("UNet model created")



 #Print model's state_dict
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
## Print optimizer's state_dict
#print("Optimizer's state_dict:")
#for var_name in optimizer.state_dict():
#    print(var_name, "\t", optimizer.state_dict()[var_name])





    