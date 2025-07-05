import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn as nn 
import torch.nn.functional as F
import random
import torch
import math





class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_c, track_running_stats=True)         
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_c, track_running_stats=True)         
        self.relu = nn.ReLU()     
        self.tanh = nn.Tanh()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))     
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=(3,3), stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)     
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class conv_block_heuristic(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c, track_running_stats=True)         
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c, track_running_stats=True)         
        self.relu = nn.ReLU()     
        self.tanh = nn.Tanh()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        return x

class build_net(nn.Module):
    def __init__(self):
        super().__init__()
        # '''Encoder ''' 
        self.e1 = encoder_block(3,16)
        self.e2 = encoder_block(16,32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)         
        self.e5 = encoder_block(128, 256)
        self.b = conv_block_heuristic(256, 512)
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)
        self.d5 = decoder_block(32, 16)    
        self.outputs = nn.Conv2d(16, 1, kernel_size=(3,3),stride=(1,1),padding=(1,1))    
        self.dropout1 = nn.Dropout2d(p=0.4)
        self.dropout2 = nn.Dropout2d(p=0.4)
        self.dropout3 = nn.Dropout2d(p=0.4)
        self.dropout4 = nn.Dropout2d(p=0.4)
        self.dropout5 = nn.Dropout2d(p=0.4)
        self.dropout6 = nn.Dropout2d(p=0.4)
    def forward(self, inputs):
        #'''Encoder'''
        s1, p1 = self.e1(inputs)
        p1 = self.dropout1(p1)
        s2, p2 = self.e2(p1)
        p2 = self.dropout2(p2)
        s3, p3 = self.e3(p2)
        p3 = self.dropout3(p3)
        s4, p4 = self.e4(p3)      
        p4 = self.dropout4(p4)
        s5, p5 = self.e5(p4)
        p5 = self.dropout5(p5)
        b =self.b(p5)       
        # '''Decoder'''  
        ## In decoding path, padding & cropping as needed to match dimension for concatenation
        s5 = F.pad(s5,(0,0,1,0))
        b = F.pad(b,(0,0,0,0))
        d1 = self.d1(b, (s5))
        s4 = F.pad(s4,(0,0,1,1))
        d2 = self.d2(d1, (s4) )
        s3 = F.pad(s3, (1,0,1,0))
        d2 = d2[:,:,1:-1,:]
        d3 = self.d3(d2, (s3))
        d3 = d3[:,:,1:-1,:]
        s2 = s2[:,:,1:,:]
        s2 = F.pad(s2, (1,2,0,0))
        d4 = self.d4(d3, (s2))    
        d4 = F.pad(d4, (0,0,0,1))
        d4 = d4[:,:,:,2:-1]
        s1 = F.pad(s1,(0,0,0,0))
        d5 = self.d5(d4, torch.zeros_like(s1))
        outputs = (self.outputs(d5))   
        return outputs
