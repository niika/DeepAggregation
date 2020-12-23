from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,models
import cv2
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from scipy import random, linalg
from sklearn.model_selection import train_test_split
import torch.optim as optim
import re
import json
import time

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return out
    
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, dilation=dilation,padding=dilation, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return out
     
class MeanAgg(nn.Module):
    def __init__(self):
        super(MeanAgg, self).__init__()

    def forward(self, x):
        """ Forward-pass of mean-aggregation
        x: list of tensor of size S x (B, C, H, W)

        """
        x = torch.stack(x).mean(0)
        return x    
        
class DeepSetNet(nn.Module):
    """ Deep Set Residual Neural Network """
    def __init__(self, encoder_num_blocks=10, decoder_num_blocks=10, smooth_num_blocks=6, planes=32, agg_block=MeanAgg):
        """
        encoder_num_blocks: Number of residual blocks used for encoding the images into an embedding
        decoder_num_blocks: Number of residual blocks used for decoding the embeddings into an image
        smooth_num_blocks:  Number of residual blocks used for smoothing the upsampled/decoded embedding
        planes:             Number of feature planes used in the initial embedding,
                            the number of planes double after each downsampling
        agg_block:          A block that aggregates a series of embeddings into a singular embedding: 
                            S x (B, C, H, W) -> (B, C, H, W)
        """
        super(DeepSetNet, self).__init__()
        self.planes = planes
        self.input = nn.Conv2d(3, self.planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.output= nn.Conv2d(self.planes, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        
        # Create a down-/up-sampling architecture
        self.downsample = []
        self.upsample = []
        n = planes
        for i in range(2):
            self.downsample.append( nn.Conv2d(in_channels = n, out_channels=n*2, kernel_size=3, stride=2, padding=1 ) )
            self.downsample.append(nn.ReLU(inplace=True))

            
            self.upsample = [nn.ReLU(inplace=True)] + self.upsample
            self.upsample = [nn.ConvTranspose2d(in_channels=n*2, out_channels=n, kernel_size=3, stride=2, padding=1, output_padding=1)] + self.upsample
            n *= 2

        self.downsample = nn.Sequential(*self.downsample)
        self.upsample = nn.Sequential(*self.upsample)
        
        
        # Embedding of downsampled features
        self.encoder = self._make_layer(n, encoder_num_blocks)
        
        self.agg = agg_block()
        self.decoder = self._make_layer(n, decoder_num_blocks)
        self.smooth  = self._make_smooth_layer(planes, smooth_num_blocks)
        
    def _make_layer(self, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(DilatedResidualBlock(planes, planes,2))
        return nn.Sequential(*layers)
    
    def _make_smooth_layer(self, planes, num_blocks):
        layers = []
        dilation = 1
        for i in range(num_blocks):
            layers.append(DilatedResidualBlock(planes,planes,dilation))
            if i%2 == 0:
                dilation *= 2
        layers.append( nn.Conv2d(in_channels = planes, out_channels=planes, kernel_size=3, stride=1, padding=1 ) )
        layers.append(nn.ReLU(inplace=True))
        layers.append( nn.Conv2d(in_channels = planes, out_channels=planes, kernel_size=3, stride=1, padding=1 ) )
        return nn.Sequential(*layers)
            
        

    def forward(self, x):
        """Forward pass of our DeepSet Network 
        
        x: of tensor of size (B, S, C, H, W)
        """

        xs = torch.split(x,1,dim = 1)
        xs = [torch.squeeze(x,dim=1) for x in xs]
        embedding = [self.encoder(self.downsample(self.input(x))) for x in xs]
        embedding = self.agg(embedding)
        out = self.output(self.smooth(self.upsample(self.decoder(embedding))))

        
        return out

class AttentionAggregation(nn.Module):
    """ This Block uses Multi-head attention to aggregate a series 2D feautures"""
    
    def __init__(self, embed_dim = 64, num_heads=8):
        super(AttentionAggregation, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, xs):
        """Forward pass of our Aggregation Block 
        
        xs: list of tensors of size S x (B, C, H, W)
            S: Sequence Length
            B: batch size
            C: Channels (= embed_dim)
            H: Height
            W: Width
        """
        B, C, H, W = xs[0].size()
        S = len(xs)
        # Transform list of tensors S x (B, C, H, W) -> B x (S, H x W, C)
        x =  torch.stack(xs)                          # -> (S, B, C, H, W)
        xs = torch.split(x,1,1)                       # ->  B x (S, 1, H, W, C)
        xs = [x.squeeze(1).view(S,-1,C) for x in xs]  # ->  B x (S, H x W, C)
        
        # Compute attetion over the sequence of images
        xs = [self.attention(x,x,x)[0] for x in xs]   # -> B x (S, H x W, C)
        xs =  [x.view(S,H,W,C) for x in xs]           # -> B x (S, H, W, C)
        xs =  [x.view(S,C,H,W) for x in xs]           # -> B x (S, C, H, W, )
        xs = torch.stack(xs, 1)                       # -> (S, B, C, H, W)
        
        
        #xs = torch.split(xs, 1, 0)                    # -> S x (B, 1, C, H, W)
        #xs = [x.squeeze(0) for x in xs]               # -> S x (B, C, H, W)
        return xs