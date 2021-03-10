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
import pytorch_lightning as pl


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
        x: list of tensor of size (B, S, C, H, W)

        """
        return x.mean(1)    
        


    
class AttentionAggregation(nn.Module):
    """ This Block uses Multi-head attention to aggregate a series of 2D feautures"""
    
    def __init__(self, embed_dim = 64, num_heads=4, dim_feedforward=64, num_layers=3):
        super(AttentionAggregation, self).__init__()
        #self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.softmax = nn.Softmax(1)
        


    def forward(self, xs):
        """Forward pass of our Aggregation Block 
        
        xs: list of tensors of size (B, S, C, H, W)
            S: Sequence Length
            B: batch size
            C: Channels (= embed_dim)
            H: Height
            W: Width 
        """
        B, S, C, H, W = xs.size()             # -> (B, S, C, H, W)
        y = xs.permute(1,0,3,4,2)             # -> (S, B, H, W, C)
        y = xs.view(S,-1,C)                   # -> (S, B x H x W, C)
        
        y = self.transformer_encoder(y)       # -> (S, B x H x W, C)
        y = y.view(S, B, H, W, C)             # -> (S, B, H, W, C)
        y = y.permute(1,0,4,2,3)              # -> (B, S, C, H, W)
        y = self.softmax(y)                   # -> (B, S, C, H, W) normalized along (S)equence dimension
        y = (y*xs).sum(1)                     # -> (B, C, H, W)
        
        return y

    
class DeepAggNet(pl.LightningModule):
    """ Deep Set Residual Neural Network """
    def __init__(self, encoder_num_blocks=10, decoder_num_blocks=10, smooth_num_blocks=6, planes=32, agg_block="Mean", agg_params=None):
        """
        encoder_num_blocks: Number of residual blocks used for encoding the images into an embedding
        decoder_num_blocks: Number of residual blocks used for decoding the embeddings into an image
        smooth_num_blocks:  Number of residual blocks used for smoothing the upsampled/decoded embedding
        planes:             Number of feature planes used in the initial embedding,
                            the number of planes double after each downsampling
        agg_block:          A block that aggregates a series of embeddings into a singular embedding: 
                            (B, S, C, H, W) -> (B, C, H, W)
        """
        super(DeepAggNet, self).__init__()
                
        
        
        self.planes = planes
        self.input = nn.Conv2d(3, self.planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.output= nn.Conv2d(self.planes, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        
        # Create a down-/up-sampling architecture
        self.downsample = []
        self.upsample = []
        n = planes
        for i in range(2):
            # create downsampling layers using convolutions with strides
            self.downsample.append( nn.Conv2d(in_channels = n, out_channels=n*2, kernel_size=3, stride=2, padding=1 ) )
            self.downsample.append(nn.ReLU(inplace=True))

            # create upsampling layers using transposed convolutions (should be symmetric to downsampling)
            self.upsample = [nn.ReLU(inplace=True)] + self.upsample
            self.upsample = [nn.ConvTranspose2d(in_channels=n*2, out_channels=n, kernel_size=3, stride=2, padding=1, output_padding=1)] + self.upsample
            n *= 2
            
            
        
        self.downsample = nn.Sequential(*self.downsample)
        self.upsample = nn.Sequential(*self.upsample)
        
        
        # Embedding of downsampled features
        self.encoder = self._make_layer(n, encoder_num_blocks)
        
        # Define Aggregation-Block
        if agg_block == "Attention":
            self.agg = AttentionAggregation(embed_dim=n, **agg_params)
        else:
            self.agg = MeanAgg()
        
        # create decoder layers that are applied on the aggregated features
        self.decoder = self._make_layer(n, decoder_num_blocks)
        # create smoothing layers that are applied on the upsampled features
        self.smooth  = self._make_smooth_layer(planes, smooth_num_blocks)
        
        #### LOGGING #####
        self.hyperparams = {
            "encoder_num_blocks":encoder_num_blocks, 
            "decoder_num_blocks":decoder_num_blocks, 
            "smooth_num_blocks":smooth_num_blocks, 
            "planes":planes, 
            "embed_dim":n,
            "agg_block":"Mean",
        }
        self.hyperparams.update(agg_params)
        self.save_hyperparameters(self.hyperparams)

        
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
        embedding = torch.stack(embedding,1)
        embedding = self.agg(embedding)
        out = self.output(self.smooth(self.upsample(self.decoder(embedding))))
        
        return out
    
    def training_step(self, batch, batch_idx):
        """Forward pass of our DeepSet Network 
        
        batch : tuple of tensors of size (B, S, C, H, W)
        """
        # training_step defined the train loop. It is independent of forward
        x, y = batch

        # Forward pass
        xs = torch.split(x,1,dim = 1)
        xs = [torch.squeeze(x,dim=1) for x in xs]
        embedding = [self.encoder(self.downsample(self.input(x))) for x in xs]
        embedding = torch.stack(embedding,1)
        embedding = self.agg(embedding)
        out = self.output(self.smooth(self.upsample(self.decoder(embedding))))
        
        loss = F.mse_loss(y, out)
        self.log('train_loss', loss)
        #self.logger.experiment.log_metric({'train_loss':loss.item()})
        logs = {'train_loss':loss}
        return {'loss': loss, 'logs': logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
