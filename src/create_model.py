import torch
import torch.nn as nn
import torch.nn.function as F


class resnet(nn.Module):
    
    def __init__(self, in_channels, out_channels)