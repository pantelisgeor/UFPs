import torch
import torch.nn as nn
import torch.nn.functional as F


class convAutoEncoder(nn.Module):
    
    def __init__(self, channels):
        super(convAutoEncoder, self).__init__()
        # Normalisation layer for the input
        
        # Encoder layers (convolutional autoencoder)        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, 
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128,
                          kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, 
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2)
        )
        # Feed forward model
        self.fc = nn.Sequential(
            # nn.Flatten(0, 3),
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            # nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )


    def forward(self, X):
        """Forward layer"""
        # Rest of network
        x = self.encoder(X)
        # Flatten it
        x = x.view(x .size(0), -1)
        x = self.fc(x)
        
        return x
    
    
class FeedForward(nn.Module):
    
    def __init__(self, channels):
        super(FeedForward, self).__init__()

        # Feed forward model
        self.fc = nn.Sequential(
            # nn.Flatten(0, 3),
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            # nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )


    def forward(self, X):
        """Forward layer"""
        # Flatten it
        x = X.view(X .size(0), -1)
        x = self.fc(x)
        
        return x


class convAutoEncoder2(nn.Module):
    
    def __init__(self, channels):
        super(convAutoEncoder2, self).__init__()
        # Normalisation layer for the input
        
        # Encoder layers (convolutional autoencoder)        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=256, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Feed forward model
        self.fc = nn.Sequential(
            # nn.Flatten(0, 3),
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )


    def forward(self, X):
        """Forward layer"""
        # Rest of network
        x = self.layer3(self.layer2(self.layer1(X)))
        # print(f"x.shape: {x.shape}\n")
        # Flatten it
        x = x.view(x .size(0), -1)
        x = self.fc(x)
        
        return x
    
    
class convAutoEncoder3(nn.Module):
    
    def __init__(self, channels):
        super(convAutoEncoder3, self).__init__()
        # Normalisation layer for the input
        
        # Encoder layers (convolutional autoencoder)        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Feed forward model
        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )


    def forward(self, X):
        """Forward layer"""
        # Rest of network
        x = self.layer4(self.layer3(self.layer2(self.layer1(X))))
        # print(f"x.shape: {x.shape}\n")
        # Flatten it
        x = x.view(x .size(0), -1)
        x = self.fc(x)
        
        return x
    
    
# Resnet implementations
# https://github.com/JayPatwardhan/ResNet-PyTorch 
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, 
                 i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 
                               kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, 
                                     padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], 
                                       planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], 
                                       planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], 
                                       planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], 
                                       planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, 
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, 
                               i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

class ResNetConv(nn.Module):
    
    def __init__(self, channels, resnet="ResNet50"):
        super(ResNetConv, self).__init__()
        # Resnet encoder
        if resnet == "ResNet50":
            self.ResNet = ResNet50(num_classes=512, channels=channels)
        elif resnet == "ResNet101":
            self.ResNet = ResNet101(num_classes=512, channels=channels)
        elif resnet == "ResNet152":
            self.ResNet = ResNet152(num_classes=512, channels=channels)
        
        # Feed forward model
        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        
    def forward(self, X):
        """Forward layer"""
        # Rest of network
        x = self.ResNet(X)
        # print(f"x.shape: {x.shape}\n")
        # Flatten it
        # x = x.view(x .size(0), -1)
        x = self.fc(x)
        
        return x