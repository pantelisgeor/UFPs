import torch
import torch.nn as nn
import torch.nn.functional as F
from src._ResNet import ResNet50, ResNet101, ResNet152


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
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=512),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=128),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=32),
            nn.Tanh(),
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


class ResNetConv2(nn.Module):

    def __init__(self, channels, resnet="ResNet50"):
        super(ResNetConv2, self).__init__()
        # Resnet encoder
        if resnet == "ResNet50":
            self.ResNet = ResNet50(num_classes=512, channels=channels)
        elif resnet == "ResNet101":
            self.ResNet = ResNet101(num_classes=512, channels=channels)
        elif resnet == "ResNet152":
            self.ResNet = ResNet152(num_classes=512, channels=channels)

        # Feed forward model
        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=1)
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


class ResNetConvSmall(nn.Module):
    
    def __init__(self, channels, resnet="ResNet50"):
        super(ResNetConvSmall, self).__init__()
        # Resnet encoder
        if resnet == "ResNet50":
            self.ResNet = ResNet50(num_classes=512, channels=channels)
        elif resnet == "ResNet101":
            self.ResNet = ResNet101(num_classes=512, channels=channels)
        elif resnet == "ResNet152":
            self.ResNet = ResNet152(num_classes=512, channels=channels)
        
        # Feed forward model
        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=1)
        )
        
        self.fc_ = nn.Sequential(
            nn.LazyLinear(out_features=1)
        )
        
    def forward(self, X):
        """Forward layer"""
        # Rest of network
        x = self.ResNet(X)
        # print(f"x.shape: {x.shape}\n")
        # Flatten it
        # x = x.view(x .size(0), -1)
        x = self.fc_(x)
        
        return x