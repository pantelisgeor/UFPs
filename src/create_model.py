import torch
import torch.nn as nn
import torch.nn.functional as F


class convAutoEncoder(nn.Module):
    
    def __init__(self, channels):
        super(convAutoEncoder, self).__init__()
        # Normalisation layer for the input
        
        # Encoder layers (convolutional autoencoder)        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, 
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=32,
                          kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, 
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
            nn.ReLU(),
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



class convAutoEncoder(nn.Module):
    
    def __init__(self, channels):
        super(convAutoEncoder, self).__init__()
        # Normalisation layer for the input
        
        # Encoder layers (convolutional autoencoder)        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, 
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=32,
                          kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, 
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
            nn.ReLU(),
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
