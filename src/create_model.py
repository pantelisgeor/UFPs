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
        self.flatten = nn.Flatten(0, 2)
        self.fc1 = nn.LazyLinear(out_features=512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=1)


    def forward(self, X):
        """Forward layer"""
        # Normalise input
        # x = F.normalize(X)
        # Rest of network
        x = self.encoder(X)
        x = self.fc1(self.flatten(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = self.dropout(self.fc4(x))
        x = self.fc5(x)
        
        return x
