import argparse
import torch
import torch.nn as nn


def str2bool(v):
    """
    Function to convert a set of inputs to a boolean argument
    using argparse

    Args:
        v (str): Boolean input (can be any of a set of strings)

    Raises:
        argparse.ArgumentTypeError: Raised when unexpected values
                                    is entered.

    Returns:
        bool: Boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
class RMSELoss(nn.Module):
    """ Pytorch Implementation of Root Mean Square Error

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        """_summary_

        Args:
            yhat (torch.tensor): Prediction by model
            y (torch.tensor): Ground Truth

        Returns:
            float: Root mean squared error between yhat and y
        """
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
    
def train_loop(dataloader, model, loss_fn, optimizer, device="cuda"):
    """Train loop for training the pytorch model

    Args:
        dataloader (torch.dataloader): DataLoader object
        model (torch model): PyTorch model
        loss_fn (): Loss function instance
        optimizer (_type_): Optimizer instance
        device (_type_): Device for PyTorch to use (Default: "cuda")
    """
    # Get the size of the minibatches
    size = len(dataloader.dataset)
    
    # Loop through the mini-batches and perform the training procedures
    for batch, (X, y) in enumerate(dataloader):
        # Move the batch to gpu (cuda) if device == "cuda"
        if device == "cuda":
            X, y = X.cuda(), y.cuda()
            
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Perform backpropagation procedures (and update the parameters)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]", flush=True)
            
            
def test_loop(dataloader, model, loss_fn, device="cuda"):
    """Iterates over the test set and calculates the errors

    Args:
        dataloader (torch.DataLoader): Pytorch DataLoader object
        model (PyTorch model): PyTorch model class instance
        loss_fn (): Loss function
        device (str, optional): _description_. Defaults to "cuda".
    """
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Set to no_grad() so weights are not updated
    with torch.no_grad():
        for X, y in dataloader:
            # Move these to GPU if device is cuda
            if device == "cuda":
                X, y = X.cuda(), y.cuda()
                
            # Calculate the model's prediction
            pred = model(X)
            # Get the loss of this batch and add it to the total
            test_loss += loss_fn(pred, y)
                        
    test_loss /= num_batches
    print(f"Test Error: Average MSE: {test_loss:>8f} \n")