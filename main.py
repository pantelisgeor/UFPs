import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.create_model import convAutoEncoder
from src.data_loader import NcDataset, NcDatasetMem
from src.utils import str2bool, RMSELoss, train_loop, test_loop
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--dat_dir", type=str, 
                    help="Path to directory where data is stored")
parser.add_argument("--model", type=str,
                    default="convAutoEncoder")
parser.add_argument("--views", default=None, nargs="+",
                    help="The variables in the datasets to \
                        be used as input features.")
parser.add_argument("--batch_size", default=64,
                    help="Batch size for training and evaluation methods.")
parser.add_argument("--shuffle", type=str2bool, nargs='?',
                    const=False, default=True, 
                    help="Boolean indicator whether to shuffle \
                        training and test data (Default: True)")
parser.add_argument("--learning_rate", type=float,
                    default=1e-3, help="Neural Network's learning rate")
parser.add_argument("--epochs", default=20, type=int,
                    help="Number of epochs to train neural network for.")
parser.add_argument("--device", type=str, default="cuda",
                    help="Which device for pytorch to use (Default: CUDA)")

args = parser.parse_args()


if __name__ == "__main__":
    # Read the data using the custom Dataset class
    training_data = NcDataset(
        parquet_file=f"{args.dat_dir}/train.parquet",
        root_dir=f"{args.dat_dir}/train", views=args.views)
    test_data = NcDataset(
        parquet_file=f"{args.dat_dir}/test.parquet",
        root_dir=f"{args.dat_dir}/test", views=args.views)
    
    # DataLoader objects
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=args.shuffle)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=args.shuffle)
    
    # Define the model
    if args.model == "convAutoEncoder":
        model = convAutoEncoder(channels=len(args.views) if args.views else 74)
    if args.device == "cuda":
        model.to("cuda")
        
    # Loss function definition
    loss_fn = RMSELoss()  #  nn.MSELoss()
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    for ep in range(args.epochs):
        print(f"Epoch {ep} \n")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        print("Test score: ")
        test_loop(test_dataloader, model, loss_fn)
        
