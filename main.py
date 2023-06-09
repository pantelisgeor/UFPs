import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.create_model import convAutoEncoder
from src.data_loader import NcDataset, NcDatasetMem
from src.utils import str2bool, RMSELoss, train_loop, test_loop

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
        
        
        
        
        
# Datasets (training and test)
views=['land_1', 'land_2', 'land_3', 'land_4', 'land_5', 'land_6', 'land_7',
       "t2m0", "d2m0", "tp0", "u100", "v100", "no0", "no20", "co0"]

training_data = NcDataset(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/train.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/train",
    views=views)
training_data2 = NcDataset(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test2/train.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test2/train",
    views=views)
training_data3 = NcDataset(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test3/train.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test3/train",
    views=views)
training_data4 = NcDataset(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test4/train.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test3/train",
    views=views)

test_data = NcDataset(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/test.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/test",
    views=views)

# DataLoader objects
train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
train_dataloader2 = DataLoader(training_data2, batch_size=128, shuffle=True)
train_dataloader3 = DataLoader(training_data3, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)


model = convAutoEncoder(channels=len(views))
model.to("cuda")
        
 # Loss function definition
loss_fn = RMSELoss()  #nn.MSELoss()
        
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)


for ep in range(150):
    print(f"Epoch {ep} \n")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    train_loop(train_dataloader2, model, loss_fn, optimizer)
    train_loop(train_dataloader3, model, loss_fn, optimizer)
    print("Test score: ")
    test_loop(test_dataloader, model, loss_fn)

        
        
path_model = "/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/models/test1.pt"

torch.save(model.state_dict(), path_model)

the_model = convAutoEncoder(channels=len(views))
the_model.load_state_dict(torch.load(path_model))
the_model.eval()
the_model.to("cuda")


the_model = torch.load("/home/pantelisgeorgiades/Python/UFP/datasets_/models/test1")

the_model = convAutoEncoder(channels=len(views))
the_model.load_state_dict(
    torch.load("/home/pantelisgeorgiades/Python/UFP/datasets_/models/test1"))


test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

train_dataloader_Test = DataLoader(training_data, batch_size=1, shuffle=False)

device = "cuda"
outputs = []
gr_truth = []
inputs = []
with torch.no_grad():
    for idx, (X, y) in tqdm(enumerate(test_dataloader)):
        # Move these to GPU if device is cuda
        if device == "cuda":
            X, y = X.cuda(), y.cuda()
        # Calculate the model's prediction
        gr_truth.append(y)
        outputs.append(model(X))
        inputs.append(X)
        
        
        
        # if idx%10 == 0:
        #     print('.', end='', flush=True)