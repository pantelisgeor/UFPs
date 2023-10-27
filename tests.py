import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.create_model import convAutoEncoder, FeedForward, convAutoEncoder2
from src.data_loader import NcDataset, NcDatasetMem
from src.utils import str2bool, RMSELoss, train_loop, test_loop
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


# Datasets (training and test)
views=["land_1", "land_2", "land_3", "land_4", "land_5", 
       "t2m0", "u100", "v100", "tp0", "no0", "no20"]

views = ["no-2", "no2-2", "no0", "no20", "no2", "no22"]

training_data = NcDatasetMem(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test1/train_.parquet",
    root_dir="/home/pantelisgeorgiades/Python/UFP/datasets_/test1/train",
    views=views)
"""
training_data2 = NcDatasetMem(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test2/train_.parquet",
    root_dir="/home/pantelisgeorgiades/Python/UFP/datasets_/test2/train",
    views=views)
training_data3 = NcDatasetMem(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test2/test_.parquet",
    root_dir="/home/pantelisgeorgiades/Python/UFP/datasets_/test2/test",
    views=views)
"""
test_data = NcDatasetMem(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test1/test_.parquet",
    root_dir="/home/pantelisgeorgiades/Python/UFP/datasets_/test1/test",
    views=views)

# DataLoader objects
train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)
# train_dataloader2 = DataLoader(training_data2, batch_size=128, shuffle=True)
# train_dataloader3 = DataLoader(training_data3, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)


model = convAutoEncoder2(channels=len(views))
model.to("cuda")
        
 # Loss function definition
loss_fn = nn.MSELoss()  #RMSELoss()
        
# Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # , weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


for ep in range(4):
    print(f"Epoch {ep} \n")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    # train_loop(train_dataloader2, model, loss_fn, optimizer)
    # train_loop(train_dataloader3, model, loss_fn, optimizer)
    print("Test score: ")
    test_loop(test_dataloader, model, loss_fn)


test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

train_dataloader_Test = DataLoader(training_data, batch_size=1, shuffle=False)

device = "cuda"
outputs = []
outputs_ = []
gr_truth = []
inputs = []
with torch.no_grad():
    for idx, (X, y) in tqdm(enumerate(test_dataloader)):
        # Move these to GPU if device is cuda
        if device == "cuda":
            X, y = X.cuda(), y.cuda()
        # Calculate the model's prediction
        gr_truth.append(y.item())
        outputs.append(model(X).item())



sns.lineplot(gr_truth, label="Ground truth")
sns.lineplot(outputs, label="predicted")
plt.show()




        
path_model = "/home/pantelisgeorgiades/Python/UFP/datasets_/models/test1.pt"

torch.save(model.state_dict(), path_model)

the_model = convAutoEncoder(channels=len(views))
the_model.load_state_dict(torch.load(path_model))
the_model.eval()
the_model.to("cuda")


the_model = torch.load("/home/pantelisgeorgiades/Python/UFP/datasets_/models/test1")

the_model = convAutoEncoder(channels=len(views))
the_model.load_state_dict(
    torch.load("/home/pantelisgeorgiades/Python/UFP/datasets_/models/test1"))

import pandas as pd

df = pd.read_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test2/test.parquet")
df.sensor_3 = df.sensor_3 / 238.94
df.to_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test2/test_.parquet")

df = pd.read_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test2/train.parquet")
df.sensor_3 = df.sensor_3 / 238.94
df.to_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test2/train_.parquet")


df = pd.read_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test3/test.parquet")
df.sensor_3 = df.sensor_3 / 238.94
df.to_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test3/test_.parquet")

df = pd.read_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test3/train.parquet")
df.sensor_3 = df.sensor_3 / 238.94
df.to_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test3/train_.parquet")


df = pd.read_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test4/test.parquet")
df.sensor_3 = df.sensor_3 / 238.94
df.to_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test4/test_.parquet")

df = pd.read_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test4/train.parquet")
df.sensor_3 = df.sensor_3 / 238.94
df.to_parquet("/home/pantelisgeorgiades/Python/UFP/datasets_/test4/train_.parquet")

del df





import xarray as xr
import pandas as pd
import datetime, gc


path_dat = "/nvme/h/pgeorgiades/data_p143/UFPs/datasets/stations2_cleaned"
path_target = "/nvme/h/pgeorgiades/data_p143/UFPs/datasets/Evaluation_cleaned.parquet"

# List the stations
files = os.listdir(path_dat)
stations = [x.split(".")[0] for x in os.listdir(path_dat)]

# Read the target variable dataset
target = pd.read_parquet(path_target)

    
def get_station(station, path_dat=path_dat, target=target,
                views=views):
    # Read the data for the specified station (aux. info)
    ds = xr.open_dataset(f"{path_dat}/{station}.nc")
    ds['time'] = pd.to_datetime(ds.time.values)
    if "spatial_ref" in ds.variables:
        ds = ds.drop("spatial_ref")
    # Subset for the station from target dataset
    targ_ = target.loc[target.station == station]
    targ_ = targ_.reset_index(drop=True)
    # Convert both to pytorch tensors
    if views:
        # List the variables
        ds = ds[views]
    ds = torch.tensor(ds.to_array().values.squeeze(),
                      dtype=torch.float32)
    targ_ = torch.tensor(targ_.loc[targ_.type == "obs"].value.values,
                         dtype=torch.float32)
    return ds[:, :, :, :], targ_


# If station_list is None, load them all, else only load the
# station names in the list
stations_ = station_list if station_list is not None else stations
for i, st in tqdm(enumerate(stations_)):
    if i == 0:
        st_dat, targ_dat = get_station(st)
    else:
        st_temp, targ_temp = get_station(st)
        if views:
            try:
                st_dat = torch.concat((st_dat, st_temp), 1)
            except Exception as e:
                st_dat = torch.concat((st_dat, st_temp), 3)
        else:
            st_dat = torch.concat((st_dat, st_temp), 3)
        targ_dat = torch.concat((targ_dat, targ_temp))
        del st_temp, targ_temp


dat = st_dat
target = targ_dat
del i, st, st_dat, targ_dat
gc.collect()



torch.Size([26, 100, 100, 728])



import numpy as np


sns.set(font_scale=1.8)
sns.set_style("white")
for station_ in tqdm(stations):
    if station_ == "GR0002R":
        continue
    training_data = pm01data_(path_dat=path_dat, path_target=path_target,
                              station_list=[station_],
                              views=["no", "no2", "pm2p5", "t2m"])

    # DataLoader objects
    train_dataloader = DataLoader(training_data, batch_size=1,
                                  shuffle=False)
    
    no = []
    no2 = []
    pm25 = []
    t2m = []
    target = []

    for i in np.arange(len(train_dataloader)):
        d_ = training_data.__getitem__(i)
        no.append(d_[0][0].mean().item())
        no2.append(d_[0][1].mean().item())
        pm25.append(d_[0][2].mean().item())
        t2m.append(d_[0][3].mean().item())
        target.append(d_[1].item())

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    sns.lineplot(no, label="NO", ax=ax, linewidth=2)
    sns.lineplot(no2, label="NO2", ax=ax, linewidth=2)
    sns.lineplot(pm25, label="PM25", ax=ax, linewidth=2)
    sns.lineplot(t2m, label="t2m", ax=ax, linewidth=2)
    sns.lineplot(target, label="target", ax=ax, linewidth=3.5, linestyle="--")
    plt.tight_layout()
    ax.set(xlabel=("Day of year"), ylabel="Normalised value")
    plt.savefig(f"/nvme/h/pgeorgiades/data_p143/UFPs/datasets/plots/test_{station_}.png")
    plt.close()
    
    
sns.set(font_scale=1.5)
for station in tqdm(target.station.unique()):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.lineplot(x=target.loc[(target.station == station) & 
                            (target.type == "obs")].time,
                 y= target.loc[(target.station == station) & 
                               (target.type == "obs")].value,
                 ax=ax, label=station)
    ax.set(xlabel="Date", ylabel="PM0.1")
    plt.tight_layout()
    plt.savefig(f"/nvme/h/pgeorgiades/data_p143/UFPs/data/UFP_Datasets/plots/pm01_{station}.png")
    plt.close()