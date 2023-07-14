import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.create_model import convAutoEncoder, FeedForward, \
    convAutoEncoder2, convAutoEncoder3, ResNetConv
from src.data_loader import pm01data, pm01data_
from src.utils import str2bool, RMSELoss, train_loop, test_loop
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


path_dat = "/nvme/h/pgeorgiades/data_p143/UFPs/datasets/stations2_cleaned"
path_target = "/nvme/h/pgeorgiades/data_p143/UFPs/datasets/Evaluation_cleaned.parquet"
# List the stations
stations = [x.split(".")[0] for x in os.listdir(path_dat)]
station_train = stations[:28]
station_test = stations[28:]

views = [ "no", "no2", "d2m", "tp", "ssr", "pm1", "pm2p5", "t2m", "u10", "v10",
         "land_1", "land_2", "land_3", "land_4", "land_5", "land_6","land_7",
         "dem", "population", "air_dst", "air_br", "sea_dst", "sea_br", 
         "em_NOx", "em_CO2"]
training_data = pm01data_(path_dat=path_dat, path_target=path_target,
                          station_list=station_train,
                          views=views)
test_data = pm01data_(path_dat=path_dat, path_target=path_target,
                      station_list=station_test,
                      views=views)

# DataLoader objects
train_dataloader = DataLoader(training_data, batch_size=512,
                              shuffle=True)
# train_dataloader2 = DataLoader(training_data2, batch_size=128, shuffle=True)
# train_dataloader3 = DataLoader(training_data3, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=512,
                             shuffle=False)

# TEST THIS OUT
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

# model = convAutoEncoder3(channels=training_data.__getitem__(0)[0].shape[0])
model = ResNetConv(channels=len(views), resnet="ResNet50")  # ResNet101, ResNet152
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}.")
model.to(device)

# Loss function definition
loss_fn = RMSELoss()  #nn.MSELoss()

# Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


for ep in range(200):
    print(f"Epoch {ep} \n")
    train_loop(train_dataloader, model, loss_fn, optimizer, device=device)
    # train_loop(train_dataloader2, model, loss_fn, optimizer)
    # train_loop(train_dataloader3, model, loss_fn, optimizer)
    print("Test score: ")
    test_loop(test_dataloader, model, loss_fn, device=device)


test_dataloader = DataLoader(pm01data_(path_dat=path_dat, 
                                       path_target=path_target,
                                       station_list=[station_train[2]],
                                       views=views),
                             batch_size=1, shuffle=False)

# train_dataloader_Test = DataLoader(training_data, batch_size=1, shuffle=False)

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
plt.savefig("test_resnet50.png")
plt.close()

plt.show()




        
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