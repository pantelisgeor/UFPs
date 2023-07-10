import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.create_model import convAutoEncoder, FeedForward, convAutoEncoder2
from src.data_loader import pm01data
from src.utils import str2bool, RMSELoss, train_loop, test_loop
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


path_dat = "/nvme/h/pgeorgiades/data_p143/UFPs/datasets/stations"
path_target = "/nvme/h/pgeorgiades/data_p143/UFPs/data/UFP_Datasets/Evaluation_Europe.nc"
# List the stations
stations = [x.split(".")[0] for x in os.listdir(path_dat)]
station_train = stations[:26]
station_test = stations[26:]


training_data = pm01data(path_dat=path_dat, path_target=path_target,
                         station_list=station_train)
test_data = pm01data(path_dat=path_dat, path_target=path_target,
                     station_list=station_test)

# DataLoader objects
train_dataloader = DataLoader(training_data, batch_size=256, 
                              shuffle=True)
# train_dataloader2 = DataLoader(training_data2, batch_size=128, shuffle=True)
# train_dataloader3 = DataLoader(training_data3, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256, 
                             shuffle=False)


model = convAutoEncoder2(channels=training_data.__getitem__(10)[0].shape[0])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}.")
model.to(device)
        
# Loss function definition
loss_fn = nn.MSELoss()  #RMSELoss()
        
# Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # , weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


for ep in range(4):
    print(f"Epoch {ep} \n")
    train_loop(train_dataloader, model, loss_fn, optimizer, device=device)
    # train_loop(train_dataloader2, model, loss_fn, optimizer)
    # train_loop(train_dataloader3, model, loss_fn, optimizer)
    print("Test score: ")
    test_loop(test_dataloader, model, loss_fn, device=device)


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
