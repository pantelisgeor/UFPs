import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.create_model import convAutoEncoder, FeedForward, convAutoEncoder2
from src.data_loader import NcDataset, NcDatasetMem, NcDatasetMem2
from src.utils import str2bool, RMSELoss, train_loop, test_loop
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt



# Datasets (training and test)
views=['land_1', 'land_2', 'land_3', 'land_4', 'land_5', 'land_6', 'land_7',
       "t2m0", "d2m0", "tp0", "u100", "v100", "no0", "no20", "co0"]

# Datasets (training and test)
# views=["land_1", "land_2", "land_3", "land_4", "land_5", 
#        "t2m0", "u100", "v100", "tp0", "no0", "no20"]

# views = ["no-2", "no2-2", "no0", "no20", "no2", "no22"]

training_data = NcDatasetMem(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/train.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/train",
    views=views)

"""
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
"""

test_data = NcDatasetMem(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/test.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets_/test1/test",
    views=views)

# DataLoader objects
train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)
# train_dataloader2 = DataLoader(training_data2, batch_size=512, shuffle=True)
# train_dataloader3 = DataLoader(training_data3, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False)



model = convAutoEncoder(channels=len(views))
model.to("cuda")
        
 # Loss function definition
loss_fn = nn.MSELoss()  #RMSELoss()
        
# Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # , weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


for ep in range(3):
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
plt.tight_layout()
plt.savefig("../test2.png")
plt.close()