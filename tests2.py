import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.create_model import convAutoEncoder, FeedForward, \
    convAutoEncoder2, convAutoEncoder3, ResNetConv, \
        ResNetConv2, ResNetConvSmall
from src.data_loader import pm01data, pm01data_
from src.utils import str2bool, RMSELoss, train_loop, test_loop
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


path_dat = "/nvme/h/pgeorgiades/data_p143/UFPs/datasets/stations2_cleaned"
path_target = "/nvme/h/pgeorgiades/data_p143/UFPs/datasets/Evaluation_cleaned.parquet"
# List the stations
stations = [x.split(".")[0] for x in os.listdir(path_dat)]
station_train = stations[:25]
station_test = stations[25:]

# views = [ "no", "no2", "d2m", "tp", "ssr", "pm1", "pm2p5", "t2m", "u10", "v10"] # ,
#         "land_1", "land_2", "land_3", "land_4", "land_5", "land_6","land_7",
#         "dem", "population", "air_dst", "air_br", "sea_dst", "sea_br",
#         "em_NOx", "em_CO2"]
views = ["no", "no2", "pm2p5", "t2m", "tp", "u10", "v10",
         "land_1", "land_2", "land_3", "land_4", "land_5", "land_6", "land_7"]

training_data = pm01data_(path_dat=path_dat, path_target=path_target,
                          station_list=station_train,                
                          views=views)
test_data = pm01data_(path_dat=path_dat, path_target=path_target,
                      station_list=station_test,
                      views=views)

train_dataloader = DataLoader(training_data, batch_size=512,
                              shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=512,
                             shuffle=False)

# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch


model = ResNetConvSmall(channels=len(views), resnet="ResNet152")  # ResNet101, ResNet152
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}.")
model.to(device)

# Loss function definition
loss_fn = RMSELoss()  #nn.MSELoss()
# Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-6)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for ep in range(1000):
    print(f"Epoch {ep} \n")
    train_loop(train_dataloader, model, loss_fn, optimizer, device=device)
    print("Test score: ")
    test_loop(test_dataloader, model, loss_fn, device=device)

test_dataloader = DataLoader(pm01data_(path_dat=path_dat,
                                       path_target=path_target,
                                       station_list=[station_train[-2]],
                                       views=views),
                             batch_size=1, shuffle=False)

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
plt.savefig("plots2/test_ResNet152_no_no2_pm25_t2m_tp_uv10_land_1000epochs_SMALL.png")  # _t2m_tp_uv10_land
plt.close()