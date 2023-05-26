import os
import torch
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Custom dataset class to read in netcdf files
class NcDataset(Dataset):

    def __init__(self, parquet_file, root_dir, transform=None):
        """
        Custom Dataset object for netcdf files in pytorch
        
        Args:
            parquet_file (string): Path to the parquet file with the
                                   target values
            root_dir (string): Directory with all the netcdf files.
            tranform (callable, optional): Optional transform to be applied
                                           on the samples 
        """
        self.target_value = pd.read_parquet(parquet_file)
        self.target_value = self.target_value.assign(
            names = [f"dat_{x}.nc" for x in self.target_value.index.values])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.target_value)

    def __getitem__(self, idx):
        # netcdf names
        nc_name = os.path.join(self.root_dir,
                               self.target_value.iloc[idx, -1])
        # Read the dataset
        dat = xr.open_dataset(nc_name).to_array().values.squeeze()
        # sample dictionary (convert numpy array to tensor)
        sample = {'nc': torch.tensor(dat), 'target': self.target_value.iloc[idx, -2]}
        # Terminate the function
        return sample


# Datasets (training and test)
training_data = NcDataset(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test1/train.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test1/train")
test_data = NcDataset(
    parquet_file="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test1/test.parquet",
    root_dir="/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test1/test"
)

# DataLoader objects
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)