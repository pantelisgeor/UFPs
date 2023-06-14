import os
import torch
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Custom dataset class to read in netcdf files
class NcDataset(Dataset):

    def __init__(self, parquet_file, root_dir, transform=None, views=None):
        """
        Custom Dataset object for netcdf files in pytorch
        
        Args:
            parquet_file (string): Path to the parquet file with the
                                   target values
            root_dir (string): Directory with all the netcdf files.
            tranform (callable, optional): Optional transform to be applied
                                           on the samples 
            views: User selected user variables (default: None). By default 
                   it returns all the channels.
        """
        self.target_value = pd.read_parquet(parquet_file)
        self.target_value = self.target_value.assign(
            names = [f"dat_{x}.nc" for x in self.target_value.index.values])
        self.root_dir = root_dir
        self.transform = transform
        self.views = views

    def __len__(self):
        return len(self.target_value)

    def __getitem__(self, idx):
        # netcdf names
        nc_name = os.path.join(self.root_dir,
                               self.target_value.iloc[idx, -1])
        # Read the dataset
        dat = xr.open_dataset(nc_name)
        # If some channels only are selected, subset them
        if self.views:
            dat = dat[self.views]
        # sample dictionary (convert numpy array to tensor)
        sample = {'nc': torch.tensor(dat.to_array().values.squeeze(), 
                                     dtype=torch.float32), 
                  'target': self.target_value.iloc[idx, -2],
                  'views': self.views}
        # Terminate the function
        return sample['nc'].float(), torch.tensor(sample['target'], 
                                                  dtype=torch.float32)


class NcDatasetMem(Dataset):

    def __init__(self, parquet_file, root_dir, transform=None, views=None):
        """
        Custom Dataset object for netcdf files in pytorch
        
        Args:
            parquet_file (string): Path to the parquet file with the
                                   target values
            root_dir (string): Directory with all the netcdf files.
            tranform (callable, optional): Optional transform to be applied
                                           on the samples 
            views: User selected user variables (default: None). By default 
                   it returns all the channels.
        """
        self.target_value = pd.read_parquet(parquet_file)
        self.target_value = self.target_value.assign(
            names = [f"dat_{x}.nc" for x in self.target_value.index.values],
            path = [f"{root_dir}/dat_{x}.nc" for x in self.target_value.index.values])
        self.root_dir = root_dir
        self.transform = transform
        self.views = views
        self.datasets = [xr.open_dataset(x) for x in\
            tqdm(self.target_value['path'].values)]

    def __len__(self):
        return len(self.target_value)

    def __getitem__(self, idx):
        # Read the dataset
        dat = self.datasets[idx]
        # If some channels only are selected, subset them
        if self.views:
            dat = dat[self.views]
        # sample dictionary (convert numpy array to tensor)
        sample = {'nc': torch.tensor(dat.to_array().values.squeeze(), 
                                     dtype=torch.float32), 
                  'target': self.target_value.iloc[idx, -3],
                  'views': self.views}
        # Terminate the function
        return sample['nc'].float(), torch.tensor(sample['target'], 
                                                  dtype=torch.float32)


class NcDatasetMem2(Dataset):

    def __init__(self, parquet_file, root_dir, transform=None, views=None):
        """
        Custom Dataset object for netcdf files in pytorch
        
        Args:
            parquet_file (string): Path to the parquet file with the
                                   target values
            root_dir (string): Directory with all the netcdf files.
            tranform (callable, optional): Optional transform to be applied
                                           on the samples 
            views: User selected user variables (default: None). By default 
                   it returns all the channels.
        """
        self.target_value = pd.read_parquet(parquet_file)
        self.target_value = self.target_value.assign(
            names = [f"dat_{x}.nc" for x in self.target_value.index.values],
            path = [f"{root_dir}/dat_{x}.nc" for x in self.target_value.index.values])
        self.root_dir = root_dir
        self.transform = transform
        self.views = views
        self.datasets = [xr.open_dataset(x) for x in\
            tqdm(self.target_value['path'].values)]

    def __len__(self):
        return len(self.target_value)

    def __getitem__(self, idx):
        # Read the dataset
        dat = self.datasets[idx]
        # If some channels only are selected, subset them
        if self.views:
            dat = dat[self.views]
        dat = dat.to_array().values.squeeze()
        dat = dat[:, 27:37, 27:37]
        # sample dictionary (convert numpy array to tensor)
        sample = {'nc': torch.tensor(dat.to_array().values.squeeze(), 
                                     dtype=torch.float32), 
                  'target': self.target_value.iloc[idx, -3],
                  'views': self.views}
        # Terminate the function
        return sample['nc'].float(), torch.tensor(sample['target'], 
                                                  dtype=torch.float32)
