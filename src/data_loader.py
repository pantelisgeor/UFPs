import os
import torch
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
import gc

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class pm01data(Dataset):
    
    def __init__(self, path_dat, path_target, station_list=None, views=None):
        # List the stations
        self.files = os.listdir(path_dat)
        self.stations = [x.split(".")[0] for x in os.listdir(path_dat)]
        
        # Read the target classes
        self.target = xr.open_dataset(path_target) 
        # Convert time to %d-%m-%y
        times = pd.to_datetime([datetime.datetime(2015, 1, 1)
                                + datetime.timedelta(int(x))
                                for x in self.target.time.values])
        # Replace the time variable of the xarray
        self.target['time'] = times
        self.target = self.target.to_dataframe().reset_index()
        self.target = pd.melt(self.target, id_vars="time")
        self.target = self.target.assign(
            time=pd.to_datetime(self.target.time.values),
            station=[x.split('_')[0] for x in self.target.variable.values],
            type=[x.split('_')[1] for x in self.target.variable.values])
        
        
        def get_station(station, path_dat=path_dat, target=self.target,
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
                ds = ds[views]
            ds = torch.tensor(ds.to_array().values.squeeze(), 
                              dtype=torch.float32)
            targ_ = torch.tensor(targ_.loc[targ_.type == "obs"].value.values,
                                 dtype=torch.float32)
            return ds[:, :89, :89 :], targ_
        
        # If station_list is None, load them all, else only load the
        # station names in the list
        self.stations_ = station_list if station_list is not None else self.stations
        for i, st in tqdm(enumerate(self.stations_)):
            if i == 0:
                st_dat, targ_dat = get_station(st)
            else:
                st_temp, targ_temp = get_station(st)
                st_dat = torch.concat((st_dat, st_temp), 3)
                targ_dat = torch.concat((targ_dat, targ_temp))
                del st_temp, targ_temp
                
        self.dat = st_dat
        self.target = targ_dat
        del i, st, st_dat, targ_dat
        gc.collect()
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.dat[:, :, :, idx], self.target[idx]
        


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
