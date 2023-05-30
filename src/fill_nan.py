import os
from glob import glob1
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial

path_dat = "/home/pantelisgeorgiades/Python/UFP/datasets/test1"

def repl_nan(f, train=True, path_dat=path_dat,
             path_ranges="/home/pantelisgeorgiades/Python/UFP/cams_clim_lims.csv"):
    # Read the dataset
    ds = xr.open_dataset(f"{'train' if train else 'test'}/{f}")
    # Read the limits dataframe
    df_lims = pd.read_csv(path_ranges)
    # df_lims = df_lims\
    #     .groupby('var', as_index=False)\
    #         .agg({'min': np.min, 'max': np.max})
    # Replace missing values with mean in the channel
    for vr in ds.variables:
        if np.sum(ds[vr].isnull()).item() > 0:
            ds[vr] = ds[vr].fillna(ds[vr].mean())
        del vr
    # List the variables which have nan    
    nan_list = []
    for vr in ds.variables:
        if np.sum(ds[vr].isnull()).item() > 0:
            nan_list.append(vr)
        del vr
    # List the variables
    var_list = [i for i in ds.variables]
    # Loop through the variables in the df_lim dataframe and
    # normalise the variables in ds
    for vr_lim in df_lims['var'].unique():
        for vr_ds in var_list:
            if vr_lim in vr_ds:
                # Get the limits
                min_temp = df_lims.loc[df_lims['var'] == vr_lim]['min'].item()
                max_temp = df_lims.loc[df_lims['var'] == vr_lim]['max'].item()
                ds[vr_ds] = (ds[vr_ds] - min_temp) / (max_temp - min_temp)
    if len(nan_list) > 0:
        print("ERROR")
        print(f)
        return None
    else:
        ds.to_netcdf(
            f"{path_dat.replace('datasets', 'datasets_')}/{'train' if train else 'test'}/{f}")
        

def replace_nan_all(path_dat):
    os.chdir(path_dat)

    # List the files
    dat_train = glob1("train/", "*.nc")
    dat_test = glob1("test/", "*.nc")
    
    process_map(repl_nan, dat_train, max_workers=10)
    process_map(partial(repl_nan, train=False), 
                dat_test, max_workers=10)


replace_nan_all(path_dat = "/home/pantelisgeorgiades/Python/UFP/datasets/test1")
replace_nan_all(path_dat = "/home/pantelisgeorgiades/Python/UFP/datasets/test2")
replace_nan_all(path_dat = "/home/pantelisgeorgiades/Python/UFP/datasets/test3")