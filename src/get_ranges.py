import os
from glob import glob
import xarray as xr
import rioxarray as rxr
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd
from multiprocessing import Pool
from functools import partial
from tqdm.contrib.concurrent import process_map


# ========================================================= #
# ======================= FUNCTIONS ======================= #
def get_lims(path_dat, dataset, extension, var, print_file=False):
    # List the datasets
    files = glob(f"{path_dat}{dataset}/{extension}")
    # Loop through the files and get the min and maximum value
    # from each. Then compare it to the global one and replace 
    # it min is smaller or max is larger.
    for f in tqdm(files):
        if print_file:
            print(f)
        ds = xr.open_dataset(f)
        # Put the data into a numpy ndarray
        try:
            dat_temp = ds[var].to_array()
        except AttributeError:
            dat_temp = ds[var].values
        # Get the initial values from the 1st dataset in the files list
        if f == files[0]:
            min_val = np.nanmin(dat_temp).item()
            max_val = np.nanmax(dat_temp).item()
            continue
        # Get the min and max values from the dataset f
        min_temp = np.nanmin(dat_temp).item()
        max_temp = np.nanmax(dat_temp).item()
        # Compare to the global ones and replace if necessary
        min_val = min_temp if min_temp < min_val else min_val
        max_val = max_temp if max_temp > max_val else max_val
        # House keeping (delete temp vars and garbage collection)
        del f, ds, min_temp, max_temp, dat_temp
        gc.collect()
    # Terminate function and return min and max
    return [min_val, max_val]


def min_max(f, var):
    ds = xr.open_dataset(f)
    # Put the data into a numpy ndarray
    try:
        dat_temp = ds[var].to_array()
    except AttributeError:
        dat_temp = ds[var].values
    # Get the min and max values from the dataset f
    min_temp = np.nanmin(dat_temp).item()
    max_temp = np.nanmax(dat_temp).item()
    # Garbage collection
    del ds, dat_temp
    gc.collect()
    return pd.DataFrame({"min": [min_temp], "max": [max_temp]})


def get_lims_par(path_dat, dataset, extension, var, max_workers=6):
    # List the datasets
    files = glob(f"{path_dat}/{dataset}/{extension}")
    # Loop through the files and get the min and maximum value
    # from each. Then compare it to the global one and replace
    # it min is smaller or max is larger.
    df_ = process_map(partial(min_max, var=var),
                      files, max_workers=max_workers)
    df_ = pd.concat(df_)
    # Terminate function and return min and max
    return df_.assign(var=var)


# ========================================================= #
# ========================================================= #
# Path to datasets
path_dat = "/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/UFPs/data"

# CAMS_NO2 Datasets
# Dataset relative path to path_dat and extension of data
dataset = "/CAMS/NO2"
extension = "*.nc"
# Vars
var = 'no2'
cams_lims_no2 = get_lims_par(path_dat, dataset, extension, var, max_workers=10)

# CAMS_NO Datasets
# Dataset relative path to path_dat and extension of data
dataset = "/CAMS/NO"
extension = "*.nc"
# Vars
var = 'no'
cams_lims_no = get_lims_par(path_dat, dataset, extension, var, max_workers=10)

# CAMS_CO Datasets
# Dataset relative path to path_dat and extension of data
dataset = "/CAMS/CO"
extension = "*.nc"
# Vars
var = 'co'
cams_lims_co = get_lims_par(path_dat, dataset, extension, var, max_workers=10)

cams_lims = pd.concat([cams_lims_no2, cams_lims_no, cams_lims_co], axis=0)
cams_lims.to_csv(f"{path_dat}/cams_lims.csv", index=False)

# ERA5L
datasets = ["d2m", "t2m", "tp", "solar_net", "thermal_net", "u_wind", "v_wind"]
extension = "*.grib"
vars = ["d2m", "t2m", "tp", "ssr", "ssr", "u10", "v10"]


for idx, d in enumerate(datasets):
    print(f"Processing {d}\n")
    if idx == 0:
        df_era = df_temp = get_lims_par(path_dat=path_dat,
                                        dataset=f"/ERA5_land/{d}",
                                        extension="*.grib",
                                        var=vars[idx])
        del df_temp
    else:
        df_era = pd.concat([df_era,
                            get_lims_par(path_dat=path_dat,
                                         dataset=f"/ERA5_land/{d}",
                                         extension="*.grib",
                                         var=vars[idx])])
    # Garbage collection
    gc.collect()

df = pd.concat([cams_lims, df_era])
df.to_csv(f"{path_dat}/lims.csv", index=False)