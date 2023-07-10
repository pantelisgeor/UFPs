# Standard library imports
import os
from glob import glob
import re
import gc
import math
import pickle
from itertools import repeat
from functools import partial
from multiprocessing import Pool

# External libraries imports
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray as rxr
from tqdm import tqdm


# ================================================================ #
path_dat = "/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/UFPs/data"
os.chdir(path_dat)
# Read the airports and seaports datasets
df_air = pd.read_csv(f"{path_dat}/ports/Global_ports/airport-locations.csv")
df_sea = pd.read_csv(f"{path_dat}/ports/Global_ports/seaport-locations.csv")


# ================================================================ #
def parse_era(x):
    """
    Parse the information of the ERA5-Land dataset based on the path
    to the .grib file
    """
    return pd.DataFrame({"variable": [x.split("/")[-2]],
                         "month": [int(x.split("/")[-1]\
                             .split("_")[0])],
                         "year": [int(x.split("/")[-1]\
                             .split("_")[1].split(".")[0])],
                         "path": [x]})


def list_era(path_era):
    """
    List the ERA5-Land datasets in the directory path_era
    """
    # List the variables in the directory
    variables = [x.split('/')[-2] for x in glob(f"{path_era}/*/")]
    df_info = pd.DataFrame()
    # Loop through the variables and create the info dataframe
    for v in tqdm(variables):
        # List the directory (get the info for the var)
        df_temp = pd.concat(map(parse_era, glob(f"{path_era}/{v}/*.grib")))
        # Append to the df_info dataframe
        df_info = pd.concat([df_info,
                             df_temp.sort_values(by=["year", "month"], 
                                                 ascending=True)])
    return df_info.reset_index(drop=True)


def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    # Terminate function
    return d


def get_bearing(lat1, long1, lat2, long2):
    """
    Function to calculate the bearing between two coordinates
    defined by lat/lon.
    """
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) \
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2))\
            * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    
    return brng


def get_ports(station, df_air=df_air, df_sea=df_sea):
    """
    Function to calculate the distance and bearing to the nearest
    airport and sea port. 
    
    Args:
        station: coordinates of ground station (lat, lon)
        df_air: dataframe with the airports information
        df_sea: dataframe with the sea ports information
    Returns:
        pandas dataframe with the coordinates of the station and the
        distance and bearings to the closest airport and sea port.
    """
    # Get the coordinates from the airports and seaports into a numpy array
    coords_air = df_air[["latitude", "longitude"]].values
    coords_sea = df_sea[["latitude", "longitude"]].values
    # Calculate the coords of the closest one (by distance)
    air_closest = coords_air[np.argmin([distance(station, x) for x in coords_air])]
    sea_closest = coords_sea[np.argmin([distance(station, x) for x in coords_sea])]
    # Calculate the azimuth to both
    br_air = get_bearing(station[0], station[1], air_closest[0], air_closest[1])
    br_sea = get_bearing(station[0], station[1], sea_closest[0], sea_closest[1])
    # Calculate their distance
    dist_air, dist_sea = distance(station, air_closest), distance(station, sea_closest)
    return pd.DataFrame({"lat": [station[0]], "lon": [station[1]],
                         "air_dst": [dist_air], "air_br": [br_air],
                         "sea_dst": [dist_sea], "sea_br": [br_sea]})


def get_all_ports(coords, path_dat, n_jobs=15):
    """
    Calculates the distance and bearing to the closest airport
    and sea port for all the points in a grid. It uses the multiprocessing 
    backend to perform the calculations in parallel for speed improvements.
    
    Args:
        coords: List that holds the latitudes and longitudes
                coordinates in the form of [[lats], [lons]]
        path_dat: Path to directory where data is stored
        n_jobs: number of parallel processes to use to perform calculations
    """
    # Read the airports and sea ports datasets in pandas dataframes
    df_air = pd.read_csv(f"{path_dat}/ports/Global_ports/airport-locations.csv")
    df_sea = pd.read_csv(f"{path_dat}/ports/Global_ports/seaport-locations.csv")
    # Create a list with all the combinations of lats/lons
    coord_list = [[lat, lon] for lat in coords[0] for lon in coords[1]]
    # Calculate the distance and bearings (using multiprocessing backend)
    pool = Pool(n_jobs)
    dists = pd.concat(pool.map(partial(get_ports, df_air=df_air, df_sea=df_sea), 
                      coord_list))
    pool.close()
    pool.join()
    # Convert to xarray dataset
    ds_dists = dists.rename(columns={"lat": "y", "lon": "x"})\
        .set_index(["x", "y"]).to_xarray()
    return ds_dists


def dem(station, path_dat=path_dat, return_point=False,
        grid_size=64):
    """
    Read the digital elevation dataset and return 
    a 7x7 km box around it.
    
    Args:
        station: (lat, lon) point
        path_dat: Path to directory where data is stored
        return_point: Boolean (default=False) to return only the grid cell
                      in which the station point lies in
        grid_size: If return_point is False, it returns a grid with size
                   grid_size around the station point
    """
    # Read the digital elevation dataset
    ds_dem = rxr.open_rasterio(f"{path_dat}/dem/elevation_1KMmd_GMTEDmd.tif")
    # Create a list with the closest coordinates in the ds_dem dataset
    # to find the grid cell in which the station lies in
    ds_temp = ds_dem.sel(y=slice(station[0]+0.4, station[0]-0.4),
                         x=slice(station[1]-0.4, station[1]+0.4))
    coords_dem = [[y, x] for y in ds_temp.y.values for x in ds_temp.x.values]
    coords_closest = coords_dem[np.argmin([distance(station, x) for x in coords_dem])]
    if return_point:
        return ds_temp.sel(x=coords_closest[1], y=coords_closest[0]).values[0]
    else:
        # Get the index of the lat and lon from ds_temp to the corresponding
        # grid cell in which station lies in
        y_idx = int(np.where(ds_temp.y.values == coords_closest[0])[0])
        x_idx = int(np.where(ds_temp.x.values == coords_closest[1])[0])
        # Get the lats and lons around the station with specified grid_size 
        lats = ds_temp.y.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
        lons = ds_temp.x.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
        return ds_temp.sel(x=slice(lons.min(), lons.max()),
                           y=slice(lats.max(), lats.min()))


def get_pop(station, year, path_dat=path_dat,
            return_point=False, grid_size=8):
    """
    Retrieve the population dataset for the specified coordinates and year
    
    Args:
        station: (lat, lon) point
        path_dat: Path to directory where data is stored
        return_point: Boolean (default=False) to return only the grid cell
                      in which the station point lies in
        grid_size: If return_point is False, it returns a grid with size
                   grid_size around the station point
    """
    # If the year is not in the 2000-2020 range, get the closest year
    year = 2000 if year < 2000 else year
    year = 2020 if year > 2020 else year
    # Read the corresponding dataset for the year
    ds_pop = rxr.open_rasterio(f"{path_dat}/population/world_pop/ppp_{int(year)}_1km_Aggregated.tif")
    # Get the grid cells close to the station point
    ds_pop = ds_pop.sel(y=slice(station[0]+0.2, station[0]-0.2),
                        x=slice(station[1]-0.2, station[1]+0.2))
    coords_pop = [[y, x] for y in ds_pop.y.values for x in ds_pop.x.values]
    coords_closest = coords_pop[np.argmin([distance(station, x) for x in coords_pop])]
    if return_point:
        return ds_pop.sel(x=coords_closest[1], y=coords_closest[0]).values[0]
    else:
        # Get the index of the lat and lon from ds_temp to the corresponding
        # grid cell in which station lies in
        y_idx = int(np.where(ds_pop.y.values == coords_closest[0])[0])
        x_idx = int(np.where(ds_pop.x.values == coords_closest[1])[0])
        # Get the lats and lons around the station with specified grid_size 
        lats = ds_pop.y.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
        lons = ds_pop.x.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
        return ds_pop.sel(x=slice(lons.min(), lons.max()),
                          y=slice(lats.max(), lats.min()))


def get_land(station, year, path_dat=path_dat, return_point=False,
             grid_size=80):
    """
    Reads the land use dataset and returns the data corresponding to the specified
    point (defined by (lat, lon))
    
    Args:
        station: (lat, lon) point
        path_dat: Path to directory where data is stored
        return_point: Boolean (default=False) to return only the grid cell
                      in which the station point lies in.
        grid_size: If return_point is False, it returns a grid with size grid_size
                   around the station point.
        year: Year to retrieve land use data
    """
    # If year is less than 2015 or larger than 2019, set the limits
    year = year if year > 2015 else 2015
    year = year if year < 2019 else 2019
    # List the datasets in the land use directory
    land_files = glob(f"{path_dat}/land_use/*.tif")
    # Create a list with the year of each dataset in the land_files list of filenames
    land_years = [int(re.search(r"_[0-9]{4}", x).group(0).replace("_", "")) for x in land_files]
    # Get the dataset corresponding to the year
    land_dat = land_files[land_years.index(year)]
    # Read the land use dataset
    ds_land = rxr.open_rasterio(land_dat)
    # Create a list with the closest coordinates in the ds_dem dataset
    # to find the grid cell in which the station lies in
    ds_temp = ds_land.sel(y=slice(station[0]+0.1, station[0]-0.1),
                          x=slice(station[1]-0.1, station[1]+0.1))
    coords_land = [[y, x] for y in ds_temp.y.values for x in ds_temp.x.values]
    coords_closest = coords_land[np.argmin([distance(station, x) for x in coords_land])]
    if return_point:
        return ds_temp.sel(x=coords_closest[1], y=coords_closest[0]).values[0]
    else:
        # Get the index of the lat and lon from ds_temp to the corresponding
        # grid cell in which station lies in
        y_idx = int(np.where(ds_temp.y.values == coords_closest[0])[0])
        x_idx = int(np.where(ds_temp.x.values == coords_closest[1])[0])
        # Get the lats and lons around the station with specified grid_size 
        lats = ds_temp.y.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
        lons = ds_temp.x.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
        return ds_temp.sel(x=slice(lons.min(), lons.max()),
                        y=slice(lats.max(), lats.min()))


def map_groups(data, groupings):
    """
    Maps the copernicus land use classes to a new classification system
    which groups the relevant classes together (from 23 classes to 7)
    
    Args:
        data: 2d numpy array (i.e. ds.values).
        groupings: dictionary with the new grouped classes.
    """
    data = np.stack(
             [np.stack(
                [data == int(x) for x in grouping])\
                    .sum(axis=0)\
                        .astype(bool) * i for i, grouping in groupings.items()])\
                .sum(axis=0
                )
    return data


def era_dat(station, month, year, path_dat,
            variables=["t2m", "d2m", "tp", "solar_net", "u_wind", "v_wind"],
            return_point=False, grid_size=8):
    """
    Constructs the ERA5-land dataset for the specified coordinate (station)
    for a given month and year and a set of variables.
    
    Args:
        station: (lat, lon) which defines the point for which to get the data
        month: Month of the year to get the data for
        year: Year to get the data for
        path_era: Path to directory where the ERA5L data is stored
        variables: List of variables to retrieve
        return_point: Boolean (default=False) to return only the grid cell
                      in which the station point lies in.
        grid_size: If return_point is False, it returns a grid with size grid_size
                   around the station point.
    """
    # Construct the filename given the month and year
    filename = f"{int(month)}_{year:4d}.grib"
    # Path to directory
    path_era = f"{path_dat}/ERA5_land"
    # Loop through the variables, load the grib file corresponding
    # to the specified month and year and put them together
    for v in tqdm(variables):
        ds_temp = xr.open_dataset(f"{path_era}/{v}/{filename}",
                                  engine="cfgrib")
        if v == variables[0]:
            ds_era = ds_temp
        else:
            ds_era = ds_era.merge(ds_temp)
        del ds_temp, v
        gc.collect()
    # If the longitude is in the range 0-360 -> Change it to -180 to 180
    ds_era = ds_era\
        .assign_coords(longitude=(((ds_era.longitude + 180) % 360) - 180))\
            .sortby('longitude')
    # Create a list with the closest coordinates in the ds_dem dataset
    # to find the grid cell in which the station lies in
    ds_temp = ds_era.sel(latitude=slice(station[0]+0.2, station[0]-0.2),
                         longitude=slice(station[1]-0.2, station[1]+0.2))
    coords_era = [
        [y, x] for y in ds_temp.latitude.values for x in ds_temp.longitude.values]
    coords_closest = coords_era[np.argmin([distance(station, x) for x in coords_era])]
    if return_point:
        return ds_temp.sel(x=coords_closest[1], y=coords_closest[0]).values[0]
    else:
        # Get the index of the lat and lon from ds_temp to the corresponding
        # grid cell in which station lies in
        y_idx = int(np.where(ds_temp.latitude.values == coords_closest[0])[0])
        x_idx = int(np.where(ds_temp.longitude.values == coords_closest[1])[0])
        # Get the lats and lons around the station with specified grid_size 
        lats = ds_temp.latitude.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
        lons = ds_temp.longitude.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
        return ds_temp.sel(longitude=slice(lons.min(), lons.max()),
                           latitude=slice(lats.max(), lats.min()))


def getCAMS(station, month, year, path_dat, pollutant="NO2",
            return_point=False, grid_size=8):
    """
    
    """
    # Construct the path to the pollutant's directory
    path_cams = f"{path_dat}/CAMS/{pollutant}"
    # List the netcdf files in the directory
    filenames = sorted(glob(f"{path_cams}/*.nc"))
    months = [int(x.split('.')[-2].split('-')[-1]) for x in filenames]
    years = [int(x.split('.')[-2].split('-')[-2]) for x in filenames]
    cams_files = pd.DataFrame({"month": months,
                               "year": years,
                               "path": filenames})
    # Read the file for the corresponding month
    try:
        ds_cams = xr.open_dataset(cams_files.loc[(cams_files.year == year) &
                                                 (cams_files.month == month)]\
                                                     .path.values[0])
    except Exception as ex:
        print(ex)
    # Create a list with the closest coordinates in the ds_dem dataset
    # to find the grid cell in which the station lies in
    ds_temp = ds_cams.sel(lat=slice(station[0]-0.2, station[0]+0.2),
                          lon=slice(station[1]-0.2, station[1]+0.2))
    coords_cams = [
        [y, x] for y in ds_temp.lat.values for x in ds_temp.lon.values]
    coords_closest = coords_cams[np.argmin([distance(station, x) for x in coords_cams])]
    if return_point:
        return ds_temp.sel(x=coords_closest[1], y=coords_closest[0]).values[0]
    else:
        # Get the index of the lat and lon from ds_temp to the corresponding
        # grid cell in which station lies in
        y_idx = int(np.where(ds_temp.lat.values == coords_closest[0])[0])
        x_idx = int(np.where(ds_temp.lon.values == coords_closest[1])[0])
        # Get the lats and lons around the station with specified grid_size
        lats = ds_temp.lat.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
        lons = ds_temp.lon.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
        return ds_temp.sel(lon=slice(lons.min(), lons.max()),
                           lat=slice(lats.min(), lats.max()))


def get_centre(coord, grid_size_lon=0.1, grid_size_lat=0.1):
    """
    The ERA5L grid points are placed on the top left corner of the grid cell.
    This returns the coordinate of the centre of the grid cell.
    Ref: https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference
    """
    return [round(coord[0]-(grid_size_lat/2), 4), round(coord[1]+(grid_size_lon/2), 4)]


def get_era_point(df_era, lat, lon, time_):
    """
    
    """
    # Get the coordinates of the centre of the closest grid cell
    close_temp = df_closest\
        .loc[(df_closest.lat == lat) & (df_closest.lon == lon)][["lat_cen", "lon_cen"]]\
            .values[0]
    # Get the specified point
    df_temp = df_era.loc[(df_era.lat_centre == close_temp[0]) & 
                        (df_era.lon_centre == close_temp[1]) & 
                        (df_era.time == time_)]\
                            .drop(['latitude', 'longitude', 
                                    'lat_centre', 'lon_centre',
                                    "time"], axis=1)
    return df_temp



# ================================================================ #
# ================================================================ #
# Read the dataset
df = pd.read_parquet(f"{path_dat}/stations/zenodo2/test3.parquet")
df = df.rename(columns={"date": "time", "latitude": "lat", "longitude": "lon"})

# List the unique stations
stations = df.groupby(["lat", "lon"], as_index=False).size()

path_save = "/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/datasets/test3/test"
if not os.path.isdir(path_save):
    os.mkdir(path_save)


# ================================================================ #
# ================================================================ #
# Get the 1st station coords
station = stations[["lat", "lon"]].values[0]
# Subset for station 1
df2 = df.loc[(df.lat == station[0]) & (df.lon == station[1])]
# Add a column for the year and month for each row
df2 = df2.assign(year=[int(x.split("-")[0]) for x in df2.time.values],
                 month=[int(x.split('-')[1]) for x in df2.time.values])
# Convert time to pandas datetime object
df2 = df2.assign(time=pd.to_datetime(df2.time.values))

# Aggregate to hourly means
df2 = df2.drop('Location.ID', axis=1)\
    .groupby(["lat", "lon", pd.Grouper(key="time", freq="H")], as_index=False)\
        .mean()

df2 = df2.loc[df2.year <= 2020]

# List the ERA5L available datasets
df_era = list_era("/nvme/h/pgeorgiades/data_p102/pantelis/UFPs/UFPs/data/ERA5_land")


# ==================================================================== #
# ======================= Dataset construction ======================= #
# -------------------------------------------------------------------- #

# ======================= LAND USE ===================== #
# List the years in the ground station dataset
years = df2.year.unique()
# Read the copernicus classes mappings
with open(f"{path_dat}/land_use/copernicus_mappings.pickle", "rb") as f:
    cop_map = pickle.load(f)
# First get the land use data
for yr in years:
    ds_temp = get_land(station, path_dat=path_dat, return_point=False,
                       grid_size=100, year=yr)\
                           .sel(band=1)
    # Map the land use classes to the new grouped classes
    ds_temp.values = map_groups(ds_temp.values, cop_map)
    # Add the year variable
    ds_temp = ds_temp.to_dataset(name="land_use").expand_dims(year=[yr])
    if yr == years[0]:
        ds_land = ds_temp
    else:
        ds_land = ds_land.merge(ds_temp)
    del yr, ds_temp
    gc.collect()
# If band is in the coordinate list, drop it
try:
    ds_land = ds_land.drop("band")
except Exception as e:
    print("band not in coordinate list")
# Extract the coordinates of the land use dataset to use as a base for the rest
# Will be stored in a list of [[lats], [lons]]
land_coords = [ds_land.y.values, ds_land.x.values]

# Get the pixel size for the land use dataset
sizex, sizey = np.mean(np.diff(ds_land.x.values)), \
    np.mean(np.diff(ds_land.y.values))
    
# Create the binary masks for the land use
for i in np.arange(1, 8, 1):
    land_temp = ds_land.land_use.values
    ds_land[f"land_{i}"] = xr.DataArray(np.where(land_temp == i, 1, 0),
                                        dims=("year", "y", "x"))
    
# Drop the classes
ds_land = ds_land.drop("land_use")

# =========================== POPULATION ======================== #
# Next get the population dataset
for yr in years:
    ds_temp = get_pop(station=station, year=yr, grid_size=20).sel(band=1)
    # Add the year variable
    ds_temp = ds_temp.to_dataset(name="pop").expand_dims(year=[yr])
    if yr == years[0]:
        ds_pop = ds_temp
    else:
        ds_pop = ds_pop.merge(ds_temp)
    del yr, ds_temp
    
gc.collect()

# Drop the band coordinate, if present
try:
    ds_pop = ds_pop.drop("band")
except Exception as e:
    print("The band coordinate was not present in the population dataset")

# Spatially interpolate the population data to match the land grid
ds_pop = ds_pop.interp(y=land_coords[0], x=land_coords[1], method="nearest")

# Merge them
ds = ds_land.merge(ds_pop)

# ==================== DISTANCE TO PORTS ===================== #
ds_dists = get_all_ports(coords=land_coords, path_dat=path_dat)

# Merge them
ds = ds_land.merge(ds_dists)
del ds_dists, ds_pop
gc.collect()


# ========================== ERA5-Land ======================= #
# Get the years and months combinations from the dataset
era_dates = df2.groupby(["month", "year"], as_index=False)\
    .size()\
        .drop("size", axis=1)\
            .sort_values(by=["year", "month"], ascending=True)\
                .reset_index(drop=True)


for month, year in era_dates[["month", "year"]].values:
    print(f"ERA5L: {int(year)} - {int(month)}")
    ds_temp = era_dat(station, int(month), int(year),
                      path_dat=path_dat,
                      variables=["t2m", "d2m", "tp", "solar_net",
                                 "u_wind", "v_wind"],
                      return_point=False, grid_size=10)
    if month == era_dates['month'].values[0]:
        ds_era = ds_temp
    else:
        ds_era = ds_era.merge(ds_temp)
    del ds_temp, month, year
    gc.collect()


# ========================== CAMS ======================= #
c = 0
for month, year in era_dates[["month", "year"]].values:
    print(f"CAMS: {int(year)} - {int(month)}")
    for pollutant in ["NO2", "NO", "CO"]:
        ds_temp = getCAMS(station=station, month=int(month),
                        year=int(year), path_dat=path_dat,
                        grid_size=10, pollutant=pollutant)
        if c == 0:
            ds_cams = ds_temp
            c += 1
        else:
            ds_cams = ds_cams.merge(ds_temp)
    del ds_temp, month, year
    gc.collect()

del c

# Interpolate to the land coordinates
ds_cams = ds_cams.interp(lat=land_coords[0],
                         lon=land_coords[1],
                         method="nearest")
ds_cams = ds_cams.rename({"lat": "y", "lon": "x"})


# ========================== Hourly dataset construction ========================== #

# ERA5L
# Get the grid size for the ERA5L dataset
era_x, era_y = round(np.mean(np.diff(ds_era.longitude.values)), 3), \
    round(np.mean(np.diff(ds_era.latitude.values)), 3)
# Convert the ERA5L dataset to dataframe
df_era = ds_era.to_dataframe().reset_index(drop=False)
# Add the centre of each grid cell
df_era = pd.concat([df_era, pd.DataFrame(
    [get_centre(x) for x in tqdm(df_era[['latitude', 'longitude']].values)],
    columns=["lat_centre", "lon_centre"])], axis=1)
# Unique set of grid cell centre coordinates
centre_coords = df_era.groupby(['lat_centre', 'lon_centre'], as_index=False)\
    .size().drop('size', axis=1)[['lat_centre', "lon_centre"]].values
# Find the closest ERA5 grid cell to each of the land_use grid points
coords = [[y, x] for y in land_coords[0] for x in land_coords[1]]
closest_points = [centre_coords[np.argmin([distance(coord, x) for x in centre_coords])] \
    for coord in coords]
df_closest = pd.concat([pd.DataFrame(coords, columns=["lat", "lon"]),
                        pd.DataFrame(closest_points,
                                     columns=["lat_cen", "lon_cen"])], axis=1)
del closest_points, centre_coords
gc.collect()

# Drop the time and step columns and convert valid_time to pandas datetime 
# (and name it time to match the initial dataset)
df_era = df_era.drop(["time", "step"], axis=1)\
    .assign(time=pd.to_datetime(df_era.valid_time))\
        .drop("valid_time", axis=1)
# If number of surface in columns, drop them
try:
    df_era.drop('number', axis=1, inplace=True)
except Exception as e:
    pass

try:
    df_era.drop("surface", axis=1, inplace=True)
except Exception as e:
    pass

# ================================================================================= #
# Find closest point to station from land_coords
land_centre = coords[
    np.argmin([distance(station, coord) for coord in coords])
]
idx_x = np.where(land_coords[1] == land_centre[1])[0][0]
idx_y = np.where(land_coords[0] == land_centre[0])[0][0]

# ================================================================================= #


def get_day(dataset_index, time, path_save,
            time_steps= [-8, -4, -2, 0, 2, 4, 8], centre_size=10,
            idx_x=idx_x, idx_y=idx_y, grid_size=64,
            ds=ds, df_era=df_era, ds_cams=ds_cams):
    """
    
    """
    if os.path.isfile(f"{path_save}/dat_{dataset_index}.nc"):
        print(f"{dataset_index} already processed")
        return None
    try:
        print(dataset_index)
        for t_ in time_steps:
            # ERA5:
            df_temp = pd.concat(
                [get_era_point(df_era,
                               lat, lon,
                               time + np.timedelta64(t_, "h")) for lat, lon in coords])
            df_temp.columns = [f"{x}{t_}" for x in df_temp.columns]
            df_temp = pd.concat([df_temp.assign(time=time).reset_index(drop=True),
                                pd.DataFrame(coords, columns=['lat', 'lon'])], axis=1)
            # Rename lat, lon to y, x
            df_temp = df_temp.rename(columns={"lat": "y", "lon": "x"})
            # Convert it to xarray dataset (gridded dataset)
            ds_temp = df_temp.set_index(['time', 'y', 'x']).to_xarray()
            # CAMS
            ds_cams_temp = ds_cams.sel(time=time + np.timedelta64(t_, "h"))\
                .rename({"no2": f"no2{t_}"})
            if 'no' in ds_cams.variables:
                ds_cams_temp = ds_cams_temp.rename({"no": f"no{t_}"})
            if 'co' in ds_cams.variables:
                ds_cams_temp = ds_cams_temp.rename({"co": f"co{t_}"})
            if t_ == time_steps[0]:
                ds_ = ds_temp.merge(ds_cams_temp)
            else:
                ds_ = ds_.merge(ds_temp.merge(ds_cams_temp))
            del t_, ds_temp, df_temp, ds_cams_temp
        # Merge with the other variables
        ds_ = ds_.merge(ds.sel(year=float(str(time).split('-')[0])))
        # Random sample in the centre (centre defined by a square of size centre_size)
        # If centre_size is not an even number add one
        if centre_size % 2 != 0:
            centre_size += 1
            print("Adding 1 to centre_size to make it even.")
        # Calculate a random offset from the centre
        offset_x = np.random.randint(-centre_size/2, centre_size/2 + 1)
        offset_y = np.random.randint(-centre_size/2, centre_size/2 + 1)
        # Adjust the x and y indices accordingly
        idx_x = idx_x + offset_x
        idx_y = idx_y + offset_y
        # Get the coordinate slices
        lons = slice(ds_.x.values[int(idx_x-(grid_size/2)+1)],
                     ds_.x.values[int(idx_x+(grid_size/2))])
        lats = slice(ds_.y.values[int(idx_y-(grid_size/2)+1)],
                     ds_.y.values[int(idx_y+(grid_size/2))])
        # Slice the dataset
        ds_ = ds_.sel(x=lons, y=lats)
        # Terminate function
        if path_save:
            ds_.to_netcdf(f"{path_save}/dat_{dataset_index}.nc")
        else:
            return ds_
    except Exception as e:
        print(f"no {dataset_index} FAILED because of \n\n{e}\n")


pool = Pool(40)
pool.starmap(get_day,
             zip(np.arange(0, len(df2.time.values)),
                 df2.time.values, repeat(path_save)))
pool.close()
pool.join()
