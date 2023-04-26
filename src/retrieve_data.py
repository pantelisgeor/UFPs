from pathlib import Path
import argparse
import cdsapi
import wget
import os


# ====================================================================== #
# ========================== Argument parser =========================== #
parser = argparse.ArgumentParser()

parser.add_argument("data_root", type=str, 
                    help="Path to directory where data is stored")

args = parser.parse_args()
# ====================================================================== #


def get_population(data_root=args.data_root, year=2020):
    """
    Downloads the 1 km world population dataset from worldpop.
    https://hub.worldpop.org/geodata
    """
    # Check if year is in the available range
    if year < 2000 or year > 2020:
        raise ValueError("Year outside of available range (2000-2020)")
    # Url to dataset
    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/0_Mosaicked/ppp_{year}_1km_Aggregated.tif"
    # Path to directory to store the dataset
    path = f"{data_root}/world_pop"
    # Check if directory exists and create it if not
    if not os.path.isdir(path):
        os.mkdir(path)
    # Download the dataset
    response = wget.download(url, 
                             f"{path}/ppp_{year}_1km_Aggregated.tif")