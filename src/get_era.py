import os
import argparse
import datetime
import cdsapi
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--data_root",
                    help="Path to directory where data is stored")
parser.add_argument("--variable",
                    help="Variable to download (match the era5-land variable name)")
parser.add_argument("--years", type=lambda x: x.split(','),
                    help="Years to download data for (separated by comma)")
parser.add_argument("--short_name",
                    help="Short name for variable (also dir name in data_root)")
parser.add_argument("--dataset", help="CDS datastore dataset identifier",
                    default="reanalysis-era5-land")

args = parser.parse_args()

c = cdsapi.Client()

def get_era(path_save, #  dataset=args.dataset,
            variable="2m_temperature", year=2022, month=1):
    if os.path.isfile(f"{path_save}/{month}_{year}.grib"):
        return None
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': variable,
            'year': str(year),
            'month': f"{month:02}",
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'format': 'grib',
        },
        f"{path_save}/{month}_{year}.grib")
    
    
# Check if folder exists and create it if not
if not os.path.isdir(args.data_root):
    os.mkdir(args.data_root)
# Add the short name in the path
path_save = f"{args.data_root}/{args.short_name}"
if not os.path.isdir(path_save):
    os.mkdir(path_save)
    
# Get the current date
date = datetime.datetime.now()

for yr in np.arange(int(args.years[0]), int(args.years[1])+1, 1):
    if yr == date.year:
        # current month -1 to account for model lag (around 2 months)
        months = np.arange(1, date.month-1, 1)
    else:
        months = np.arange(1, 13, 1)
    # Loop through the years and months to download the data
    for mth in months:
        print(f"Downloading {args.variable} for {yr}-{mth}")
        get_era(path_save=path_save,
                # dataset=args.dataset,
                variable=args.variable,
                year=yr, month=mth)
    
