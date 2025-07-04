import glob
import os
import h5py
from tqdm import tqdm
import csv

from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.coordinates import SkyCoord
import astropy.units as u

def load_from_radius(vot_table_path, target_ra_deg, target_dec_deg, radius_deg, exp_base_dir, contain_pattern=''):
    print(f'Loading exposures from VOTable: {vot_table_path}')
    # Ensure VOTable file exists
    if not os.path.exists(vot_table_path):
        raise FileNotFoundError(f'VOTable file not found: {vot_table_path}')
    table = parse_single_table(vot_table_path)
    data = table.array
    target_coord = SkyCoord(target_ra_deg, target_dec_deg, unit='deg')
    
    exposure_list = []
    for row in tqdm(data, desc='Filtering exposures by radius'):
        # Assuming RA/Dec are in specific columns, adjust if necessary
        # Also ensure row[1] (filename part) and row[3], row[4] (coords) exist
        if len(row) > 4 and isinstance(row[3], (float, int)) and isinstance(row[4], (float, int)) and isinstance(row[1], str):
            exp_coord = SkyCoord(row[3], row[4], unit='deg') 
            separation = target_coord.separation(exp_coord).value
            if contain_pattern in row[1] and separation < radius_deg:
                exposure_list.append(os.path.join(exp_base_dir, row[1]))
        else:
            print(f'Skipping row due to missing data or incorrect type: {row}')

    return exposure_list

def load_from_csv(csv_path):
    print(f'Loading exposures from CSV: {csv_path}')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV file not found: {csv_path}')
    exposure_list = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].strip(): # Ensure row is not empty and first element is not empty
                exposure_list.append(row[0].strip())
    return exposure_list

def load_from_directory(exp_dir, contain_pattern=''):
    print(f'Loading exposures from directory: {exp_dir} with pattern {contain_pattern}')
    if not os.path.isdir(exp_dir):
        raise NotADirectoryError(f'Exposure directory not found: {exp_dir}')
    exposure_list = glob.glob(os.path.join(exp_dir, f'*{contain_pattern}*')) 
    return exposure_list