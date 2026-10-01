"""Helpers in this module build lists of Euclid exposure file paths for reprojection.

Each returns a ``list`` of path strings: :func:`load_from_radius` keeps the rows of a
VOTable exposure catalogue that lie within a radius of a target, :func:`load_from_csv`
reads the first column of a CSV file, and :func:`load_from_directory` globs one
directory. Such a list is the ``exposure_list`` of
:class:`~selfcal.pipeline.pipeline_wrapper.Reprojector`. The helpers are independent of
:class:`~selfcal.instruments.euclid.adapter.EuclidInstrument` and of the runner, whose
``reproject`` task globs ``[reproject].input_dirs`` with ``file_pattern`` instead.
"""
import glob
import logging
import os
from tqdm import tqdm
import csv

from astropy.io.votable import parse_single_table
from astropy.coordinates import SkyCoord

from ... import _state

logger = logging.getLogger(__name__)

def load_from_radius(vot_table_path, target_ra_deg, target_dec_deg, radius_deg, exp_base_dir, contain_pattern=''):
    """Return the exposures of a VOTable catalogue that lie within a radius of a target.

    The columns of the first table in ``vot_table_path`` are read by position,
    counted from 0: column 1 is the exposure file path relative to ``exp_base_dir``,
    columns 3 and 4 are its RA and Dec in degrees. A row is kept when its angular separation from
    (``target_ra_deg``, ``target_dec_deg``) is less than ``radius_deg`` degrees and
    its path contains the substring ``contain_pattern`` (the default ``''`` matches
    every path). The result holds ``os.path.join(exp_base_dir, path)`` for each kept
    row, in table order; the files are not checked. RA and Dec must be ``float64``
    columns: a row whose RA or Dec has another type (``float32``, integer, masked),
    whose path is not a string, or that has fewer than five columns is skipped with
    a logged warning. Raises ``FileNotFoundError`` when the VOTable does not exist."""
    logger.info(f'Loading exposures from VOTable: {vot_table_path}')
    # Ensure VOTable file exists
    if not os.path.exists(vot_table_path):
        raise FileNotFoundError(f'VOTable file not found: {vot_table_path}')
    table = parse_single_table(vot_table_path)
    data = table.array
    target_coord = SkyCoord(target_ra_deg, target_dec_deg, unit='deg')
    
    exposure_list = []
    for row in tqdm(data, desc='Filtering exposures by radius',
                    disable=not _state.progress_enabled):
        # Expected VOTable row schema: row[1] = exposure file path relative to
        # exp_base_dir, row[3] = RA (deg), row[4] = Dec (deg). The guard below
        # skips rows where any of these is missing or mistyped.
        if len(row) > 4 and isinstance(row[3], (float, int)) and isinstance(row[4], (float, int)) and isinstance(row[1], str):
            exp_coord = SkyCoord(row[3], row[4], unit='deg') 
            separation = target_coord.separation(exp_coord).value
            if contain_pattern in row[1] and separation < radius_deg:
                exposure_list.append(os.path.join(exp_base_dir, row[1]))
        else:
            logger.warning(f'Skipping row due to missing data or incorrect type: {row}')

    return exposure_list

def load_from_csv(csv_path):
    """Return the exposure paths listed in the first column of a CSV file.

    Each row whose first field is not blank contributes that field, stripped of
    surrounding whitespace, in file order; the other columns are ignored. There is
    no header handling: a header line is returned as an entry too. The paths are
    returned as written, neither joined to a directory nor checked. Raises
    ``FileNotFoundError`` when ``csv_path`` does not exist."""
    logger.info(f'Loading exposures from CSV: {csv_path}')
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
    """Return the paths of the entries of ``exp_dir`` whose names contain ``contain_pattern``.

    The result is ``glob.glob(os.path.join(exp_dir, f'*{contain_pattern}*'))``: paths
    prefixed with ``exp_dir``, in file-system order (not sorted), subdirectories
    included and hidden names (starting with ``.``) left out; glob wildcards in
    ``contain_pattern`` keep their meaning. The default ``''`` lists every entry.
    Raises ``NotADirectoryError`` when ``exp_dir`` is not a directory."""
    logger.info(f'Loading exposures from directory: {exp_dir} with pattern {contain_pattern}')
    if not os.path.isdir(exp_dir):
        raise NotADirectoryError(f'Exposure directory not found: {exp_dir}')
    exposure_list = glob.glob(os.path.join(exp_dir, f'*{contain_pattern}*')) 
    return exposure_list