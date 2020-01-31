# This script takes the clip geometry and zip files with image features and spectral features from GEE
# and puts them into one folder, then moves them to the images directory in the repo data folder

import __init__
from pathlib import Path
import sys
import os
import shutil
sys.path.append('../')
from CPR.configs import data_path

from zipfile import ZipFile

img_list = ['4050_LC08_023036_20130429_1',
            '4050_LC08_023036_20130429_2']


for img in img_list:
    downloads = data_path.parents[1] / 'Downloads'
    dst_file = downloads / img
    try:
        dst_file.mkdir()
    except FileExistsError:
        pass
    src_feat = downloads / '{}'.format(img+' (1)')
    # Extract feature files
    with ZipFile(str(src_feat) + '.zip', 'r') as src:
        try:
            src_feat.mkdir()
        except FileExistsError:
            pass
        src.extractall(str(src_feat))
    # Add extracted files to other zipped feat directory
    with ZipFile(str(dst_file) + '.zip', 'a') as dst:
        extracted_files = [x for x in src_feat.glob('**/*') if x.is_file()]
        for file in extracted_files:
            dst.write(file, os.path.basename(file))

    # Same thing for spectral files
    dst_file_spec = downloads / '{}'.format('spectra_' + img)
    src_spec = downloads / '{}'.format('spectra_' + img + ' (1)')
    # Extract feature files
    with ZipFile(str(src_spec) + '.zip', 'r') as src:
        try:
            src_spec.mkdir()
        except FileExistsError:
            pass
        src.extractall(str(src_spec))
    # Add extracted files to other zipped feat directory
    with ZipFile(str(dst_file_spec) + '.zip', 'a') as dst:
        extracted_files = [x for x in src_spec.glob('**/*') if x.is_file()]
        for file in extracted_files:
            dst.write(file, os.path.basename(file))

    # Move zip folders to image folder
    geom_file = downloads / '{}'.format('clip_geometry_' + img + '.csv')
    shutil.move(str(dst_file) + '.zip', dst_file)
    shutil.move(str(dst_file_spec) + '.zip', dst_file)
    shutil.move(str(geom_file), dst_file)
    shutil.move(str(dst_file), data_path / 'images')

    # Delete unneeded folders
    for file in [src_feat, src_spec, str(src_feat) + '.zip', str(src_spec) + '.zip']:
        shutil.rmtree(str(file))