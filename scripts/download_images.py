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

img_list = os.listdir(data_path / 'images')
# removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
#            '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1'}
removed = {'new'}
img_list = [x for x in img_list if x not in removed]

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

# ==============================================================================================
# Have to download images separately from drive with latest GEE update.
# This puts them into zip folders, then runs the script below

new_features = ['landcover', 'spi', 'sti', 'twi']

for img in img_list:
    print(img)
    downloads = data_path / 'images' / 'new'
    image_dir = Path('D:/Workspace/ipdavies/CPR/data/images')
    zip_dir = str(downloads / '{}'.format(img + '.zip'))
    with ZipFile(zip_dir, 'w') as dst:
        files = [Path(downloads / '{}'.format(img + '_' + feat + '.tif')) for feat in new_features]
        for file in files:
            dst.write(file, os.path.basename(file))

# ==============================================================================================
# Replacing files in existing zip folders with newly downloaded files (in zip folders)

remove_these = ['sti', 'spi', 'twi', 'developed', 'forest', 'other_landcover', 'planted', 'wetlands']
for img in img_list:
    print(img)
    downloads = data_path / 'images' / 'new'
    image_dir1 = Path('D:/Workspace/ipdavies/CPR/data/images')
    image_dir1 = image_dir1 / img
    image_dst = image_dir1 / img
    try:
        image_dst.mkdir()
    except FileExistsError:
        pass
    zip_dir1 = str(image_dir1 / '{}'.format(img + '.zip'))
    with ZipFile(zip_dir1, 'r') as dst:
        dst.extractall(str(image_dst))
    for file in remove_these:
        try:
            os.remove(str(image_dst / '{}'.format(img + '_' + file + '.tif')))
        except FileNotFoundError:
            pass
        try:
            os.remove(str(image_dst / '{}'.format(img + '_' + file + '.tfw')))
        except FileNotFoundError:
            pass

    # Unzip new features into these folders
    image_dir2 = downloads / img
    zip_dir2 = str(downloads / '{}'.format(img + '.zip'))
    try:
        image_dir2.mkdir(parents=True)
    except FileExistsError:
        pass
    with ZipFile(zip_dir2, 'r') as dst:
        dst.extractall(str(image_dst))

    # Delete zip folder
    os.remove(str(zip_dir1))

    # Now zip up files
    with ZipFile(zip_dir1, 'w') as dst:
        files = [x for x in image_dst.glob('**/*') if x.is_file()]
        for file in files:
            dst.write(file, os.path.basename(file))

    # Remove unzipped folder with files
    shutil.rmtree(str(image_dst))

# ==============================================================================================
# Renaming zipped files with individual .tifs downloaded from GDrive
# Get list of folders
# Get list of contents
# Rename folder

image_dir = data_path / 'images'
zip_list = os.listdir(image_dir)
for zip_name_old in zip_list:
    zip_dir = ZipFile(image_dir / zip_name_old)
    names = ZipFile.namelist(zip_dir)
    name = names[0].split('_')[:-1]
    zip_name_new = '_'.join(name) + '.zip'
    zip_dir.close()
    os.rename(str(image_dir / zip_name_old), str(image_dir / zip_name_new))

# Putting zip files into directories
image_dir = data_path / 'images'
zip_list = os.listdir(image_dir)
for zip_file in zip_list:
    name = zip_file.split('.')[0]
    Path(image_dir / name).mkdir()
    shutil.move(str(image_dir / zip_file), str(image_dir / name))


# ==============================================================================================
# Swapping out a .tif in a zip folder with a new .tif
remove_these = ['curve']
for img in img_list:
    # Unzip zip folder
    print(img)
    image_dir1 = Path('D:/Workspace/ipdavies/CPR/data/images')  # Directory of zip folder containing old .tif
    image_dir1 = data_path / 'images'
    image_dir1 = image_dir1 / img
    image_dst = image_dir1 / img
    image_dir2 = data_path / 'new'  # Directory containing new .tifs
    try:
        image_dst.mkdir()
    except FileExistsError:
        pass
    zip_dir1 = str(image_dir1 / '{}'.format(img + '.zip'))
    with ZipFile(zip_dir1, 'r') as dst:
        dst.extractall(str(image_dst))
    # Remove old .tif
    for file in remove_these:
        try:
            os.remove(str(image_dst / '{}'.format(img + '_' + file + '.tif')))
        except FileNotFoundError:
            pass
        try:
            os.remove(str(image_dst / '{}'.format(img + '_' + file + '.tfw')))
        except FileNotFoundError:
            pass
        # Move new .tif to the unzipped folder that held the old .tif
        new_tif = str(image_dir2 / '{}_{}.tif'.format(img, file))
        shutil.move(new_tif, image_dst)

    # Delete old zip folder
    os.remove(str(zip_dir1))

    # Now zip up files with new .tif
    with ZipFile(zip_dir1, 'w') as dst:
        files = [x for x in image_dst.glob('**/*') if x.is_file()]
        for file in files:
            dst.write(file, os.path.basename(file))

    # Remove unzipped folder with files
    shutil.rmtree(str(image_dst))