import __init__
import numpy as np
import rasterio
import rasterio.plot
from pathlib import Path
from rasterio import features
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import os
import sys
import pandas as pd

sys.path.append('../')
from CPR.configs import data_path

# os.environ['PROJ_LIB'] = 'C:/Users/ipdavies/.conda/pkgs/proj-6.2.1-h9f7ef89_0/Library/share/proj'
# ======================================================================================================================
# Define variables
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test'}
img_list = [x for x in img_list if x not in removed]

soil_path = data_path / 'soil'

# ======================================================================================================================
# Find states that satellite image intersects, fetch soil data for those states, create soil raster image with
# satellite image dimensions

for i, img in enumerate(img_list):
    print('Image {}/{}, ({})'.format(i + 1, len(img_list), img))
    # Grabbing a random tif file from zipped files. Can use stack.tif instead but will need to amend the code below
    tif_file = 'zip://' + str(data_path / 'images' / img / img) + '.zip!' + img + '_aspect.tif'
    with rasterio.open(tif_file, 'r', crs='EPSG:4326') as ds:
        img_bounds = ds.bounds

    left, bottom, right, top = img_bounds[0], img_bounds[1], img_bounds[2], img_bounds[3]
    img_extent = [[left, bottom], [right, bottom], [right, top], [left, top]]
    img_extent = Polygon(img_extent)

    county_shp_path = data_path / 'vector' / 'tl_2019_us_county' / 'tl_2019_us_county.shp'
    counties = gpd.read_file(county_shp_path)
    counties['COUNTYFP'] = counties.STATEFP + counties.COUNTYFP

    states_select = counties[counties.intersects(img_extent)].STATE_ABB.to_list()
    states_select = np.unique(states_select)

    del counties

    soil_maps = []
    for state in states_select:
        file_name = 'gSSURGO_{}.gdb'.format(state)
        mupolygon = gpd.read_file(soil_path / file_name, layer='MUPOLYGON', crs='ESRI:102039')
        muaggatt = gpd.read_file(soil_path / file_name, layer='muaggatt')
        muaggatt = pd.DataFrame(muaggatt)
        muaggatt.drop(muaggatt.columns.difference(['mukey', 'hydgrpdcd']), 1, inplace=True)
        # Map alpha hydro group codes to numeric
        alpha_map = {'A': 1, 'A/D': 2, 'B': 3, 'B/D': 4, 'C': 5, 'C/D': 6, 'D': 7, '0': 0}
        muaggatt['hydgrpdcd'] = muaggatt['hydgrpdcd'].fillna('0')
        muaggatt = muaggatt.replace({'hydgrpdcd': alpha_map})
        muaggatt.rename(columns={'mukey': 'MUKEY'}, inplace=True)
        soil_map = mupolygon.merge(muaggatt, on='MUKEY')
        soil_map.crs = 'ESRI:102039'
        soil_map = soil_map.to_crs('EPSG:4326')
        soil_maps.append(soil_map)
    soil_merge = soil_maps[0]
    if len(soil_maps) > 1:
        soil_merge.append(soil_maps[1:])
    soil_merge['MUKEY'] = soil_merge['MUKEY'].astype('int64')
    soil_merge['hydgrpdcd'] = soil_merge['hydgrpdcd'].astype('int64')
    del soil_maps, soil_map

    # Burn soil map vector into raster

    try:
        Path(soil_path / 'rasters').mkdir(parents=True)
    except FileExistsError:
        pass
    out_file = soil_path / 'rasters' / '{}'.format(img + '_soil.tif')
    with rasterio.open(tif_file, 'r') as src:
        in_arr = src.read(1)
        in_arr[:] = np.nan
        meta = src.meta.copy()
        meta.update(compress='lzw')
        with rasterio.open(out_file, 'w+', **meta) as out:
            shapes = ((geom, value) for geom, value in zip(soil_merge.geometry, soil_merge.hydgrpdcd))
            del soil_merge
            burned = features.rasterize(shapes=shapes, fill=np.nan, out=in_arr, transform=out.transform)
            out.write_band(1, burned)
            del shapes

# # Examine images - requires clicking image to close window and continue with script
# # Closing image window instead of clicking image will cause program to crash.
# for i, img in enumerate(img_list):
#     print('Image {}/{}, ({})'.format(i + 1, len(img_list), img))
#     with rasterio.open(out_file, 'r', crs='EPSG:4326') as ds:
#         rasterio.plot.plotting_extent(ds)
#         fig, ax = plt.subplots(figsize=(8, 8))
#         rasterio.plot.show(ds, ax=ax, with_bounds=True)
#         plt.waitforbuttonpress()
#         plt.close()
