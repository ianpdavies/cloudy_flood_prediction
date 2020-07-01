import __init__
import numpy as np
import rasterio
import rasterio.plot
from rasterio import features
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import os
import sys
sys.path.append('../')
from CPR.configs import data_path

# ======================================================================================================================
# Define variables
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]


soil_path = data_path / 'soil'

# ======================================================================================================================
# Find states that satellite image intersects, fetch soil data for those states, create soil raster image with
# satellite image dimensions

for i, img in enumerate(img_list):
    print('Image {}/{}, ({})'.format(i+1, len(img_list), img))
# Grabbing a random tif file from zipped files. Can use stack.tif instead but will need to amend the code below
    tif_file = 'zip://' + str(data_path / 'images' / img / img) + '.zip!' + img + '.aspect.tif'
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

    # Get soil maps and merge together
    soil_maps = []
    for state in states_select:
        file_name = 'gSSURGO_{}.gdb'.format(state)
        soil_map = gpd.read_file(soil_path / file_name, layer='MUPOLYGON', crs='ESRI:102039')
        soil_map.crs = 'ESRI:102039'
        soil_map = soil_map.to_crs('EPSG:4326')
        soil_maps.append(soil_map)
    soil_merge = soil_maps[0]
    if len(soil_maps) > 1:
        soil_merge.append(soil_maps[1:])
    soil_merge['MUSYM'] = soil_merge['MUSYM'].astype('int64')
    del soil_maps, soil_map

    # Burn soil map vector into raster
    out_file = soil_path / 'rasters' / '{}'.format(img + '_soil.tif')
    with rasterio.open(tif_file, 'r') as src:
        in_arr = src.read(1)
        in_arr[:] = np.nan
        meta = src.meta.copy()
        meta.update(compress='lzw')
        with rasterio.open(out_file, 'w+', **meta) as out:
            shapes = ((geom, value) for geom, value in zip(soil_merge.geometry, soil_merge.MUSYM))
            del soil_merge
            burned = features.rasterize(shapes=shapes, fill=np.nan, out=in_arr, transform=out.transform)
            out.write_band(1, burned)

# Examine images
for i, img in enumerate(img_list):
    print('Image {}/{}, ({})'.format(i + 1, len(img_list), img))
    with rasterio.open(out_file, 'r', crs='EPSG:4326') as ds:
        rasterio.plot.plotting_extent(ds)
        fig, ax = plt.subplots(figsize=(8, 8))
        rasterio.plot.show(ds, ax=ax, with_bounds=True)
        plt.waitforbuttonpress()
        plt.close()
