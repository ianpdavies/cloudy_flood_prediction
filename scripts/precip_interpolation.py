import __init__
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import rasterio.plot
from rasterio import features
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
from CPR.configs import data_path

# ======================================================================================================================
# Burn vector of kriging contours into a raster

img_list = os.listdir(str(data_path / 'images'))
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]
raster_path = data_path / 'precip' / 'kriged_rasters'
vector_path = data_path / 'precip' / 'kriged_vectors'

for i, img in enumerate(img_list):
    print('Image {}/{}, ({})'.format(i+1, len(img_list), img))
    contour_vector = gpd.read_file(vector_path / img / '{}'.format(img + '_krig_vector.shp'))
    contour_vector = contour_vector[~contour_vector.isna().geometry]  # Remove any empty geometry artifacts

    out_file = raster_path / '{}'.format(img + '_precip.tif')
    tif_file = 'zip://' + str(data_path / 'images' / img / img) + '.zip!' + img + '.aspect.tif'
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
    with rasterio.open(str(stack_path), 'r') as src:
        in_arr = src.read(1).astype('float32')
        in_arr[:] = np.nan
        meta = src.meta.copy()
        meta = src.meta
        meta['dtype'] = 'float32'
        meta.update(compress='lzw')
        with rasterio.open(out_file, 'w+', **meta) as out:
            shapes = ((geom, value) for geom, value in zip(contour_vector.geometry, contour_vector.Value_Max))
            burned = features.rasterize(shapes=shapes, fill=np.nan, out=in_arr, transform=out.transform)
            out.write_band(1, burned)

# Examine images
for i, img in enumerate(img_list):
    print('Image {}/{}, ({})'.format(i + 1, len(img_list), img))
    out_file = raster_path / '{}'.format(img + '_precip.tif')
    with rasterio.open(out_file, 'r', crs='EPSG:4326') as ds:
        rasterio.plot.plotting_extent(ds)
        fig, ax = plt.subplots(figsize=(8, 8))
        rasterio.plot.show(ds, ax=ax, with_bounds=True)
        plt.waitforbuttonpress()
        plt.close()

