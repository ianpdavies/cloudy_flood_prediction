import geopandas
from geopandas import GeoDataFrame as gdf
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
import gdal

# ======================================================================================================================
# Method 1: using rain gauge point data and interpolating into a raster

stations_file = 'C:/Users/ipdavies/Downloads/2175903.csv'
df = pd.read_csv(stations_file)
# Sum hourly measurements
# df = df[(df['Measurement Flag'] != ']') & (df['Measurement Flag'] != '[')]
# df = df.groupby(['STATION', 'STATION_NAME', 'ELEVATION', 'LATITUDE', 'LONGITUDE']).sum()
df = df.groupby(['STATION', 'NAME', 'ELEVATION', 'LATITUDE', 'LONGITUDE']).sum().reset_index()
stations = gdf(df, geometry=geopandas.points_from_xy(df.LONGITUDE, df.LATITUDE))
stations.crs = 'EPSG:4269'
stations = stations.to_crs('EPSG:4326')

# tif_file = 'C:/Users/ipdavies/CPR/data/images/4115_LC08_021033_20131227_test/4115_LC08_021033_20131227_test.aspect.tif'
tif_file = 'C:/Users/ipdavies/CPR/data/images/4115_LC08_021033_20131227_1/4115_LC08_021033_20131227_1.aspect.tif'
with rasterio.open(tif_file, 'r', crs='EPSG:4326') as ds:
    print(ds.crs)
    print(ds.transform)
    img_bounds = ds.bounds
    img = ds.read()
    rasterio.plot.plotting_extent(ds)
    fig, ax = plt.subplots(figsize=(8, 8))
    rasterio.plot.show(ds, ax=ax, with_bounds=True)
    stations.plot(ax=ax, facecolor='none', edgecolor='red')

# Want only nearest points to image otherwise will hit memory caps
left = img_bounds[0]
bottom = img_bounds[1]
right = img_bounds[2]
top = img_bounds[3]

img_extent = [[left, bottom],
              [right, bottom],
              [right, top],
              [left, top]]
img_extent = Polygon(img_extent)
x, y = img_extent.exterior.xy
plt.plot(x, y)

buffer_size = 0.1  # degrees
leftb = left - buffer_size
bottomb = bottom - buffer_size
rightb = right + buffer_size
topb = top + buffer_size

buffer_extent = [[leftb, bottomb],
                 [rightb, bottomb],
                 [rightb, topb],
                 [leftb, topb]]
buffer_extent = Polygon(buffer_extent)
xb, yb = buffer_extent.exterior.xy
plt.plot(xb, yb)

buffer_stations = stations[stations.within(buffer_extent)]
buffer_stations.plot(ax=ax, facecolor='blue', edgecolor='blue')
buffer_stations[buffer_stations.PRCP > 0].plot(ax=ax, facecolor='yellow', edgecolor='yellow')

# Rasterize buffer polygon
out_file = 'C:/Users/ipdavies/CPR/data/images/4115_LC08_021033_20131227_1/buffer1.tif'
output_options = gdal.WarpOptions(outputBounds=[leftb, topb, rightb, bottomb], multithread=True)
gdal.Warp(out_file, tif_file, options=output_options)


with rasterio.open(out_file, 'r', crs='EPSG:4326') as ds:
    rasterio.plot.show(ds, ax=ax)
    img_buffer = ds.read().copy()
    img_buffer[:] = np.nan
    meta = ds.meta.copy()
    meta.update(compress='lzw')

# Burn station points into buffer raster
from rasterio import features

with rasterio.open(out_file, 'w+', **meta) as out:
    out_arr = out.read(1)
    out_arr[:] = np.nan
    shapes = ((geom, value) for geom, value in zip(buffer_stations.geometry, buffer_stations.PRCP))
    burned = features.rasterize(shapes=shapes, fill=np.nan, out=out_arr, transform=out.transform)
    out.write_band(1, burned)

with rasterio.open(out_file, 'r', crs='EPSG:4326') as ds:
    rasterio.plot.plotting_extent(ds)
    fig, ax = plt.subplots(figsize=(8, 8))
    rasterio.plot.show(ds, ax=ax, with_bounds=True)
    burned_img = ds.read(1)

# Interpolate precipitation
import pykrige.kriging_tools as kt
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging

# from mpl_toolkits.basemap import Basemap

# Need data in form np.array([[x, y, value], ... ])
values = burned_img[~np.isnan(burned_img)]
indices = np.argwhere(~np.isnan(burned_img))
# lons = list(indices[0])
# lats = list(indices[1])
# values = list(values)
# grid_lon = np.arange(0.0, burned_img.shape[0], 1.0)
# grid_lat = np.arange(0.0, burned_img.shape[1], 1.0)
grid_lon = np.linspace(leftb, rightb, burned_img.shape[0])
grid_lat = np.linspace(bottomb, topb, burned_img.shape[1])
lons = list(grid_lon[indices[:, 0]])
lats = list(grid_lat[indices[:, 1]])
values = list(values)

ok = OrdinaryKriging(lons, lats, values, variogram_model='gaussian', nlags=35)
z, ss = ok.execute('grid', grid_lon, grid_lat, backend='loop')

xintrp, yintrp = np.meshgrid(grid_lon, grid_lat)
fig, ax = plt.subplots(figsize=(10, 10))
# m = Basemap(llcrnrlon=lons.min() - 0.1, llcrnrlat=lats.min() - 0.1, urcrnrlon=lons.max() + 0.1,
#             urcrnrlat=lats.max() + 0.1, projection='merc', resolution='h', area_thresh=1000., ax=ax)
cs = ax.contourf(grid_lon, grid_lat, z, np.linspace(0, np.max(values), 10), extend='both', cmap='jet')


# ======================================================================================================================
# Method 2: Using gridded rainfall data and upsampling to 30m resolution
