import __init__
import geopandas
from geopandas import GeoDataFrame as gdf
import descartes
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
import gdal
import sys

sys.path.append('../')
from CPR.configs import data_path
from CPR.keys import noaa_key

# ======================================================================================================================
img = '4115_LC08_021033_20131227_1'
precip_path = data_path / 'precip' / img
try:
    precip_path.mkdir(parents=True)
except FileExistsError:
    pass
stations = pd.read_csv(precip_path / '{}'.format(img + '_precip.csv', index=False))
stations = gdf(stations, geometry=geopandas.points_from_xy(stations.long, stations.lat))

buffer = data_path / 'images' / img / '{}'.format(img + '_buffer.tif')

# Sum daily precip measurements
stations = pd.read_csv(precip_path / '{}'.format(img + '_precip.csv', index=False))
stations = stations.groupby(['station_id', 'name', 'elevation', 'lat', 'long']).sum().reset_index()
stations = gdf(stations, geometry=geopandas.points_from_xy(stations.long, stations.lat))
stations.crs = 'EPSG:4269'
stations = stations.to_crs('EPSG:4326')
stations.to_file(str(data_path / 'precip' / '{}'.format(img + 'precip.shp')))

# Plot original image extent, buffer extent, and stations
tif_file = 'zip://' + str(data_path / 'images' / img / img) + '.zip!' + img + '.aspect.tif'
with rasterio.open(tif_file, 'r', crs='EPSG:4326') as ds:
    img_bounds = ds.bounds
    rasterio.plot.plotting_extent(ds)
    fig, ax = plt.subplots(figsize=(8, 8))
    rasterio.plot.show(ds, ax=ax, with_bounds=True)
    stations.plot(ax=ax, facecolor='none', edgecolor='red')

buffer_size = 0.1  # degrees
left, bottom, right, top = img_bounds[0], img_bounds[1], img_bounds[2], img_bounds[3]
img_extent = [[left, bottom], [right, bottom], [right, top], [left, top]]
img_extent = Polygon(img_extent)
x, y = img_extent.exterior.xy
plt.plot(x, y)
leftb, bottomb, rightb, topb = left - buffer_size, bottom - buffer_size, right + buffer_size, top + buffer_size
buffer_extent = [[leftb, bottomb], [rightb, bottomb], [rightb, topb], [leftb, topb]]
buffer_extent = Polygon(buffer_extent)
xb, yb = buffer_extent.exterior.xy
plt.plot(xb, yb)

# ======================================================================================================================
buffer_stations = stations[stations.within(buffer_extent)]
buffer_stations.plot(ax=ax, facecolor='red', edgecolor='red')
buffer_stations[buffer_stations.prcp > 0].plot(ax=ax, facecolor='yellow', edgecolor='yellow')

# Rasterize buffer polygon
out_file = str(data_path / 'images' / img / '{}'.format(img + '_buffer.tif'))
output_options = gdal.WarpOptions(outputBounds=[leftb, topb, rightb, bottomb], multithread=True)
tif_file_gdal = str(data_path / 'images' / img / img) + '.zip/' + img + '.aspect.tif'

# this returns SystemError: <built-in function wrapper_GDALWarpDestName> returned NULL without setting an error
# Cant access zipped rasters in gdal?
tif_file_gdal = 'C:/Users/ipdavies/CPR/data/images/4115_LC08_021033_20131227_1/4115_LC08_021033_20131227_1.zip/4115_LC08_021033_20131227_1.aspect.tif'
tif_file_gdal = 'C:/Users/ipdavies/CPR/data/images/4115_LC08_021033_20131227_1/4115_LC08_021033_20131227_1.aspect.tif'
gdal.Warp(out_file, tif_file_gdal, options=output_options, overwrite=True)

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
    shapes = ((geom, value) for geom, value in zip(buffer_stations.geometry, buffer_stations.prcp))
    burned = features.rasterize(shapes=shapes, fill=np.nan, out=out_arr, transform=out.transform)
    out.write_band(1, burned)

with rasterio.open(out_file, 'r', crs='EPSG:4326') as ds:
    # rasterio.plot.plotting_extent(ds)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # rasterio.plot.show(ds, ax=ax, with_bounds=True)
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

# ok = OrdinaryKriging(lons, lats, values, variogram_model='spherical', nlags=35)
uk = UniversalKriging(lons, lats, values, variogram_model='spherical', nlags=12, verbose=True, enable_plotting=True)
z, ss = uk.execute('grid', grid_lon, grid_lat)
#
# xintrp, yintrp = np.meshgrid(grid_lon, grid_lat)
# fig, ax = plt.subplots(figsize=(10, 10))
# # m = Basemap(llcrnrlon=lons.min() - 0.1, llcrnrlat=lats.min() - 0.1, urcrnrlon=lons.max() + 0.1,
# #             urcrnrlat=lats.max() + 0.1, projection='merc', resolution='h', area_thresh=1000., ax=ax)
# cs = ax.contourf(grid_lon, grid_lat, z, np.linspace(0, np.max(values), 10), extend='both', cmap='jet')

# ======================================================================================================================
from scipy import interpolate

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
grid_x, grid_y = np.mgrid[np.min(grid_lon):np.max(grid_lon):5451j, np.min(grid_lat):np.max(grid_lat):4242j]
z = interpolate.griddata(np.column_stack([lons, lats]), values, (grid_x, grid_y), method='cubic')
z = interpolate.interp2d(lons, lats, values, kind='cubic')
z = z(grid_lon, grid_lat)
fig, ax = plt.subplots(figsize=(10, 10))
cs = ax.contourf(grid_lon, grid_lat, z, np.linspace(0, np.max(values), 10), extend='both', cmap='jet')
buffer_stations.plot(ax=ax, facecolor='red', edgecolor='red')
buffer_stations[buffer_stations.prcp > 0].plot(ax=ax, facecolor='yellow', edgecolor='yellow')


def barnes_objective(xs, ys, zs, XI, YI, XR, YR, RUNS=3):
    # -- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    # -- size of new matrix
    if (np.ndim(XI) == 1):
        nx = len(XI)
    else:
        nx, ny = np.shape(XI)

    # -- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')

    # -- square of Barnes smoothing lengths scale
    xr2 = XR ** 2
    yr2 = YR ** 2
    # -- allocate for output zp array
    zp = np.zeros_like(XI.flatten())
    # -- first analysis
    for i, XY in enumerate(zip(XI.flatten(), YI.flatten())):
        dx = np.abs(xs - XY[0])
        dy = np.abs(ys - XY[1])
        # -- calculate weights
        w = np.exp(-dx ** 2 / xr2 - dy ** 2 / yr2)
        zp[i] = np.sum(zs * w) / sum(w)

    # -- allocate for even and odd zp arrays if iterating
    if (RUNS > 0):
        zpEven = np.zeros_like(zs)
        zpOdd = np.zeros_like(zs)
    # -- for each run
    for n in range(RUNS):
        # -- calculate even and odd zp arrays
        for j, xy in enumerate(zip(xs, ys)):
            dx = np.abs(xs - xy[0])
            dy = np.abs(ys - xy[1])
            # -- calculate weights
            w = np.exp(-dx ** 2 / xr2 - dy ** 2 / yr2)
            if ((n % 2) == 0):  # -- even (% = modulus)
                zpEven[j] = zpOdd[j] + np.sum((zs - zpOdd) * w) / np.sum(w)
            else:  # -- odd
                zpOdd[j] = zpEven[j] + np.sum((zs - zpEven) * w) / np.sum(w)
        # -- calculate zp for run n
        for i, XY in enumerate(zip(XI.flatten(), YI.flatten())):
            dx = np.abs(xs - XY[0])
            dy = np.abs(ys - XY[1])
            w = np.exp(-dx ** 2 / xr2 - dy ** 2 / yr2)
            if ((n % 2) == 0):  # -- even (% = modulus)
                zp[i] = zp[i] + np.sum((zs - zpEven) * w) / np.sum(w)
            else:  # -- odd
                zp[i] = zp[i] + np.sum((zs - zpOdd) * w) / np.sum(w)

    # -- reshape to original dimensions
    if (np.ndim(XI) != 1):
        ZI = zp.reshape(nx, ny)
    else:
        ZI = zp.copy()

    # -- return output matrix/array
    return ZI

new_z = barnes_objective(lons, lats, values, grid_lon, grid_lat)



# ======================================================================================================================
# Detrend z = x + y (fit precip using lat/long as predictors), 1st or 2nd order polynomial

import scipy.signal as signal

signal.detrend()

randgen = np.random.RandomState(9)
npoints = 1000
noise = randgen.randn(npoints)
x = 3 + 2 * np.linspace(0, 1, npoints) + noise
signal.detrend(x) - noise

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# create data
x = np.linspace(0, 2 * np.pi, 500)
y = np.random.normal(0.3 * x, np.random.rand(len(x)))
plt.plot(x, y)

not_nan_ind = ~np.isnan(y)
m, b, r_val, p_val, std_err = stats.linregress(x, y)
detrend_y = y - (m * x + b)
plt.plot(x, detrend_y)

# ===================

myData = np.column_stack([stations.prcp, stations.lat, stations.long])
x = myData[:, 1:]
y = myData[:, 0]
x = np.c_[x, np.ones(x.shape[0])]
np.linalg.lstsq(x, y)
m, b1, b2, r_val, p_val, std_err = stats.linregress(x, y)

from sklearn import linear_model

clf = linear_model.LinearRegression(fit_intercept=False)
clf.fit(x, y)
b0, b1 = clf.coef_
y_detrended = y - ((x[:, 0] * b0) + x[:, 1] * b1)
y_detrended

np.polyfit(x)

randgen = np.random.RandomState(9)
npoints = 100 * 100
noise = randgen.randn(npoints).reshape([100, 100])
z_trend = np.mgrid[0:100:100j, 0:100:100j][0] * np.mgrid[0:100:100j, 0:100:100j][1]
z = z_trend * noise
plt.imshow(z_trend)
plt.imshow(z)

gridx, gridy = np.mgrid[0:100:100j, 0:100:100j][0], np.mgrid[0:100:100j, 0:100:100j][1]
samplex = np.random.choice(gridx.shape[0], 100, replace=False)
sampley = np.random.choice(gridx.shape[1], 100, replace=False)
sample_points = np.zeros(shape=[100 * 100, 3])
sample_points = np.column_stack([z.reshape([100 * 100]), gridx.reshape([100 * 100, ]), gridy.reshape([100 * 100, ])])

clf = linear_model.LinearRegression(fit_intercept=False)
clf.fit(sample_points[:, :2], sample_points[:, 2])
b0, b1 = clf.coef_
y_detrended = sample_points[:, 2] - ((sample_points[:, 0] * b0) + sample_points[:, 1] * b1)
y_trend = (sample_points[:, 0] * b0) + (sample_points[:, 1] * b1)
plt.imshow(y_detrended.reshape([100, 100]))

with rasterio.open(str(data_path / 'images' / img / '{}'.format(img + '_buffer.tif')), 'r') as ds:
    print(ds.res)
    print(ds.bounds)
    print(ds.shape)
    img = ds.read()
    rasterio.plot.plotting_extent(ds)
    fig, ax = plt.subplots(figsize=(8, 8))
    rasterio.plot.show(ds, ax=ax, with_bounds=True)
    stations.plot(ax=ax, facecolor='none', edgecolor='red')
    rasterio.plot.show(ds)

with rasterio.open(str(data_path / 'precip' / 'kriged_rasters' / 'precip_krig2' / 'hdr.adf'), 'r') as ds:
    print(ds.res)
    print(ds.shape)
    grid_lon = np.linspace(ds.bounds[0], ds.bounds[2], ds.shape[1])
    grid_lat = np.linspace(ds.bounds[1], ds.bounds[3], ds.shape[0])
    ds_img = ds.read(1)
    # rasterio.plot.show(ds)
    fig, ax = plt.subplots(figsize=(10, 10))
    cs = ax.contourf(grid_lon, grid_lat, ds_img, [1.96, 2.84, 3.24, 3.42, 3.5, 3.75, 4.16, 5.04, 7], extend='both',
                     cmap='jet')


# ======================================================================================================================
# Burn vector of kriging contours into a raster
import geopandas as gpd
from rasterio import features

contour_vector = gpd.read_file(data_path / 'precip' / 'krig_vector.shp')
contour_vector = gpd.read_file(data_path / 'precip' / 'kriged_vectors' / '4115_LC08_021033_20131227_1krig_vector.shp')
contour_vector = contour_vector[~contour_vector.isna().geometry]


out_file = data_path / 'precip' / 'krig_raster.tif'
tif_file = 'zip://' + str(data_path / 'images' / img / img) + '.zip!' + img + '.aspect.tif'

with rasterio.open(tif_file, 'r') as src:
    in_arr = src.read(1)
    in_arr[:] = np.nan
    meta = src.meta.copy()
    meta.update(compress='lzw')
    with rasterio.open(out_file, 'w+', **meta) as out:
        shapes = ((geom, value) for geom, value in zip(contour_vector.geometry, contour_vector.Value_Max))
        burned = features.rasterize(shapes=shapes, fill=np.nan, out=in_arr, transform=out.transform)
        out.write_band(1, burned)

with rasterio.open(out_file, 'r', crs='EPSG:4326') as ds:
    # rasterio.plot.plotting_extent(ds)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # rasterio.plot.show(ds, ax=ax, with_bounds=True)
    burned_img = ds.read(1)
    plt.imshow(burned_img)