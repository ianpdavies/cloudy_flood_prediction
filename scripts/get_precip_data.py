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
import os
import datetime
import requests
import json
import sys
sys.path.append('../')
from CPR.configs import data_path
from CPR.keys import noaa_key

# ======================================================================================================================

token = noaa_key
county_shp_path = data_path / 'vector' / 'tl_2019_us_county' / 'tl_2019_us_county.shp'
dfo_path = data_path / 'vector' / 'dfo_floods' / 'dfo_floods.shp'
ghcnd_station_inventory = data_path / 'ghcnd_stations.csv'

img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

for img in img_list:
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
    # Grabbing a random tif file from zipped files. Can use stack.tif instead but will need to amend the code below
    tif_file = 'zip://' + str(data_path / 'images' / img / img) + '.zip!' + img + '.aspect.tif'
    # with rasterio.open(str(stack_path), 'r', crs='EPSG:4326') as ds:
    with rasterio.open(tif_file, 'r', crs='EPSG:4326') as ds:
        print(ds.crs)
        print(ds.transform)
        img_bounds = ds.bounds
        rasterio.plot.plotting_extent(ds)
        fig, ax = plt.subplots(figsize=(8, 8))
        rasterio.plot.show(ds.read(1), ax=ax, transform=ds.transform, with_bounds=True)
        # stations.plot(ax=ax, facecolor='none', edgecolor='red')

    # Can't intepolate all points without hitting memory cap, so buffer raster and interpolate only the points within
    left, bottom, right, top = img_bounds[0], img_bounds[1], img_bounds[2], img_bounds[3]
    img_extent = [[left, bottom], [right, bottom], [right, top], [left, top]]
    img_extent = Polygon(img_extent)
    x, y = img_extent.exterior.xy
    plt.plot(x, y)

    buffer_size = 0.1  # degrees
    leftb = left - buffer_size
    bottomb = bottom - buffer_size
    rightb = right + buffer_size
    topb = top + buffer_size
    buffer_extent = [[leftb, bottomb], [rightb, bottomb], [rightb, topb], [leftb, topb]]
    buffer_extent = Polygon(buffer_extent)
    xb, yb = buffer_extent.exterior.xy
    plt.plot(xb, yb)

    # Find US counties that intersect flood event
    counties = geopandas.read_file(county_shp_path)
    counties['COUNTYFP'] = counties.STATEFP + counties.COUNTYFP
    counties_select = counties[counties.intersects(buffer_extent)].COUNTYFP.to_list()
    counties_select = ['FIPS:'+county for county in counties_select]

    dfo_id = int(img.split('_')[0])
    dfo = geopandas.read_file(dfo_path)

    observation_period = -2
    flood = dfo[dfo.ID == dfo_id]
    start_date = flood.Began.iloc[0]
    end_date = img.split('_')[3]
    end_date = end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:]
    new_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    new_start_date += datetime.timedelta(days=observation_period)
    new_start_date = new_start_date.strftime('%Y-%m-%d')

    # Initialize df for NOAA data
    df_prcp = pd.DataFrame()
    dates_prcp = []
    prcp = []
    station = []
    attributes = []

    # Fetch NOAA data
    for county in counties_select:
        req = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&datatypeid=PRCP&limit=1000&locationid=' + \
              county + '&startdate=' + new_start_date + '&enddate=' + end_date
        d = json.loads(requests.get(req, headers={'token': token}).text)
        if len(d) == 0:
            continue
        items = [item for item in d['results'] if item['datatype'] == 'PRCP']
        dates_prcp += [item['date'] for item in items]
        prcp += [item['value'] for item in items]
        station += [item['station'].split(':')[1] for item in items]
        attributes += [item['attributes'] for item in items]

    df_prcp['date'] = [datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S") for d in dates_prcp]
    df_prcp['prcp'] = [p for p in prcp]
    df_prcp['attributes'] = [p for p in attributes]
    df_prcp['station_id'] = [p for p in station]

    # Get lat/long of stations
    station_list = pd.read_csv(ghcnd_station_inventory)
    stations = df_prcp.merge(station_list, on='station_id', how='left')

    # Save precip data
    precip_path = data_path / 'precip' / img
    try:
        precip_path.mkdir(parents=True)
    except FileExistsError:
        pass
    stations.to_csv(precip_path / '{}'.format(img + '_precip.csv', index=False))



