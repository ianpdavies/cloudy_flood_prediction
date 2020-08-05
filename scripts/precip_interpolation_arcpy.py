# For use with Standard ArcGIS with Geostatistical Analyst Extension
# SCRIPT IS WRITTEN IN PYTHON 2 FOR ARCGIS DESKTOP LICENSE - PYTHON 3 IS ONLY COMPATIBLE WITH ARCGIS PRO

import __init__
import arcpy
from arcpy import env
from arcpy.sa import *
import os
import pathlib
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from CPR.configs import data_path

# ======================================================================================================================

# Set environment settings
env.workspace = str(data_path / 'precip')
arcpy.CheckOutExtension("GeoStats")
arcpy.env.overwriteOutput = True

img_list = os.listdir(str(data_path / 'images'))
removed = {'4115_LC08_021033_20131227_test'}
img_list = [x for x in img_list if x not in removed]

for i, img in enumerate(img_list):
    print('Image {}/{}, ({})'.format(i+1, len(img_list), img))
    ga_layer_path = data_path / 'precip' / 'kriged_rasters' / 'arc_layers'
    try:
        ga_layer_path.mkdir(parents=True)
    except Exception:
        pass

    vector_path = data_path / 'precip' / 'kriged_vectors' / img
    try:
        vector_path.mkdir(parents=True)
    except Exception:
        pass

    raster_path = data_path / 'precip' / 'kriged_rasters'
    try:
        raster_path.mkdir(parents=True)
    except Exception:
        pass

    # Get buffer raster information
    img_path = str(data_path / 'images' / img / img) + '_aspect.tif'
    raster = arcpy.Raster(img_path)
    description = arcpy.Describe(raster)
    cellsize1 = description.children[0].meanCellHeight  # Cell size y
    cellsize2 = description.children[0].meanCellWidth  # Cell size x
    print('Cellsize:', cellsize1, cellsize2)

    # Set kriging parameters
    in_features = 'station_data/' + img + '/shp/' + img + '_precip.shp'
    field = "prcp"
    cellsize = cellsize1
    out_ga_layer = 'kriged_rasters/arc_layers/' + img + '_krig_ga.lyr'
    transformation = 'EMPIRICAL'
    semivariogram = 'EXPONENTIAL'
    output_type = 'PREDICTION'

    # Set contour filling parameters
    out_vector = 'kriged_vectors/' + img + '/' + img +'_krig_vector.shp'
    class_type = 'QUANTILE'
    contour_type = 'FILLED_CONTOUR'
    classes_count = 15

    # Set cv parameters
    out_cv_shp = 'kriged_vectors/' + img + '/' + img +'_krig_cv.shp'
    in_cv_dbf = 'kriged_vectors/' + img + '/' + img +'_krig_cv.dbf'
    out_cv_csv_name = img + '_krig_cv.csv'

    # Empirical Bayesian Kriging
    ebayes_krig = arcpy.EmpiricalBayesianKriging_ga(in_features=in_features,
                                                    z_field=field,
                                                    out_ga_layer='ga_layer',
                                                    cell_size=cellsize,
                                                    transformation_type=transformation,
                                                    output_type=output_type,
                                                    semivariogram_model_type=semivariogram)

    arcpy.SaveToLayerFile_management(in_layer=ebayes_krig, out_layer=out_ga_layer, is_relative_path='ABSOLUTE')

    # Cross-validate kriged raster and export prediction vs. actual as table
    cv = arcpy.CrossValidation_ga(in_geostat_layer=out_ga_layer, out_point_feature_class=out_cv_shp)
    arcpy.TableToTable_conversion(in_rows=in_cv_dbf, out_path=str(raster_path), out_name=out_cv_csv_name)
    arcpy.Delete_management(out_cv_shp)

    # Save results of cross-validation as image and csv
    plt.ioff()
    cv_df = pd.read_csv(str(raster_path / '{}'.format(img + '_krig_cv.csv')))
    x, y = cv_df['Measured'], cv_df['Predicted']
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    max_value = np.max([cv_df['Measured'].max(), cv_df['Predicted'].max()])
    lims = [0, max_value]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), '--', color='red')
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(str(raster_path / '{}'.format(img + '_krig_cv.png')), bbox_inches='tight')
    plt.close('all')

    # Convert kriged raster to filled contours vector
    contour = arcpy.GALayerToContour_ga(in_geostat_layer=out_ga_layer,
                                        contour_type=contour_type,
                                        out_feature_class=out_vector,
                                        classification_type=class_type,
                                        classes_count=classes_count)

    arcpy.Delete_management(img_path)




