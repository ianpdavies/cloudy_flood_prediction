# For use with Standard ArcGIS with Geostatistical Analyst Extension
# Requires Python 2

import __init__z
import arcpy
from arcpy import env
from arcpy.sa import *
import os
import pathlib
import sys
sys.path.append('../')
from CPR.configs import data_path

# ======================================================================================================================

# Set environment settings
env.workspace = str(data_path / 'precip')
arcpy.CheckOutExtension("GeoStats")

img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

for img in img_list:
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

    # Get buffer raster information
    buffer_path = str(data_path / 'images' / img / '{}'.format(img + '_buffer.tif'))
    buffer = arcpy.Raster(buffer_path)
    description = arcpy.Describe(buffer)
    cellsize1 = description.children[0].meanCellHeight  # Cell size y
    cellsize2 = description.children[0].meanCellWidth  # Cell size x
    print('Cellsize:', cellsize1, cellsize2)

    # Set kriging parameters
    in_features = img + '_precip.shp'
    field = "prcp"
    cellsize = cellsize1
    out_ga_layer = 'kriged_rasters/' + img + '_krig_ga.lyr'
    transformation = 'EMPIRICAL'
    semivariogram = 'EXPONENTIAL'
    output_type = 'PREDICTION'

    # Set contour filling parameters
    out_vector = 'kriged_vectors/' + img + '_krig_vector.shp'
    class_type = 'QUANTILE'
    contour_type = 'PREDICTION'
    classes_count = 15

    # Empirical Bayesian Kriging
    ebayes_krig = arcpy.EmpiricalBayesianKriging_ga(in_features=in_features,
                                                    z_field=field,
                                                    out_ga_layer='ga_layer',
                                                    cell_size=cellsize,
                                                    transformation_type=transformation,
                                                    output_type=output_type,
                                                    semivariogram_model_type=semivariogram)
    arcpy.SaveToLayerFile_management(in_layer=ebayes_krig, out_layer=out_ga_layer, is_relative_path='RELATIVE')

    # Convert kriged raster to filled contours vector
    contour = arcpy.GALayerToContour_ga(in_geostat_layer=out_ga_layer,
                                        contour_type=contour_type,
                                        out_feature_class=out_vector,
                                        classification_type=class_type,
                                        classes_count=classes_count)





