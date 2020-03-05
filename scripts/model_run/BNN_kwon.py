import __init__
import tensorflow as tf
from models import get_nn_bn2_kwon_v2 as model_func
import os
from training import training_bnn_kwon
from prediction import prediction_bnn_kwon
from results_viz import VizFuncs
import sys

sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)
# ==================================================================================
# Parameters

pctls = [10, 30, 50, 70, 90]
batch = 'BNN_kwon'
batch_size = 8192
epochs = 50
MC_passes = 50
dropout_rate = 0.2

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'GSW_perm', 'flooded']

feat_list_all = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'carbonate', 'noncarbonate', 'akl_intrusive',
                 'silicic_resid', 'silicic_resid', 'extrusive_volcanic', 'colluvial_sed', 'glacial_till_clay',
                 'glacial_till_loam', 'glacial_till_coarse', 'glacial_lake_sed_fine', 'glacial_outwash_coarse',
                 'hydric', 'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed', 'alluv_coastal_sed_fine',
                 'coastal_sed_coarse', 'GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope', 'spi',
                 'twi', 'sti', 'GSW_perm', 'flooded']

model_params = {'epochs': epochs,
                'batch_size': batch_size}

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}


# ======================================================================================================================
training_bnn_kwon(img_list, pctls, model_func, feat_list_new, feat_list_all, data_path, batch, dropout_rate, **model_params)
prediction_bnn_kwon(img_list, pctls, feat_list_all, data_path, batch, MC_passes, **model_params)
viz = VizFuncs(viz_params)
viz.metric_plots()
viz.metric_plots_multi()
viz.time_plot()
viz.false_map(probs=False, save=False)
viz.false_map_borders()
viz.fpfn_map()
viz.uncertainty_map_NN()

