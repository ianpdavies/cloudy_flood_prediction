from models import get_nn_bn2 as model_func
import tensorflow as tf
import os
import pathlib
from training import training3, SGDRScheduler, LrRangeFinder
from prediction import prediction
from results_viz import VizFuncs
import sys
import shutil

sys.path.append('../../')
from CPR.configs import data_path


# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Testing on all images (at 10, 30, 50, 70, 90%) with random cloud masks to see if poor performance is due to randomness
# 2 layer nn with batch norm
# TRIAL 5/5
# Batch size = 8192
# ==================================================================================
# Parameters

uncertainty = False
batch = 'v17'
trial = 'trial5'
pctls = [10, 30, 50, 70, 90]
BATCH_SIZE = 8192
EPOCHS = 100
DROPOUT_RATE = 0.3  # Dropout rate for MCD
HOLDOUT = 0.3  # Validation data size

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

# img_list = ['4115_LC08_021033_20131227_test']
img_list = ['4444_LC08_044033_20170222_2',
            '4101_LC08_027038_20131103_1',
            '4101_LC08_027038_20131103_2',
            '4101_LC08_027039_20131103_1',
            '4115_LC08_021033_20131227_1',
            '4115_LC08_021033_20131227_2',
            '4337_LC08_026038_20160325_1',
            '4444_LC08_043034_20170303_1',
            '4444_LC08_043035_20170303_1',
            '4444_LC08_044032_20170222_1',
            '4444_LC08_044033_20170222_1',
            '4444_LC08_044033_20170222_3',
            '4444_LC08_044033_20170222_4',
            '4444_LC08_044034_20170222_1',
            '4444_LC08_045032_20170301_1',
            '4468_LC08_022035_20170503_1',
            '4468_LC08_024036_20170501_1',
            '4468_LC08_024036_20170501_2',
            '4469_LC08_015035_20170502_1',
            '4469_LC08_015036_20170502_1',
            '4477_LC08_022033_20170519_1',
            '4514_LC08_027033_20170826_1']

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

# Set some optimized config parameters
NUM_PARALLEL_EXEC_UNITS = 4
tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)
# tf.config.experimental.set_visible_devices(NUM_PARALLEL_EXEC_UNITS, 'CPU')
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

# ==================================================================================
# Training and prediction with random batches of clouds
print('RUNNING', trial, '################################################################')
cloud_dir = data_path / 'clouds'

model_params = {'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'verbose': 2,
                'use_multiprocessing': True}

training3(img_list, pctls, model_func, feat_list_new, uncertainty,
          data_path, batch, DROPOUT_RATE, HOLDOUT, **model_params)
prediction(img_list, pctls, feat_list_new, data_path, batch, remove_perm=True, **model_params)
viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'uncertainty': uncertainty,
              'batch': batch,
              'feat_list_new': feat_list_new}
viz = VizFuncs(viz_params)
viz.metric_plots()
viz.time_plot()
viz.false_map()
viz.metric_plots_multi()
viz.time_size()

# Move cloud files to another folder so they're not overwritten
for img in img_list:
    file_name = img + '_clouds.npy'
    cloud_src = cloud_dir / file_name
    cloud_dest_dir = cloud_dir / 'random' / trial
    cloud_dest = cloud_dest_dir / file_name
    try:
        cloud_dest_dir.mkdir(parents=True)
    except FileExistsError:
        pass
    if not cloud_dest.exists():
        shutil.move(cloud_src, cloud_dest)
    else:
        print('Overwriting previous random cloud trial!')
        sys.exit()