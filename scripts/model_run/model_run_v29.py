from models import get_aleatoric_uncertainty_model as model_func
from models import get_epistemic_uncertainty_model
import tensorflow as tf
import os
from training import training6
from prediction import prediction
from results_viz import VizFuncs
import sys
sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# All images at 10-90%, no MCD, no val data, using get_nn_bn2
# Batch size = 8192
# ==================================================================================
# Parameters

uncertainty = True
batch = 'v29'
pctls = [40]
BATCH_SIZE = 8192
EPOCHS = 100
T = 50
DROPOUT_RATE = 0.2
NUM_PARALLEL_EXEC_UNITS = 4

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

img_list = ['4444_LC08_044033_20170222_2',
            '4101_LC08_027038_20131103_2',
            '4115_LC08_021033_20131227_1',
            '4337_LC08_026038_20160325_1',
            '4444_LC08_043035_20170303_1',
            '4444_LC08_044033_20170222_1',
            '4444_LC08_044033_20170222_4',
            '4444_LC08_045032_20170301_1',
            '4468_LC08_024036_20170501_1',
            '4469_LC08_015035_20170502_1',
            '4514_LC08_027033_20170826_1']

img_list = ['4514_LC08_027033_20170826_1']

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

model_params = {'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'verbose': 2,
                'use_multiprocessing': True}

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'uncertainty': uncertainty,
              'batch': batch,
              'feat_list_new': feat_list_new}

# Set some optimized config parameters
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

cloud_dir = data_path / 'clouds'

training6(img_list, pctls, model_func, feat_list_new, data_path, batch, T,
              DROPOUT_RATE, **model_params)

# prediction(img_list, pctls, feat_list_new, data_path, batch, remove_perm=True, **model_params)

# viz = VizFuncs(viz_params)
# viz.metric_plots()
# viz.color_images()
# viz.time_plot()
# viz.false_map()
# viz.metric_plots_multi()
# viz.time_size()
