# NN without batch normalization
import __init__
from models import get_nn_bn2_noBN as model_func
import tensorflow as tf
import os
from training import training4
from prediction import prediction
from results_viz import VizFuncs
import sys
sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Parameters

batch = 'NN_noBN'
pctls = [10, 30, 50, 70, 90]
BATCH_SIZE = 8192
EPOCHS = 100

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
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

model_params = {'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'verbose': 2,
                'use_multiprocessing': True}

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}

# Set some optimized config parameters
NUM_PARALLEL_EXEC_UNITS = os.cpu_count()
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=4,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

os.environ['MKL_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ['GOTO_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)

# ==================================================================================
training4(img_list, pctls, model_func, feat_list_new, data_path, batch, **model_params)
prediction(img_list, pctls, feat_list_new, data_path, batch, remove_perm=True, **model_params)
viz = VizFuncs(viz_params)
viz.metric_plots()
viz.cir_image()
viz.time_plot()
viz.false_map(probs=False, save=False)
viz.false_map_borders()
viz.metric_plots_multi()
viz.median_highlight()
