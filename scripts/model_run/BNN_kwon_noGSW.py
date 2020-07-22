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
batch = 'BNN_kwon_noGSW'
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
removed = {'4115_LC08_021033_20131227_test'}
img_list = [x for x in img_list if x not in removed]


# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

model_params = {'epochs': epochs,
                'batch_size': batch_size}

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}

# NUM_PARALLEL_EXEC_UNITS = 8
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=4,
                                  # allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)
# os.environ["KMP_BLOCKTIME"] = "30"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
# os.environ['MKL_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
# os.environ['GOTO_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
# os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)

# ======================================================================================================================
training_bnn_kwon(img_list, pctls, model_func, feat_list_new, data_path, batch, dropout_rate, **model_params)
prediction_bnn_kwon(img_list, pctls, feat_list_new, data_path, batch, MC_passes, **model_params)
viz = VizFuncs(viz_params)
viz.time_plot()
viz.false_map(probs=False, save=False)
viz.false_map_borders()
viz.uncertainty_map_NN()
viz.fpfn_map(probs=False)
