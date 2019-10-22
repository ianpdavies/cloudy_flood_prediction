from models import get_nn1 as model_func
import tensorflow as tf
import os
from training import training1, training2
from prediction import prediction
from evaluation import evaluation
from results_viz import VizFuncs
import sys

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Parameters

uncertainty = False
batch = 'v2'

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

img_list = ['4115_LC08_021033_20131227_test']
# img_list = ['4101_LC08_027038_20131103_1',
#             '4101_LC08_027038_20131103_2',
#             '4101_LC08_027039_20131103_1',
#             '4115_LC08_021033_20131227_1',
#             '4115_LC08_021033_20131227_2',
#             '4337_LC08_026038_20160325_1',
#             '4444_LC08_043034_20170303_1',
#             '4444_LC08_043035_20170303_1',
#             '4444_LC08_044032_20170222_1',
#             '4444_LC08_044033_20170222_1',
#             '4444_LC08_044033_20170222_2',
#             '4444_LC08_044033_20170222_3',
#             '4444_LC08_044033_20170222_4',
#             '4444_LC08_044034_20170222_1',
#             '4444_LC08_045032_20170301_1',
#             '4468_LC08_022035_20170503_1',
#             '4468_LC08_024036_20170501_1',
#             '4468_LC08_024036_20170501_2',
#             '4469_LC08_015035_20170502_1',
#             '4469_LC08_015036_20170502_1',
#             '4477_LC08_022033_20170519_1',
#             '4514_LC08_027033_20170826_1']


# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
BATCH_SIZE = 7000
EPOCHS = 1000
DROPOUT_RATE = 0.3  # Dropout rate for MCD
HOLDOUT = 0.3  # Validation data size
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=15, verbose=1)

model_params = {'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'verbose': 1,
                'callbacks': [es],
                'use_multiprocessing': True}

# ==================================================================================
# Training and prediction

training2(img_list, pctls, model_func, feat_list_new, uncertainty,
         data_path, batch, DROPOUT_RATE, HOLDOUT,  **model_params)

prediction(img_list, pctls, feat_list_new, data_path, batch, remove_perm=True)

evaluation(img_list, pctls, feat_list_new, data_path, batch, remove_perm=True)

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'uncertainty': uncertainty,
              'batch': batch,
              'feat_list_new': feat_list_new}

viz = VizFuncs(viz_params)
viz.metric_plots()
viz.time_plot()
# viz.metric_plots_multi()
viz.false_map()
# viz.time_size()

