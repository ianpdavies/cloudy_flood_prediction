from models import get_nn1 as model_func
import tensorflow as tf
from training import training
from prediction import prediction
from evaluation import evaluation
from results_viz import VizFuncs
import sys
sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Parameters


uncertainty = False

# Image to predict on
# img_list = ['4115_LC08_021033_20131227_test']
# img_list = ['4101_LC08_027038_20131103_1', '4101_LC08_027038_20131103_2']
img_list = ['4101_LC08_027039_20131103_1',
            '4115_LC08_021033_20131227_1',
            '4337_LC08_026038_20160325_1']

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


# training(img_list, pctls, model_func, feat_list_new, uncertainty,
#          data_path, DROPOUT_RATE, HOLDOUT,  **model_params)
#
# prediction(img_list, pctls, feat_list_new, data_path)
#
# evaluation(img_list, pctls, feat_list_new, data_path)

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'uncertainty': uncertainty}

viz = VizFuncs(viz_params)
viz.metric_plots()
viz.false_map()

importlib.reload(results_viz)
from results_viz import VizFuncs