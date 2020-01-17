# Logistic Regression in sklearn vs statsmodel

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import h5py
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import sys
sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Training on half of images WITH validation data. Compare with v24
# Batch size = 8192
# ==================================================================================
# Parameters

uncertainty = False  # Should be True if running with MCD
batch = 'test'
pctls = [10, 30, 50, 70, 90]
NUM_PARALLEL_EXEC_UNITS = 4

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

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

img_list = ['4514_LC08_027033_20170826_1']

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

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

# ======================================================================================================================


img = img_list[0]
pctl = 30
batch = 'test'

print(img + ': stacking tif, generating clouds')
times = []
tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
cloud_generator(img, data_path, overwrite=False)

print(img, pctl, '% CLOUD COVER')
print('Preprocessing')
tf.keras.backend.clear_session()
data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                         test=False)
perm_index = feat_keep.index('GSW_perm')
flood_index = feat_keep.index('flooded')
data_vector_train[
    data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
shape = data_vector_train.shape
X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]

# Logistic regression using sklearn
model_path = data_path / batch / 'models' / img
if not model_path.exists():
    model_path.mkdir(parents=True)
model_path = model_path / '{}'.format(img + '_sklearn.sav')
print('Training sklearn')
start_time = time.time()
logreg_sk = LogisticRegression(n_jobs=-1, solver='sag')
logreg_sk.fit(X_train, y_train)
end_time = time.time()
print('sklearn training time:', end_time)
times.append(timer(start_time, end_time, False))
joblib.dump(logreg_sk, model_path)

# Logistic regression using statsmodel
from tensorflow.keras.utils import to_categorical
y_train_cat = 1 - to_categorical(y_train)
import statsmodels.api as sm
model_path = data_path / batch / 'models' / img
if not model_path.exists():
    model_path.mkdir(parents=True)
model_path = model_path / '{}'.format(img + '_statsmodel.pickle')
print('Training statsmodel')
start_time = time.time()
logreg_sm = sm.GLM(y_train_cat, X_train, family=sm.families.Binomial())
result = logreg_sm.fit()
print(result.summary())
end_time = time.time()
print('statsmodel training time:', timer(start_time, end_time, False))
result.save(str(model_path))


# Prediction
preds_path = data_path / batch / 'predictions' / img
bin_file = preds_path / 'predictions.h5'
data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new, test=True)
perm_index = feat_keep.index('GSW_perm')
flood_index = feat_keep.index('flooded')
data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
data_shape = data_vector_test.shape
X_test, y_test = data_vector_test[:, 0:data_shape[1]-1], data_vector_test[:, data_shape[1]-1]

# Prediction sklearn
accuracy, precision, recall, f1 = [], [], [], []
model_path = data_path / batch / 'models' / img
model_path = model_path / '{}'.format(img + '_sklearn.sav')
trained_model = joblib.load(model_path)
preds = trained_model.predict(X_test)
with h5py.File(bin_file, 'a') as f:
    if 'sklearn' in f:
        print('Deleting earlier sklearn predictions')
        del f['sklearn']
    f.create_dataset('sklearn', data=preds)
accuracy.append(accuracy_score(y_test, preds))
precision.append(precision_score(y_test, preds))
recall.append(recall_score(y_test, preds))
f1.append(f1_score(y_test, preds))
sklearn_metrics = np.column_stack([accuracy, precision, recall, f1])

# Prediction statsmodel
accuracy, precision, recall, f1 = [], [], [], []
model_path = data_path / batch / 'models' / img
model_path = model_path / '{}'.format(img + '_statsmodel.pickle')
trained_model = sm.load(str(model_path))
preds = trained_model.predict(X_test)
with h5py.File(bin_file, 'a') as f:
    if 'statsmodel' in f:
        print('Deleting earlier statsmodel predictions')
        del f['statsmodel']
    f.create_dataset('statsmodel', data=preds)
accuracy.append(accuracy_score(y_test, preds))
precision.append(precision_score(y_test, preds))
recall.append(recall_score(y_test, preds))
f1.append(f1_score(y_test, preds))
statsmodel_metrics = np.column_stack([accuracy, precision, recall, f1])

metrics = np.row_stack([sklearn_metrics, statsmodel_metrics])
print(metrics)
