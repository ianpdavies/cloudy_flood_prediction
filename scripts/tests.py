import sys

sys.path.append('../')
import tensorflow as tf
import sys
import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from CPR.utils import preprocessing, tif_stacker, timer
from PIL import Image, ImageEnhance
import h5py

sys.path.append('../')
import numpy.ma as ma
import time
import dask.array as da

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Parameters

# uncertainty = False
# batch = 'v28'
# # pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# pctls = [10, 30, 50, 70, 90]
# BATCH_SIZE = 8192
# EPOCHS = 100
# DROPOUT_RATE = 0.3  # Dropout rate for MCD
# HOLDOUT = 0.3  # Validation data size
# NUM_PARALLEL_EXEC_UNITS = 4
# remove_perm = True
# MC_PASSES = 20
#
# try:
#     (data_path / batch).mkdir()
# except FileExistsError:
#     pass
#
# # To get list of all folders (images) in directory
# # img_list = os.listdir(data_path / 'images')
#
# img_list = ['4444_LC08_044033_20170222_2',
#             '4101_LC08_027038_20131103_1',
#             '4101_LC08_027038_20131103_2',
#             '4101_LC08_027039_20131103_1',
#             '4115_LC08_021033_20131227_1',
#             '4115_LC08_021033_20131227_2',
#             '4337_LC08_026038_20160325_1',
#             '4444_LC08_043034_20170303_1',
#             '4444_LC08_043035_20170303_1',
#             '4444_LC08_044032_20170222_1',
#             '4444_LC08_044033_20170222_1',
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
#
# # Order in which features should be stacked to create stacked tif
# feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
#                  'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

# ======================================================================================================================
# def highlight_plot():
#     print('Making highlight plots')
#     plt.ioff()
#     metrics_path = data_path / batch / 'metrics' / 'testing_nn'
#     plot_path = data_path / batch / 'plots'
#     try:
#         plot_path.mkdir(parents=True)
#     except FileExistsError:
#         pass
#
#     colors = sns.color_palette("colorblind", 4)
#
#     metrics = ['accuracy', 'recall', 'precision', 'f1']
#     file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
#     df_concat = pd.concat(pd.read_csv(file) for file in file_list)
#     mean_metrics = df_concat.groupby('cloud_cover').mean().reset_index()
#
#     for i, metric in enumerate(metrics):
#         plt.figure(figsize=(7, 5), dpi=300)
#         for file in file_list:
#             metrics = pd.read_csv(file)
#             plt.plot(metrics['cloud_cover'], metrics[metric], color=colors[i], linewidth=1, alpha=0.3)
#         plt.plot(mean_metrics['cloud_cover'], mean_metrics[metric], color=colors[i], linewidth=3, alpha=0.9)
#         plt.ylim(0, 1)
#         plt.xlabel('Cloud Cover', fontsize=13)
#         plt.ylabel(metric.capitalize(), fontsize=13)
#         plt.savefig(plot_path / '{}'.format(metric + '_highlight.png'))
#
# highlight_plot()
#         # metrics_fig = metrics_plot.get_figure()
#         # metrics_fig.savefig(plot_path / 'metrics_plot.png')

# ======================================================================================================================
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import time
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import time
import os
# Import custom functions
import sys
from models import get_aleatoric_uncertainty_model as model_func
from models import get_epistemic_uncertainty_model

sys.path.append('../')
# from CPR.configs import data_path
from CPR.utils import tif_stacker, cloud_generator, preprocessing, train_val, timer


# def training6(img_list, pctls, model_func, feat_list_new, data_path, batch, T,
#               dropout_rate=0.2, **model_params):




uncertainty = True
batch = 'v29'
pctls = [40]
batch_size = 8192
epochs = 100
T = 50
dropout_rate = 0.2
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


get_model = model_func
for j, img in enumerate(img_list):
    print(img + ': stacking tif, generating clouds')
    times = []
    tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
    cloud_generator(img, data_path, overwrite=False)

    for i, pctl in enumerate(pctls):
        print(img, pctl, '% CLOUD COVER')
        print('Preprocessing')
        tf.keras.backend.clear_session()
        data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, gaps=False, normalize=True)
        feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
        perm_index = feat_list_keep.index('GSW_perm')
        flood_index = feat_list_keep.index('flooded')
        data_vector_train[
            data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
        data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
        shape = data_vector_train.shape
        X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]
        y_train = to_categorical(y_train)
        INPUT_DIMS = X_train.shape[1]

        model_path = data_path / batch / 'models' / img
        metrics_path = data_path / batch / 'metrics' / 'training_nn' / img / '{}'.format(
            img + '_clouds_' + str(pctl))

        try:
            metrics_path.mkdir(parents=True)
            model_path.mkdir(parents=True)
        except FileExistsError:
            pass

        model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='softmax_output_categorical_accuracy', min_delta=0.005, patience=5),
                     tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor='loss',
                                                        save_best_only=True),
                     CSVLogger(metrics_path / 'training_log.log')]

        start_time = time.time()
        model = get_model(model_params['epochs'], X_train, y_train, X_train.shape, T, D=2,
                          batch_size=model_params['batch_size'], dropout_rate=dropout_rate, callbacks=callbacks)
        end_time = time.time()
        times.append(timer(start_time, end_time, False))
        # model.save(model_path)

    metrics_path = metrics_path.parent
    times = [float(i) for i in times]
    times = np.column_stack([pctls, times])
    times_df = pd.DataFrame(times, columns=['cloud_cover', 'training_time'])
    times_df.to_csv(metrics_path / 'training_times.csv', index=False)