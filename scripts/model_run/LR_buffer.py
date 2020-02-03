# Logistic Regression
# Training only on pixels within buffer of all detected water (not just floods)

import __init__
import tensorflow as tf
import os
import multiprocessing
import time
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from results_viz import VizFuncs
import h5py
from scipy.ndimage import binary_dilation
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import pandas as pd
import sys
import rasterio
import numpy as np

sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Python Version:', sys.version)

# ==================================================================================
# Parameters
batch = 'LR_buffer'
pctls = [30]
buffer_iters = [5, 10, 20, 30, 40]
NUM_PARALLEL_EXEC_UNITS = os.cpu_count()

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
img_list = os.listdir(data_path / 'images')

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new,
              'buffer_iters': buffer_iters}

# Set some optimized config parameters
tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.set_soft_device_placement(True)
# tf.config.experimental.set_visible_devices(NUM_PARALLEL_EXEC_UNITS, 'CPU')
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


# ======================================================================================================================


def log_reg_training_buffer(img_list, pctls, feat_list_new, data_path, batch, buffer_iters, buffer_flood_only):
    for img in img_list:
        print(img + ': stacking tif, generating clouds')
        times = []
        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
        cloud_generator(img, data_path, overwrite=False)

        for pctl in pctls:
            print('Preprocessing')
            data_train_full, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl,
                                                                                          feat_list_new,
                                                                                          test=False)
            for buffer_iter in buffer_iters:
                perm_index = feat_keep.index('GSW_perm')
                flood_index = feat_keep.index('flooded')
                data_train = data_train_full.copy()
                if buffer_flood_only:
                    data_train[data_train[:, :, perm_index] == 1, flood_index] = 0
                    mask = data_train[:, :, flood_index]
                    buffer_mask = np.invert(binary_dilation(mask, iterations=buffer_iter))
                else:
                    mask = data_train[:, :, flood_index]
                    buffer_mask = np.invert(binary_dilation(mask, iterations=buffer_iter))
                    data_train[data_train[:, :, perm_index] == 1, flood_index] = 0
                data_train[buffer_mask] = np.nan

                data_vector_train = data_train.reshape([data_train.shape[0] * data_train.shape[1], data_train.shape[2]])
                data_vector_train = data_vector_train[~np.isnan(data_vector_train).any(axis=1)]
                data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
                shape = data_vector_train.shape
                X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]

                model_path = data_path / batch / 'models' / img
                metrics_path = data_path / batch / 'metrics' / 'training' / img / '{}'.format(
                    img + '_clouds_' + str(pctl))

                if not model_path.exists():
                    model_path.mkdir(parents=True)
                if not metrics_path.exists():
                    metrics_path.mkdir(parents=True)

                model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + 'buff' + str(buffer_iter) + '.sav')

                # Save data flooding image to check that buffering is working correctly
                # imwrite(model_path.parents[0] / '{}'.format('buff' + str(buffer_iter) + '.jpg'), data_train[:, :, 6])

                print('Training')
                start_time = time.time()
                logreg = LogisticRegression(n_jobs=-1, solver='sag')
                logreg.fit(X_train, y_train)
                end_time = time.time()
                times.append(timer(start_time, end_time, False))
                joblib.dump(logreg, model_path)

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([np.repeat(pctls, len(buffer_iters)), np.tile(buffer_iters, len(pctls)), times])
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'buffer_iters', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)


def prediction_buffer(img_list, pctls, feat_list_new, data_path, batch, remove_perm, buffer_iters):
    for img in img_list:
        times = []
        accuracy, precision, recall, f1 = [], [], [], []
        preds_path = data_path / batch / 'predictions' / img
        bin_file = preds_path / 'predictions.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for pctl in pctls:
            print('Preprocessing', img, pctl, '% cloud cover')
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                                  test=True)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[
                data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water

            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
            data_shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:data_shape[1] - 1], data_vector_test[:, data_shape[1] - 1]

            for buffer_iter in buffer_iters:
                print('Predicting for {} at {}% cloud cover'.format(img, pctl))
                start_time = time.time()
                model_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_' + str(pctl) +
                                                                              'buff' + str(buffer_iter) + '.sav')
                trained_model = joblib.load(model_path)
                pred_probs = trained_model.predict_proba(X_test)
                preds = np.argmax(pred_probs, axis=1)

                try:
                    preds_path.mkdir(parents=True)
                except FileExistsError:
                    pass

                with h5py.File(bin_file, 'a') as f:
                    pred_name = str(pctl) + '_buff_' + str(buffer_iter)
                    if pred_name in f:
                        print('Deleting earlier mean predictions')
                        del f[pred_name]
                    f.create_dataset(pred_name, data=pred_probs)

                times.append(timer(start_time, time.time(), False))  # Elapsed time for MC simulations

                print('Evaluating predictions')
                accuracy.append(accuracy_score(y_test, preds))
                precision.append(precision_score(y_test, preds))
                recall.append(recall_score(y_test, preds))
                f1.append(f1_score(y_test, preds))

            del preds, X_test, y_test, trained_model, data_test, data_vector_test, data_ind_test

        metrics = pd.DataFrame(np.column_stack([np.repeat(pctls, len(buffer_iters)),
                                                np.tile(buffer_iters, len(pctls)), accuracy, precision, recall, f1]),
                               columns=['cloud_cover', 'buffer_iters', 'accuracy', 'precision', 'recall', 'f1'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        times = [float(i) for i in times]  # Convert time objects to float, otherwise valMetrics will be non-numeric
        times_df = pd.DataFrame(
            np.column_stack([np.repeat(pctls, len(buffer_iters)), np.tile(buffer_iters, len(pctls)), times]),
            columns=['cloud_cover', 'buffer_iters', 'testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)


# ======================================================================================================================

log_reg_training_buffer(img_list, pctls, feat_list_new, data_path, batch, buffer_iters, buffer_flood_only=True)
prediction_buffer(img_list, pctls, feat_list_new, data_path, batch, True, buffer_iters)

viz = VizFuncs(viz_params)
viz.metric_plots()
viz.cir_image()
viz.time_plot()
viz.false_map(probs=True, save=False)
viz.false_map_borders()
viz.metric_plots_multi()
viz.median_highlight()
