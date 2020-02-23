import __init__
from models import get_nn_bn2 as model_func
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

batch = 'NN_allwater'
pctls = [10, 30, 50, 70, 90]
BATCH_SIZE = 8192
EPOCHS = 100

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
img_list.remove('4115_LC08_021033_20131227_test')

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


NUM_PARALLEL_EXEC_UNITS = 4
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = NUM_PARALLEL_EXEC_UNITS,
         inter_op_parallelism_threads = 2,
         allow_soft_placement = True,
         device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

# ======================================================================================================================
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from training import LrRangeFinder, SGDRScheduler, lr_plots
from CPR.utils import tif_stacker, preprocessing, cloud_generator, timer
import numpy as np
import pandas as pd
import time
import h5py


def NN_training(img_list, pctls, model_func, feat_list_new, data_path, batch, **model_params):
    get_model = model_func
    for j, img in enumerate(img_list):
        print(img + ': stacking tif, generating clouds')
        times = []
        lr_mins = []
        lr_maxes = []
        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
        cloud_generator(img, data_path, overwrite=False)

        for i, pctl in enumerate(pctls):
            print(img, pctl, '% CLOUD COVER')
            print('Preprocessing')
            tf.keras.backend.clear_session()
            data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl,
                                                                                     feat_list_new, test=False)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            # data_vector_train[data_vector_train[:, perm_index] == 1, flood_index] = 0
            data_vector_train = np.delete(data_vector_train, perm_index, axis=1)
            shape = data_vector_train.shape
            X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]
            INPUT_DIMS = X_train.shape[1]

            model_path = data_path / batch / 'models' / img
            metrics_path = data_path / batch / 'metrics' / 'training' / img / '{}'.format(
                img + '_clouds_' + str(pctl))

            lr_plots_path = metrics_path.parents[1] / 'lr_plots'
            lr_vals_path = metrics_path.parents[1] / 'lr_vals'
            try:
                metrics_path.mkdir(parents=True)
                model_path.mkdir(parents=True)
                lr_plots_path.mkdir(parents=True)
                lr_vals_path.mkdir(parents=True)
            except FileExistsError:
                pass

            # ---------------------------------------------------------------------------------------------------
            # Determine learning rate by finding max loss decrease during single epoch training
            lrRangeFinder = LrRangeFinder(start_lr=0.1, end_lr=2)

            lr_model_params = {'batch_size': model_params['batch_size'],
                               'epochs': 1,
                               'verbose': 2,
                               'callbacks': [lrRangeFinder],
                               'use_multiprocessing': True}

            model = model_func(INPUT_DIMS)

            print('Finding learning rate')
            model.fit(X_train, y_train, **lr_model_params)
            lr_min, lr_max, lr, losses = lr_plots(lrRangeFinder, lr_plots_path, img, pctl)
            lr_mins.append(lr_min)
            lr_maxes.append(lr_max)
            # ---------------------------------------------------------------------------------------------------
            # Training the model with cyclical learning rate scheduler
            model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
            scheduler = SGDRScheduler(min_lr=lr_min, max_lr=lr_max, lr_decay=0.9, cycle_length=3, mult_factor=1.5)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0.0001, patience=10),
                tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor='loss',
                                                   save_best_only=True),
                CSVLogger(metrics_path / 'training_log.log'),
                scheduler]

            model = get_model(INPUT_DIMS)

            print('Training full model with best LR')
            start_time = time.time()
            model.fit(X_train, y_train, **model_params, callbacks=callbacks)
            end_time = time.time()
            times.append(timer(start_time, end_time, False))
            # model.save(model_path)

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([pctls, times])
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)

        lr_range = np.column_stack([pctls, lr_mins, lr_maxes])
        lr_avg = np.mean(lr_range[:, 1:2], axis=1)
        lr_range = np.column_stack([lr_range, lr_avg])
        lr_range_df = pd.DataFrame(lr_range, columns=['cloud_cover', 'lr_min', 'lr_max', 'lr_avg'])
        lr_range_df.to_csv((lr_vals_path / img).with_suffix('.csv'), index=False)

        losses_path = lr_vals_path / img / '{}'.format('losses_' + str(pctl) + '.csv')
        try:
            losses_path.parent.mkdir(parents=True)
        except FileExistsError:
            pass
        lr_losses = np.column_stack([lr, losses])
        lr_losses = pd.DataFrame(lr_losses, columns=['lr', 'losses'])
        lr_losses.to_csv(losses_path, index=False)


def NN_prediction(img_list, pctls, feat_list_new, data_path, batch, **model_params):
    for j, img in enumerate(img_list):
        times = []
        accuracy, precision, recall, f1 = [], [], [], []
        preds_path = data_path / batch / 'predictions' / img
        bin_file = preds_path / 'predictions.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for i, pctl in enumerate(pctls):
            print('Preprocessing', img, pctl, '% cloud cover')
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new, test=True)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
            data_shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:data_shape[1]-1], data_vector_test[:, data_shape[1]-1]

            print('Predicting for {} at {}% cloud cover'.format(img, pctl))
            start_time = time.time()
            model_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
            trained_model = tf.keras.models.load_model(model_path)
            preds = trained_model.predict(X_test, batch_size=model_params['batch_size'], use_multiprocessing=True)
            preds = np.argmax(preds, axis=1)  # Display most probable value

            try:
                preds_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier mean predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=preds)

            times.append(timer(start_time, time.time(), False))  # Elapsed time for MC simulations

            perm_mask = data_test[:, :, perm_index]
            perm_mask = perm_mask.reshape([perm_mask.shape[0] * perm_mask.shape[1]])
            perm_mask = perm_mask[~np.isnan(perm_mask)]
            preds[perm_mask.astype('bool')] = 0
            y_test[perm_mask.astype('bool')] = 0

            print('Evaluating predictions')
            accuracy.append(accuracy_score(y_test, preds))
            precision.append(precision_score(y_test, preds))
            recall.append(recall_score(y_test, preds))
            f1.append(f1_score(y_test, preds))

            del preds, X_test, y_test, trained_model, data_test, data_vector_test, data_ind_test

        metrics = pd.DataFrame(np.column_stack([pctls, accuracy, precision, recall, f1]),
                               columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        times = [float(i) for i in times]  # Convert time objects to float, otherwise valMetrics will be non-numeric
        times_df = pd.DataFrame(np.column_stack([pctls, times]),
                                columns=['cloud_cover', 'testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)


# ======================================================================================================================
NN_training(img_list[img_list.index('4101_LC08_027038_20131103_1'):], pctls, model_func, feat_list_new, data_path, batch, **model_params)
NN_prediction(img_list, pctls, feat_list_new, data_path, batch, **model_params)
viz = VizFuncs(viz_params)
viz.metric_plots()
viz.metric_plots_multi()
viz.median_highlight()
viz.time_plot()
viz.false_map(probs=False, save=False)
viz.false_map_borders()







