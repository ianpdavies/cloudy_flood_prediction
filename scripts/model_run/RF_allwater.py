# Random Forest using hyperparameters tuned for only one image at 50% CC

import __init__
import tensorflow as tf
import os
import sys
from results_viz import VizFuncs
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import h5py
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
from CPR.configs import data_path
import skopt
from skopt import forest_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# Version numbers
print('Python Version:', sys.version)

# ==================================================================================
# Parameters

batch = 'RF_allwater'
pctls = [10, 30, 50, 70, 90]

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

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}


# ======================================================================================================================


def rf_training(img_list, pctls, feat_list_new, data_path, batch, n_jobs):
    for j, img in enumerate(img_list):
        print(img + ': stacking tif, generating clouds')
        times = []
        # tuning_times = []
        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
        cloud_generator(img, data_path, overwrite=False)

        for i, pctl in enumerate(pctls):
            print(img, pctl, '% CLOUD COVER')
            print('Preprocessing')
            tf.keras.backend.clear_session()
            data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl,
                                                                                     feat_list_new,
                                                                                     test=False)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            # data_vector_train[data_vector_train[:, perm_index] == 1, flood_index] = 0
            data_vector_train = np.delete(data_vector_train, perm_index, axis=1)
            shape = data_vector_train.shape
            X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]

            model_path = data_path / batch / 'models' / img
            metrics_path = data_path / batch / 'metrics' / 'training' / img / '{}'.format(
                img + '_clouds_' + str(pctl))

            if not model_path.exists():
                model_path.mkdir(parents=True)
            if not metrics_path.exists():
                metrics_path.mkdir(parents=True)

            param_path = data_path / batch / 'models' / '4514_LC08_027033_20170826_1' / '{}'.format(
                '4514_LC08_027033_20170826_1_clouds_50params.pkl')

            model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + '.sav')

            # param_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_10_params.pkl')

            # if pctl is 10:
            # model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + '.sav')

            # # Hyperparameter optimization
            # print('Hyperparameter search')
            # base_rf = RandomForestClassifier(random_state=0, n_estimators=100, max_leaf_nodes=10)

            # space = [skopt.space.Integer(2, 1000, name="max_leaf_nodes"),
            # skopt.space.Integer(2, 200, name="n_estimators"),
            # skopt.space.Integer(2, 3000, name="max_depth")]

            # @use_named_args(space)
            # def objective(**params):
            # base_rf.set_params(**params)
            # return -np.mean(cross_val_score(base_rf, X_train, y_train, cv=5, n_jobs=n_jobs, scoring="f1"))

            # start_time = time.time()
            # res_rf = forest_minimize(objective, space, base_estimator='RF', n_calls=30,
            # random_state=0, verbose=True, n_jobs=n_jobs)
            # end_time = time.time()
            # tuning_times.append(timer(start_time, end_time, False))
            # print(type(res_rf))
            # skopt.utils.dump(res_rf, param_path, store_objective=False)

            # if pctl is not 10:
            # res_rf = skopt.utils.load(param_path)

            res_rf = skopt.utils.load(param_path)

            # Training
            print('Training with optimized hyperparameters')
            start_time = time.time()
            rf = RandomForestClassifier(random_state=0,
                                        max_leaf_nodes=res_rf.x[0],
                                        n_estimators=res_rf.x[1],
                                        max_depth=res_rf.x[2],
                                        n_jobs=-1)
            rf.fit(X_train, y_train)
            end_time = time.time()
            times.append(timer(start_time, end_time, False))
            joblib.dump(rf, model_path)

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([pctls, times])
        # pd.DataFrame(tuning_times, columns=['tuning_time']).to_csv(metrics_path / 'tuning_time.csv', index=False)
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)


def prediction_rf(img_list, pctls, feat_list_new, data_path, batch):
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
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                                  test=True)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0
            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)
            data_shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:data_shape[1] - 1], data_vector_test[:, data_shape[1] - 1]

            print('Predicting for {} at {}% cloud cover'.format(img, pctl))
            start_time = time.time()
            model_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.sav')
            trained_model = joblib.load(model_path)
            pred_probs = trained_model.predict_proba(X_test)
            preds = np.argmax(pred_probs, axis=1)

            try:
                preds_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier mean predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=pred_probs)

            times.append(timer(start_time, time.time(), False))  # Elapsed time for MC simulations

            print('Evaluating predictions')
            perm_mask = data_test[:, :, perm_index]
            perm_mask = perm_mask.reshape([perm_mask.shape[0] * perm_mask.shape[1]])
            perm_mask = perm_mask[~np.isnan(perm_mask)]
            preds[perm_mask.astype('bool')] = 0
            y_test[perm_mask.astype('bool')] = 0

            accuracy.append(accuracy_score(y_test, preds))
            precision.append(precision_score(y_test, preds))
            recall.append(recall_score(y_test, preds))
            f1.append(f1_score(y_test, preds))

            del preds, pred_probs, X_test, y_test, trained_model, data_test, data_vector_test, data_ind_test

        metrics = pd.DataFrame(np.column_stack([pctls, accuracy, precision, recall, f1]),
                               columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        times = [float(i) for i in times]  # Convert time objects to float, otherwise valMetrics will be non-numeric
        times_df = pd.DataFrame(np.column_stack([pctls, times]),
                                columns=['cloud_cover', 'testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)


# ======================================================================================================================
rf_training(img_list, pctls, feat_list_new, data_path, batch, n_jobs=None)
prediction_rf(img_list, pctls, feat_list_new, data_path, batch)
viz = VizFuncs(viz_params)
viz.metric_plots()
viz.metric_plots_multi()
viz.time_plot()
viz.false_map(probs=True, save=False)
viz.false_map_borders()
viz.fpfn_map()
