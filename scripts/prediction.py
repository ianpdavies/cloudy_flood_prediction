import pandas as pd
import numpy as np
import time
import h5py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from CPR.utils import preprocessing, timer
import tensorflow as tf
import tensorflow.keras.backend as K
from models import load_bayesian_model
from tensorflow import keras

# ==================================================================================


def prediction(img_list, pctls, feat_list_new, data_path, batch, **model_params):
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

from models import get_epistemic_uncertainty_model
from tensorflow.keras.utils import to_categorical
import os

def prediction_bnn(img_list, pctls, feat_list_new, data_path, batch, MC_passes):
    for j, img in enumerate(img_list):
        epistemic_times = []
        aleatoric_times = []
        accuracy, precision, recall, f1 = [], [], [], []
        preds_path = data_path / batch / 'predictions' / img
        bin_file = preds_path / 'predictions.h5'
        aleatoric_bin_file = preds_path / 'aleatoric_predictions.h5'
        uncertainties_path = data_path / batch / 'uncertainties' / img
        aleatoric_uncertainty_file = uncertainties_path / 'aleatoric_uncertainties.h5'
        epistemic_uncertainty_file = uncertainties_path / 'epistemic_uncertainties.h5'
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
            y_test = to_categorical(y_test)
            D = len(set(y_test[:, 0]))  # Target classes
            iterable = K.variable(np.ones(MC_passes))

            print('Predicting (aleatoric) for {} at {}% cloud cover'.format(img, pctl))
            model_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
            start_time = time.time()
            # aleatoric_model = tf.keras.models.load_model(model_path)
            aleatoric_model = load_bayesian_model(model_path, MC_passes, D, iterable)
            aleatoric_results = aleatoric_model.predict(X_test, verbose=1)
            aleatoric_uncertainties = np.reshape(aleatoric_results[0][:, D:], (-1))
            try:
                uncertainties_path.mkdir(parents=True)
            except FileExistsError:
                pass
            with h5py.File(aleatoric_uncertainty_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier aleatoric uncertainties')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=aleatoric_uncertainties)
            logits = aleatoric_results[0][:, 0:D]
            aleatoric_preds = np.argmax(aleatoric_results[1], axis=1)
            aleatoric_times.append(timer(start_time, time.time(), False))
            try:
                preds_path.mkdir(parents=True)
            except FileExistsError:
                pass
            with h5py.File(aleatoric_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier aleatoric predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=aleatoric_preds)

            print('Predicting (epistemic) for {} at {}% cloud cover'.format(img, pctl))
            start_time = time.time()
            epistemic_model = get_epistemic_uncertainty_model(model_path, T=MC_passes, D=D)
            epistemic_results = epistemic_model.predict(X_test, verbose=2, use_multiprocessing=True)
            epistemic_uncertainties = epistemic_results[0]
            with h5py.File(epistemic_uncertainty_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier epistemic uncertainties')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=epistemic_uncertainties)
            epistemic_preds = np.argmax(epistemic_results[1], axis=1)
            epistemic_times.append(timer(start_time, time.time(), False))
            with h5py.File(bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier epistemic predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=epistemic_preds)

            print('Evaluating predictions')
            accuracy.append(accuracy_score(y_test[:, 1], epistemic_preds))
            precision.append(precision_score(y_test[:, 1], epistemic_preds))
            recall.append(recall_score(y_test[:, 1], epistemic_preds))
            f1.append(f1_score(y_test[:, 1], epistemic_preds))

            del aleatoric_model, aleatoric_results, aleatoric_uncertainties, logits, aleatoric_preds, \
                epistemic_model, epistemic_uncertainties, epistemic_preds, epistemic_results, \
                data_test, data_vector_test, data_ind_test

        metrics = pd.DataFrame(np.column_stack([pctls, accuracy, precision, recall, f1]),
                               columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        epistemic_times = [float(i) for i in epistemic_times]
        aleatoric_times = [float(i) for i in aleatoric_times]
        times_df = pd.DataFrame(np.column_stack([pctls, epistemic_times, aleatoric_times]),
                                columns=['cloud_cover', 'epistemic_testing_time', 'aleatoric_testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)



def prediction_bnn_kwon(img_list, pctls, feat_list_new, data_path, batch, MC_passes, **model_params):
    for j, img in enumerate(img_list):
        times = []
        accuracy, precision, recall, f1 = [], [], [], []
        preds_path = data_path / batch / 'predictions' / img
        bin_file = preds_path / 'predictions.h5'

        uncertainties_path = data_path / batch / 'uncertainties' / img
        aleatoric_bin_file = uncertainties_path / 'aleatoric_uncertainties.h5'
        epistemic_bin_file = uncertainties_path / 'epistemic_uncertainties.h5'
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
            model = tf.keras.models.load_model(model_path)
            p_hat = []
            for t in range(MC_passes):
                p_hat.append(model.predict(X_test, batch_size=model_params['batch_size'], use_multiprocessing=True)[:, 1])
            p_hat = np.array(p_hat)
            preds = np.round(np.mean(p_hat, axis=0))
            aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
            epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

            try:
                preds_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier mean predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=preds)

            try:
                uncertainties_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(epistemic_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier epistemic predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=epistemic)

            with h5py.File(aleatoric_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier epistemic predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=aleatoric)

            times.append(timer(start_time, time.time(), False))

            print('Evaluating predictions')
            accuracy.append(accuracy_score(y_test, preds))
            precision.append(precision_score(y_test, preds))
            recall.append(recall_score(y_test, preds))
            f1.append(f1_score(y_test, preds))

            del preds, p_hat, aleatoric, epistemic, X_test, y_test, model, data_test, data_vector_test, data_ind_test

        metrics = pd.DataFrame(np.column_stack([pctls, accuracy, precision, recall, f1]),
                               columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        times = [float(i) for i in times]
        times_df = pd.DataFrame(np.column_stack([pctls, times]),
                                columns=['cloud_cover', 'testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)

