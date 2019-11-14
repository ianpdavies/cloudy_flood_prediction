import pandas as pd
import numpy as np
import time
import h5py
import dask.array as da
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('../')
from CPR.configs import data_path
import tensorflow as tf
from CPR.utils import preprocessing, timer

# ==================================================================================


def prediction_with_uncertainty(img_list, pctls, feat_list_new, data_path, batch, DROPOUT_RATE, MC_PASSES, remove_perm,
                                weight_decay=0.005, length_scale=0.00001, **model_params):
    for j, img in enumerate(img_list):
        times = []
        accuracy, precision, recall, f1 = [], [], [], []
        preds_path = data_path / batch / 'predictions' / 'nn' / img
        vars_path = data_path / batch / 'variances' / 'nn' / img
        mc_bin_file = preds_path / 'mc_preds.h5'
        preds_bin_file = preds_path / 'predictions.h5'
        vars_bin_file = vars_path / 'variances.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing_nn' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')
        try:
            preds_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')
        try:
            vars_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for i, pctl in enumerate(pctls):
            print('Preprocessing', img, pctl, '% cloud cover')
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, gaps=True)
            feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
            if remove_perm:
                perm_index = feat_list_keep.index('GSW_perm')
                flood_index = feat_list_keep.index('flooded')
                data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
            data_shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:data_shape[1]-1], data_vector_test[:, data_shape[1]-1]

            # Initialize binary file to hold predictions
            with h5py.File(mc_bin_file, 'w') as f:
                f.create_dataset('mc_preds', shape=(X_test.shape[0], 1),
                                 maxshape=(X_test.shape[0], None),
                                 chunks=True, compression='gzip')  # Create empty dataset with shape of data

            start_time = time.time()
            model_path = data_path / batch / 'models' / 'nn' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
            trained_model = tf.keras.models.load_model(model_path)

            for k in range(MC_PASSES):
                if k % 10 == 0 or k == MC_PASSES - 1:
                    print('Running MC {}/{} for {} at {}% cloud cover'.format(k, MC_PASSES, img, pctl))
                flood_prob = trained_model.predict(X_test, batch_size=model_params['batch_size'], use_multiprocessing=True)  # Predict
                flood_prob = flood_prob[:, 1]  # Drop probability of not flooded (0) to save space
                with h5py.File(mc_bin_file, 'a') as f:
                    f['mc_preds'][:, -1] = flood_prob  # Append preds to h5 file
                    if k < MC_PASSES - 1:  # Resize to append next pass, if there is one
                        f['mc_preds'].resize((f['mc_preds'].shape[1] + 1), axis=1)
                del flood_prob

            # Calculate MC statistics
            print('Calculating MC statistics for {} at {}% cloud cover'.format(img, pctl))
            with h5py.File(mc_bin_file, 'r') as f:
                dset = f['mc_preds']
                preds_da = da.from_array(dset, chunks="400 MiB")  # Open h5 file as dask array
                means = preds_da.mean(axis=1)
                means = means.compute()
                variance = preds_da.var(axis=1)
                variance = variance.compute()
                tau = (length_scale**2 * (1 - DROPOUT_RATE)) / (2 * data_shape[0] * weight_decay)
                variance = variance + tau
                preds = means.round()
                del f, means, preds_da, dset

            os.remove(mc_bin_file)  # Delete predictions to save space on disk

            print('Saving mean preds/vars for {} at {}% cloud cover'.format(img, pctl))
            with h5py.File(preds_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier mean predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=preds)
            with h5py.File(vars_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier variances')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=variance)

            times.append(timer(start_time, time.time(), False))  # Elapsed time for MC simulations

            print('Evaluating predictions for {} at {}% cloud cover'.format(img, pctl))
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