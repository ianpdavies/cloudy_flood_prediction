import pickle
import pandas as pd
import numpy as np
import time
import h5py
import dask.array as da
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('../')
from CPR.configs import data_path
from models import get_nn1 as get_model
from CPR.utils import preprocessing, timer
# ==================================================================================
# Parameters

# Image to predict on
# img_list = ['4115_LC08_021033_20131227_test']
# img_list = ['4101_LC08_027038_20131103_1']
img_list = ['4337_LC08_026038_20160325_1']

pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# pctls = [90]

# ==================================================================================
# Loop to do predictions

for j, img in enumerate(img_list):
    precision = []
    recall = []
    f1 = []
    accuracy = []
    times = []
    # pred_list = []
    gapMetricsList = []
    variances = []
    preds_path = data_path / 'predictions' / img
    bin_file = preds_path / 'mean_predictions.h5'
    metrics_path = data_path / 'metrics' / 'testing_nn' / img

    try:
        metrics_path.mkdir(parents=True)
    except FileExistsError:
        print('Metrics directory already exists')

    for i, pctl in enumerate(pctls):

        data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True)
        X_test, y_test = data_vector_test[:, 0:14], data_vector_test[:, 14]
        INPUT_DIMS = X_test.shape[1]

        print('Predicting for {} at {}% cloud cover'.format(img, pctl))

        # There is a problem loading keras models: https://github.com/keras-team/keras/issues/10417
        # But loading the trained weights into an identical compiled (but untrained) model works
        start_time = time.time()
        trained_model = get_model(INPUT_DIMS)  # Get untrained model to add trained weights into
        model_path = data_path / 'models' / 'nn' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
        trained_model.load_weights(str(model_path))
        preds = trained_model.predict(X_test, batch_size=7000, use_multiprocessing=True)
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
        # pred_list.append(list(preds))
        accuracy.append(accuracy_score(y_test, preds))
        precision.append(precision_score(y_test, preds))
        recall.append(recall_score(y_test, preds))
        f1.append(f1_score(y_test, preds))

        times = [float(i) for i in times]  # Need to convert time objects to float, otherwise valMetrics will be non-numeric

        gapMetrics = pd.DataFrame(np.column_stack([pctls[0:i+1], accuracy, precision, recall, f1, times]),
                                  columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1', 'time'])

        gapMetrics.to_csv(metrics_path / 'gapMetrics.csv', index=False)