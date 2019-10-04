import tensorflow as tf
import pandas as pd
import time
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
# Import custom functions
import sys
sys.path.append('../')
# from CPR.configs import data_path
from CPR.utils import tif_stacker, cloud_generator, preprocessing, train_val, timer
# from models import get_nn_uncertainty1 as get_model
# from models import get_nn1 as model_func


# ==================================================================================


def training(img_list, pctls, model_func, feat_list_new, uncertainty, data_path,
             DROPOUT_RATE=0, HOLDOUT=0.3, **model_params):

    get_model = model_func

    for j, img in enumerate(img_list):

        times = []

        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=True)
        cloud_generator(img, data_path, overwrite=False)

        for i, pctl in enumerate(pctls):
            data_train, data_vector_train, data_ind_train = preprocessing(data_path, img, pctl, gaps=False)
            perm_index = feat_list_new.index('GSW_perm')
            data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove GSW_perm column

            training_data, validation_data = train_val(data_vector_train, holdout=HOLDOUT)
            X_train, y_train = training_data[:, 0:14], training_data[:, 14]
            X_val, y_val = validation_data[:, 0:14], validation_data[:, 14]
            INPUT_DIMS = X_train.shape[1]

            if uncertainty:
                model_path = data_path / 'models' / 'nn_mcd' / img
                metrics_path = data_path / 'metrics' / 'training_nn_mcd' / img / '{}'.format(img + '_clouds_' + str(pctl))
            else:
                model_path = data_path / 'models' / 'nn' / img
                metrics_path = data_path / 'metrics' / 'training_nn' / img / '{}'.format(img + '_clouds_' + str(pctl))

            try:
                metrics_path.mkdir(parents=True)
                model_path.mkdir(parents=True)
            except FileExistsError:
                pass

            model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')

            csv_logger = CSVLogger(metrics_path / 'training_log.log')
            model_params['callbacks'].append(csv_logger)

            print('~~~~~', img, pctl, '% CLOUD COVER')

            if uncertainty:
                model = get_model(INPUT_DIMS, DROPOUT_RATE)  # Model with uncertainty
            else:
                model = get_model(INPUT_DIMS)  # Model without uncertainty

            start_time = time.time()
            model.fit(X_train, y_train, **model_params, validation_data=(X_val, y_val))

            end_time = time.time()
            times.append(timer(start_time, end_time, False))

            model.save(model_path)

        times = [float(i) for i in times]
        times_df = pd.DataFrame(times, columns=['times'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)

