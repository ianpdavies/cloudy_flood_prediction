import tensorflow as tf
import pandas as pd
import time
from tensorflow.keras.callbacks import CSVLogger
# Import custom functions
import sys
sys.path.append('../')
from CPR.configs import data_path
from CPR.utils import tif_stacker, cloud_generator, preprocessing, train_val, timer
# from models import get_nn_uncertainty1 as get_model
from models import get_nn1 as get_model

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Parameters

uncertainty = False

# Image to predict on
img_list = ['4115_LC08_021033_20131227_test']
# already did '4101_LC08_027038_20131103_1'
# img_list = ['4101_LC08_027038_20131103_2',
#             '4101_LC08_027039_20131103_1',
#             '4115_LC08_021033_20131227_1',
#             '4337_LC08_026038_20160325_1']

# Order in which features should be stacked to create stacked tif
feat_list_new = ['aspect','curve', 'developed', 'GSW_distExtent', 'elevation', 'forest',
 'GSW_maxExtent', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

# pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
pctls = [90]
BATCH_SIZE = 7000
EPOCHS = 1000
DROPOUT_RATE = 0.3
HOLDOUT = 0.3 # Validation data size
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=15, verbose=1)

valMetricsList = []

# Stack layers into a single tif, and generate cloud cover image
for j, img in enumerate(img_list):
    precision = []
    recall = []
    f1 = []
    accuracy = []
    times = []
    history = []

    tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
    cloud_generator(img, data_path, overwrite=False)

    for i, pctl in enumerate(pctls):
        data_train, data_vector_train, data_ind_train = preprocessing(data_path, img, pctl, gaps=False)
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

        print('~~~~~', img, pctl, '% CLOUD COVER')

        if uncertainty:
            model = get_model(INPUT_DIMS, DROPOUT_RATE)  # Model with uncertainty
        else:
            model = get_model(INPUT_DIMS)  # Model without uncertainty

        start_time = time.time()

        model.fit(X_train, y_train,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   verbose=1,
                   validation_data=(X_val, y_val),
                   callbacks=[es, csv_logger],
                   use_multiprocessing=True)

        end_time = time.time()
        times.append(timer(start_time, end_time, False))

        model.save(model_path)

    times = [float(i) for i in times]
    times_df = pd.DataFrame(times, columns=['times'])
    times_df.to_csv(metrics_path / 'training_times.csv', index=False)

