# Logistic Regression
# Training only on pixels within buffer of flood + randomly sampled pixels outside buffer

import __init__
import os
import time
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import h5py
from scipy.ndimage import binary_dilation
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import pandas as pd
import sys
import rasterio
import numpy as np
from results_viz import VizFuncs
from sklearn.preprocessing import MinMaxScaler
from LR_conf_intervals import get_se, get_probs

sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Python Version:', sys.version)

# ==================================================================================
# Parameters
pctls = [10, 30, 50, 70, 90]

batch = 'LR_sample'

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSWDistSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'precip', 'GSWPerm', 'flooded']

feat_list_all = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'hydgrpA',
                 'hydgrpAD', 'hydgrpB', 'hydgrpBD', 'hydgrpC', 'hydgrpCD', 'hydgrpD',
                 'GSWDistSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'precip', 'GSWPerm', 'flooded']

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new,
              'feat_list_all': feat_list_all}

n_flood, n_nonflood = 250, 250


# ======================================================================================================================
def get_sample_coords(img, pctl, n_flood, n_nonflood):
    """
    Sample pixels: # flood = # nonflood, but 1/2 of nonflood must be within buffer around flood pixels, 1/2 must be outside
    """
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'

    # load cloudmasks
    clouds_dir = data_path / 'clouds'

    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan

    # Mask out clouds
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan

    cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    data[cloudmask] = -999999
    data[data == -999999] = np.nan
    mask = np.sum(data, axis=2)
    data[np.isnan(mask)] = np.nan
    data_train = data.copy()

    perm_index = data.shape[2] - 2
    flood_index = data.shape[2] - 1

    # Set detected flood water that is actually perm water to 0
    data[data[:, :, perm_index] == 1, flood_index] = 0

    # Get mask of NaNs
    nan_mask = np.sum(data, axis=2)
    nan_mask = np.invert(np.isnan(nan_mask))

    # Create buffer around flood pixels
    buffer_iters = 15
    mask = data[:, :, flood_index]
    mask[np.isnan(mask)] = 0
    mask = binary_dilation(mask, iterations=buffer_iters)

    # Sample from nonflood pixels within buffer
    buffer_mask = mask * nan_mask
    buffer_mask[data[:, :, flood_index] == 1] = False
    buffer_inds = np.argwhere(buffer_mask)
    if n_nonflood > buffer_inds.shape[0] * 2:
        n_nonflood = buffer_inds.shape[0] * 2
    sample_nonflood_buff = np.random.choice(buffer_inds.shape[0], np.floor(n_nonflood / 2).astype('int'))
    sample_nonflood_buff = buffer_inds[sample_nonflood_buff, :]

    # Sample from nonflood pixels outside buffer
    nobuffer_mask = np.invert(mask) * nan_mask
    nobuffer_inds = np.argwhere(nobuffer_mask)
    sample_nonflood_nobuff = np.random.choice(nobuffer_inds.shape[0], np.ceil(n_nonflood / 2).astype('int'))
    sample_nonflood_nobuff = nobuffer_inds[sample_nonflood_nobuff, :]

    # Sample from flood pixels
    flood_mask = data[:, :, flood_index]
    flood_mask[np.isnan(flood_mask)] = 0
    flood_inds = np.argwhere(flood_mask)
    if n_flood > flood_inds.shape[0]:
        n_flood = flood_inds.shape[0]
    sample_flood = np.random.choice(flood_inds.shape[0], n_flood)
    sample_flood = flood_inds[sample_flood, :]

    # Export sample coordinates/indices to CSV
    sample_dir = data_path / 'sample_points'
    try:
        sample_dir.mkdir(parents=True)
    except FileExistsError:
        pass
    nonflood_samples = np.concatenate([sample_nonflood_buff, sample_nonflood_nobuff])
    samples = np.concatenate([sample_flood, nonflood_samples], axis=1)
    samples_df = pd.DataFrame(samples, columns=['floodx', 'floody', 'nonfloodx', 'nonfloody'])
    samples_df.to_csv(sample_dir / '{}_sample_points.csv'.format(img), index=False)

    return samples_df, data_train


def get_sample_data(sample_coords, data_train):
    data_vector = data_train.reshape([data_train.shape[0] * data_train.shape[1], data_train.shape[2]])
    sample_coords = pd.DataFrame(np.concatenate((sample_coords.iloc[:, :2].values, sample_coords.iloc[:, 2:4])))
    coords = np.floor(sample_coords).astype('int')
    cols, rows = coords.iloc[:, 0], coords.iloc[:, 1]
    indices = cols * data_train.shape[1] + rows

    data_vector_train = data_vector[indices, :]
    data_vector_train = data_vector_train[~np.isnan(data_vector_train).any(axis=1)]

    return data_vector_train


def standardize_data(data_vector):
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
    scaler = MinMaxScaler().fit(data_vector)
    data_vector_train = scaler.transform(data_vector)
    return data_vector_train, scaler


def standardize_data(data_vector):
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
    shape = data_vector.shape
    mins = np.min(data_vector[:, 0:shape[1] - 2], axis=0)
    maxes = np.max(data_vector[:, 0:shape[1] - 2], axis=0)
    scaler = np.column_stack([mins, maxes])
    # Adjust scaler for binary classes not present in training data that may be present in test
    scaler[:12, 0] = 0
    scaler[:12, 1] = 1
    data_vector_train = data_vector
    data_vector_train[:, 0:shape[1] - 2] = (data_vector_train[:, 0:shape[1] - 2] - scaler[:, 0]) / (scaler[:, 1] - scaler[:, 0])
    scaler = pd.DataFrame(scaler)
    return data_vector_train, scaler


def preprocessing_test(data_path, img, pctl):
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    clouds_dir = data_path / 'clouds'

    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan

    # Mask out clouds
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    cloudmask = np.greater(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    data[cloudmask] = -999999
    data[data == -999999] = np.nan
    mask = np.sum(data, axis=2)
    data[np.isnan(mask)] = np.nan

    return data


def log_reg_training_sample(img_list, pctls, feat_list_new, feat_list_all, data_path, batch, n_flood, n_nonflood):
    for img in img_list:
        print(img + ': stacking tif, generating clouds')
        times = []
        tif_stacker(data_path, img, feat_list_new, overwrite=False)
        cloud_generator(img, data_path, overwrite=False)

        for pctl in pctls:
            print(img, pctl, '% CLOUD COVER')
            print('Preprocessing')

            sample_coords, data_train = get_sample_coords(img, pctl, n_flood, n_nonflood)
            perm_index = data_train.shape[2] - 2
            flood_index = data_train.shape[2] - 1
            data_vector_train = get_sample_data(sample_coords, data_train)
            data_vector_train, scaler = standardize_data(data_vector_train)
            data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
            shape = data_vector_train.shape
            X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]
            model_path = data_path / batch / 'models' / img
            metrics_path = data_path / batch / 'metrics' / 'training' / img / '{}'.format(
                img + '_clouds_' + str(pctl))
            scaler_dir = data_path / 'scalers' / img

            if not model_path.exists():
                model_path.mkdir(parents=True)
            if not metrics_path.exists():
                metrics_path.mkdir(parents=True)
            if not scaler_dir.exists():
                scaler_dir.mkdir(parents=True)

            model_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.sav')
            # scaler_path = scaler_dir / '{}_clouds_{}_scaler_.sav'.format(img, str(pctl))
            # joblib.dump(scaler, scaler_path)
            scaler_path = scaler_dir / '{}_clouds_{}_scaler_.csv'.format(img, str(pctl))
            scaler.to_csv(scaler_path, index=False)

            print('Training')
            start_time = time.time()
            logreg = LogisticRegression(solver='lbfgs')
            logreg.fit(X_train, y_train)
            end_time = time.time()
            times.append(timer(start_time, end_time, False))
            joblib.dump(logreg, model_path)

            del data_train, data_vector_train, logreg

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([pctls, times])
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)


def log_reg_prediction_sample(img_list, pctls, feat_list_all, data_path, batch):
    for j, img in enumerate(img_list):
        times = []
        accuracy, precision, recall, f1, roc_auc = [], [], [], [], []
        preds_path = data_path / batch / 'predictions' / img
        bin_file = preds_path / 'predictions.h5'
        uncertainties_path = data_path / batch / 'uncertainties' / img
        se_lower_bin_file = uncertainties_path / 'se_lower.h5'
        se_upper_bin_file = uncertainties_path / 'se_upper.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for i, pctl in enumerate(pctls):
            print('Preprocessing', img, pctl, '% cloud cover')
            data_test = preprocessing_test(data_path, img, pctl)
            scaler_dir = data_path / 'scalers' / img
            # scaler_path = scaler_dir / '{}_clouds_{}_scaler_.sav'.format(img, str(pctl))
            scaler_path = scaler_dir / '{}_clouds_{}_scaler_.csv'.format(img, str(pctl))
            # scaler = joblib.load(scaler_path)
            scaler = np.array(pd.read_csv(scaler_path))

            perm_index = feat_list_all.index('GSWPerm')
            flood_index = feat_list_all.index('flooded')
            data_vector = data_test.reshape([data_test.shape[0] * data_test.shape[1], data_test.shape[2]])
            data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
            shape = data_vector.shape
            data_vector_test = data_vector
            data_vector_test[:, 0:shape[1] - 2] = (data_vector_test[:, 0:shape[1] - 2] - scaler[:, 0]) / (
                        scaler[:, 1] - scaler[:, 0])
            data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0
            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove perm water column
            shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:shape[1] - 1], data_vector_test[:, shape[1] - 1]

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

            # Computer standard errors
            SE_est = get_se(X_test, y_test, trained_model)
            probs, upper, lower = get_probs(trained_model, X_test, SE_est,
                                            z=1.96)  # probs is redundant, predicted above

            try:
                uncertainties_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(se_lower_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier lower SEs')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=lower)

            with h5py.File(se_upper_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier upper SEs')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=upper)

            times.append(timer(start_time, time.time(), False))

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
            roc_auc.append(roc_auc_score(y_test, pred_probs[:, 1]))

            del preds, probs, pred_probs, upper, lower, X_test, y_test, \
                trained_model, data_test, data_vector_test

        metrics = pd.DataFrame(np.column_stack([pctls, accuracy, precision, recall, f1, roc_auc]),
                               columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        times = [float(i) for i in times]  # Convert time objects to float, otherwise valMetrics will be non-numeric
        times_df = pd.DataFrame(np.column_stack([pctls, times]),
                                columns=['cloud_cover', 'testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)


# ======================================================================================================================
log_reg_training_sample(img_list, pctls, feat_list_new, feat_list_all, data_path, batch, n_flood, n_nonflood)
log_reg_prediction_sample(img_list, pctls, feat_list_all, data_path, batch)
viz = VizFuncs(viz_params)
viz.metric_plots()
viz.metric_plots_multi()
viz.time_plot()
viz.false_map(probs=True, save=False)
viz.false_map_borders()
viz.uncertainty_map_LR()
viz.fpfn_map(probs=True)

