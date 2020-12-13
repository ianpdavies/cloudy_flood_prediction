import os
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import h5py
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import sys
sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)


# =====================================================================================
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

img_list = ['4444_LC08_044033_20170222_2',
            '4101_LC08_027038_20131103_1',
            '4101_LC08_027038_20131103_2',
            '4101_LC08_027039_20131103_1',
            '4115_LC08_021033_20131227_1',
            '4115_LC08_021033_20131227_2',
            '4337_LC08_026038_20160325_1',
            '4444_LC08_043034_20170303_1',
            '4444_LC08_043035_20170303_1',
            '4444_LC08_044032_20170222_1',
            '4444_LC08_044033_20170222_1',
            '4444_LC08_044033_20170222_3',
            '4444_LC08_044033_20170222_4',
            '4444_LC08_044034_20170222_1',
            '4444_LC08_045032_20170301_1',
            '4468_LC08_022035_20170503_1',
            '4468_LC08_024036_20170501_1',
            '4468_LC08_024036_20170501_2',
            '4469_LC08_015035_20170502_1',
            '4469_LC08_015036_20170502_1',
            '4477_LC08_022033_20170519_1',
            '4514_LC08_027033_20170826_1']

img_list = ['4101_LC08_027038_20131103_2']

img = img_list[0]
batch = 'v30'
pctl = 30
remove_perm=True

times = []
accuracy, precision, recall, f1 = [], [], [], []
preds_path = data_path / batch / 'predictions' / img
bin_file = preds_path / 'predictions.h5'
metrics_path = data_path / batch / 'metrics' / 'testing' / img

print('Preprocessing', img, pctl, '% cloud cover')
data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                      test=True)
if remove_perm:
    perm_index = feat_keep.index('GSW_perm')
    flood_index = feat_keep.index('flooded')
    data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
data_shape = data_vector_test.shape
X_test, y_test = data_vector_test[:, 0:data_shape[1]-1], data_vector_test[:, data_shape[1]-1]

print('Predicting for {} at {}% cloud cover'.format(img, pctl))
# There is a problem loading keras models: https://github.com/keras-team/keras/issues/10417
# Workaround is to use load_model: https://github.com/keras-team/keras-tuner/issues/75
start_time = time.time()
model_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.sav')
trained_model = joblib.load(model_path)
pred_probs = trained_model.predict_proba(X_test)
preds = np.argmax(pred_probs, axis=1)

# Get SEs (from: https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients)

# Design matrix -- add column of 1's at the beginning of your X_train matrix
X_design = np.hstack([np.ones((data_shape[0], 1)), X_test])

# Initiate matrix of 0's, fill diagonal with each predicted observation's variance
V = np.product(pred_probs, axis=1)

# Covariance matrix
covLogit = np.linalg.pinv(np.dot(X_design.T * V, X_design))

# Standard errors
se = np.sqrt(np.diag(covLogit))

# Plot uncertainty and FP/FN
import matplotlib.pyplot as plt
import rasterio
from PIL import Image, ImageEnhance

print('Creating FN/FP map for {}'.format(img))
plot_path = data_path / batch / 'plots' / img
bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'

stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

# Get RGB image
print('Stacking RGB image')
band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
tif_stacker(data_path, img, band_list, features=False, overwrite=False)
spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'


# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    return ((array - array_min) / (array_max - array_min))


print('Processing RGB image')
with rasterio.open(spectra_stack_path, 'r') as f:
    red, green, blue = f.read(4), f.read(3), f.read(2)
    red[red == -999999] = np.nan
    green[green == -999999] = np.nan
    blue[blue == -999999] = np.nan
    redn = normalize(red)
    greenn = normalize(green)
    bluen = normalize(blue)
    rgb = np.dstack((redn, greenn, bluen))

# Convert to PIL image, enhance, and save
rgb_img = Image.fromarray((rgb * 255).astype(np.uint8()))
rgb_img = ImageEnhance.Contrast(rgb_img).enhance(1.5)
rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(2)
rgb_img = ImageEnhance.Brightness(rgb_img).enhance(2)

plt.figure()
plt.imshow(rgb_img)

print('Saving RGB image')
# rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
# rgb_img.save(rgb_file, dpi=(300, 300))

# Reshape predicted values back into image band
with rasterio.open(stack_path, 'r') as ds:
    shape = ds.read(1).shape  # Shape of full original image


print('Fetching flood prediction probs for', str(pctl) + '{}'.format('%'))

# Add prob values to cloud-covered pixel positions
probs_img = np.zeros(shape)
probs_img[:] = np.nan
rows, cols = zip(data_ind_test)
probs_img[rows, cols] = pred_probs[:, 1]
plt.figure()
plt.imshow(probs_img)

prediction_img = np.zeros(shape)
prediction_img[:] = np.nan
rows, cols = zip(data_ind_test)
prediction_img[rows, cols] = preds

# Remove perm water from predictions and actual
perm_index = feat_keep.index('GSW_perm')
flood_index = feat_keep.index('flooded')
# data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water

data_shape = data_vector_test.shape
with rasterio.open(stack_path, 'r') as ds:
    perm_feat = ds.read(perm_index + 1)
    prediction_img[perm_feat == 1] = 0

# Add actual flood values to cloud-covered pixel positions
flooded_img = np.zeros(shape)
flooded_img[:] = np.nan
flooded_img[rows, cols] = data_vector_test[:, data_shape[1] - 1]
plt.figure()
plt.imshow(flooded_img)

# Visualizing FNs/FPs
ones = np.ones(shape=shape)
red_actual = np.where(ones, flooded_img, 0.5)  # Actual
blue_preds = np.where(ones, prediction_img, 0.5)  # Predictions
green_combo = np.minimum(red_actual, blue_preds)
comparison_img = np.dstack((red_actual, green_combo, blue_preds))
plt.figure()
plt.imshow(comparison_img)
