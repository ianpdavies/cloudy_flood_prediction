# Logistic regression
# But trained on all water, tested on perm=0, preds=0 for metrics
# Based on batch LR_perm_3 in LR_perm script

import __init__
import os
import time
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import pandas as pd
from results_viz import VizFuncs
import sys
import numpy as np
import h5py
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy.ma as ma
from LR_conf_intervals import get_se, get_probs

sys.path.append('../')
from CPR.configs import data_path

# ==================================================================================
# Parameters
pctls = [10, 30, 50, 70, 90]

batch = 'LR_allwater2'

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm',
                 'flooded']

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}

# ======================================================================================================================
import rasterio
import matplotlib.pyplot as plt
probs = True
data_path = data_path
my_dpi = 300
# Get predictions and variances

img = '4101_LC08_027039_20131103_1'
pctl = 90
batch = 'trial2'

print('Creating FN/FP map for {}'.format(img))
plot_path = data_path / batch / 'plots' / img
preds_bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'
stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

try:
    plot_path.mkdir(parents=True)
except FileExistsError:
    pass

# Reshape variance values back into image band
with rasterio.open(stack_path, 'r') as ds:
    shape = ds.read(1).shape  # Shape of full original image

data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl,
                                                                      feat_list_new, test=True)

print('Fetching flood predictions for', str(pctl) + '{}'.format('%'))
with h5py.File(preds_bin_file, 'r') as f:
    predictions = f[str(pctl)]
    if probs:
        predictions = np.argmax(np.array(predictions), axis=1)  # Copy h5 dataset to array
    if not probs:
        predictions = np.array(predictions)

prediction_img = np.zeros(shape)
prediction_img[:] = np.nan
rows, cols = zip(data_ind_test)
prediction_img[rows, cols] = predictions


perm_index = feat_keep.index('GSW_perm')
flood_index = feat_keep.index('flooded')
floods = data_test[:, :, flood_index]
perm_water = (data_test[:, :, perm_index] == 1)
tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')

# Mask out clouds, etc.
tp = ma.masked_array(tp, mask=np.isnan(prediction_img))
fp = ma.masked_array(fp, mask=np.isnan(prediction_img))
fn = ma.masked_array(fn, mask=np.isnan(prediction_img))

true_false = fp + (fn * 2) + (tp * 3)
true_false[perm_water] = -1

colors = []
class_labels = []
if np.sum(perm_water) != 0:
    colors.append('darkgrey')
    class_labels.append('Permanent Water')
if np.sum(tn) != 0:
    colors.append('saddlebrown')
    class_labels.append('True Negatives')
if np.sum(fp) != 0:
    colors.append('limegreen')
    class_labels.append('False Floods')
if np.sum(fn) != 0:
    colors.append('red')
    class_labels.append('Missed Floods')
if np.sum(tp) != 0:
    colors.append('blue')
    class_labels.append('True Floods')

legend_patches = [Patch(color=icolor, label=label)
                  for icolor, label in zip(colors, class_labels)]
cmap = ListedColormap(colors)
fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(true_false, cmap=cmap)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
          ncol=5, borderaxespad=0, frameon=False, prop={'size': 7})
plt.tight_layout()
plt.savefig(plot_path / '{}'.format('map_fpfn_' + str(pctl) + '.png'), dpi=my_dpi, pad_inches=0.0)