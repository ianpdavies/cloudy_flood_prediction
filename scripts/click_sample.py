import __init__
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import os
import sys
from PIL import Image
from rasterio.windows import Window

sys.path.append('../')
from CPR.configs import data_path

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSWDistSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'precip', 'GSWPerm', 'flooded']

feat_list_all = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'hydgrpA',
                 'hydgrpAD', 'hydgrpB', 'hydgrpBD', 'hydgrpC', 'hydgrpCD', 'hydgrpD',
                 'GSWDistSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'precip', 'GSWPerm', 'flooded']

img = '4444_LC08_043034_20170303_1'
# ======================================================================================================================
# Plot flood on top of RGB, then create sample points by clicking
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d' % (
        ix, iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == 10:
        fig.canvas.mpl_disconnect(cid)
        plt.close()
    return coords

flood_index = feat_list_all.index('flooded')
perm_index = feat_list_all.index('GSWPerm')

stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
band_combo_dir = data_path / 'band_combos'
rgb_file = band_combo_dir / '{}'.format(img + '_rgb_img' + '.png')
rgb_img = Image.open(rgb_file)

with rasterio.open(stack_path, 'r') as src:
    flood = src.read(flood_index + 1)
    perm = src.read(perm_index + 1)
    flood[flood == -999999] = np.nan
    flood[flood == 0] = np.nan
    flood[perm == 1] = 0

coords = []
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.imshow(rgb_img)
plt.imshow(flood, cmap='autumn_r')
cid = fig.canvas.mpl_connect('button_press_event', onclick)
flood_coords = coords

plt.waitforbuttonpress()
while True:
    if plt.waitforbuttonpress():
        break
plt.close()

coords = []
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.imshow(rgb_img)
plt.imshow(flood, cmap='autumn_r')
cid = fig.canvas.mpl_connect('button_press_event', onclick)
nonflood_coords = coords

plt.waitforbuttonpress()
while True:
    if plt.waitforbuttonpress():
        break
plt.close()

# Save sample coordinates in csv
sample_dir = data_path / 'sample_points'
try:
    sample_dir.mkdir(parents=True)
except FileExistsError:
    pass

sample_flood_df = pd.DataFrame(flood_coords, columns=['floodx', 'floody'])
sample_nonflood_df = pd.DataFrame(nonflood_coords, columns=['nonfloodx', 'nonfloody'])
sample_df = pd.concat([sample_flood_df, sample_nonflood_df], axis=1)
sample_df.to_csv(sample_dir / '{}_sample_points.csv'.format(img), index=False)

# # ======================================================================================================================
# # Get data at sample point locations
# myCoords = coords.copy()
# myCoords = np.floor(myCoords).astype('int')
# cols, rows = zip(*myCoords)
#
# with rasterio.open(str(stack_path), 'r') as ds:
#     data = ds.read()
#     data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
#     data[data == -999999] = np.nan
#     data[np.isneginf(data)] = np.nan
#
# data_train = data[rows, cols, 1:]
# data_vector_train = data_train[~np.isnan(data_train).any(axis=1)]
#
# # ======================================================================================================================
# from sklearn.linear_model import LogisticRegression
#
# shape = data_vector_train.shape
# X_train, y_train = data_vector_train[:, 0:shape[1] - 2], data_vector_train[:, shape[1] - 2]
#
# logreg = LogisticRegression(n_jobs=-1, solver='sag')
# logreg.fit(X_train, y_train)
#
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
#
# from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
#
# pctl = 30
# data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new, test=True)
# data_vector_test = data_vector_test[:, 1:]
# X_test, y_test = data_vector_test[:, 0:shape[1] - 2], data_vector_test[:, shape[1] - 2]
# pred_probs = logreg.predict_proba(X_test)
# preds = np.argmax(pred_probs, axis=1)
#
# perm_mask = data_test[:, :, perm_index]
# perm_mask = perm_mask.reshape([perm_mask.shape[0] * perm_mask.shape[1]])
# perm_mask = perm_mask[~np.isnan(perm_mask)]
# preds[perm_mask.astype('bool')] = 0
# y_test[perm_mask.astype('bool')] = 0
#
# accuracy_score(y_test, preds)
# precision_score(y_test, preds)
# recall_score(y_test, preds)
# f1_score(y_test, preds)
# roc_auc_score(y_test, pred_probs[:, 1])
