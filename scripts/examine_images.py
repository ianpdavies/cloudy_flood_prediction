# This script is for examining images, features, and predictions to visually identify patterns
# Plot the uncertainties
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import sys
sys.path.append('../')
from CPR.configs import data_path
import h5py
from CPR.utils import preprocessing, tif_stacker


img = '4469_LC08_015036_20170502_1'
pctl = 40
batch = 'v21'
uncertainty = True
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']
myDpi = 300
plt.subplots_adjust(top=0.98, bottom=0.055, left=0.024, right=0.976, hspace=0.2, wspace=0.2)

stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
plot_path = data_path / batch / 'plots' / img
vars_bin_file = data_path / batch / 'variances' / img / 'variances.h5'
preds_bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'
stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

# ======================================================================================================================
# Get data
data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, gaps=False, normalize=True)
feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
perm_index = feat_list_keep.index('GSW_perm')
flood_index = feat_list_keep.index('flooded')
data_vector_train[
    data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
shape = data_vector_train.shape
X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]

# ======================================================================================================================
# Get predictions
print('Fetching prediction variances for', str(pctl) + '{}'.format('%'))
# Read predictions
with h5py.File(vars_bin_file, 'r') as f:
    variances = f[str(pctl)]
    variances = np.array(variances)  # Copy h5 dataset to array

data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, gaps=True,
                                                                      normalize=False)
var_img = np.zeros(shape)
var_img[:] = np.nan
rows, cols = zip(data_ind_test)
var_img[rows, cols] = variances

print('Fetching flood predictions for', str(pctl) + '{}'.format('%'))
# Read predictions
with h5py.File(preds_bin_file, 'r') as f:
    predictions = f[str(pctl)]
    predictions = np.array(predictions)  # Copy h5 dataset to array

prediction_img = np.zeros(shape)
prediction_img[:] = np.nan
rows, cols = zip(data_ind_test)
prediction_img[rows, cols] = predictions


# ======================================================================================================================
# ----------------------------------------------------------
# Uncertainty
# Reshape variance values back into image band
with rasterio.open(stack_path, 'r') as ds:
    shape = ds.read(1).shape  # Shape of full original image

aleatoric_img = np.zeros(shape)
aleatoric_img[:] = np.nan
rows, cols = zip(data_ind_test)
aleatoric_img[rows, cols] = aleatoric_uncertainties

epistemic_img = np.zeros(shape)
epistemic_img[:] = np.nan
epistemic_img[rows, cols] = epistemic_uncertainties

predictions = np.array(np.argmax(aleatoric_softmax, axis=1))
prediction_img = np.zeros(shape)
prediction_img[:] = np.nan
rows, cols = zip(data_ind_test)
prediction_img[rows, cols] = predictions

# ----------------------------------------------------------------
# Plot trues and falses
floods = data_test[:, :, 15]
tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')
falses = fp + fn
trues = tp + tn
# Mask out clouds, etc.
tp = ma.masked_array(tp, mask=p)
tn = ma.masked_array(tn, mask=np.isnan(prediction_img))
fp = ma.masked_array(fp, mask=np.isnan(prediction_img))
fn = ma.masked_array(fn, mask=np.isnan(prediction_img))
falses = ma.masked_array(falses, mask=np.isnan(prediction_img))
trues = ma.masked_array(trues, mask=np.isnan(prediction_img))

true_false = fp + (fn * 2) + (tp * 3)
colors = ['saddlebrown',
          'red',
          'limegreen',
          'blue']
class_labels = ['True Negatives',
                'False Floods',
                'Missed Floods',
                'True Floods']
legend_patches = [Patch(color=icolor, label=label)
                  for icolor, label in zip(colors, class_labels)]
cmap = ListedColormap(colors)

# fig, ax = plt.subplots(figsize=(10, 10))
fig, ax = plt.subplots()
ax.imshow(true_false, cmap=cmap)
ax.legend(handles=legend_patches,
          facecolor='white',
          edgecolor='white')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# myFig = ax.get_figure()
# myFig.savefig(plot_path / 'truefalse.png', dpi=myDpi)
# plt.close('all')

# ----------------------------------------------------------------
# Plot aleatoric uncertainty
fig, ax = plt.subplots()
fig_img = ax.imshow(aleatoric_img, cmap='plasma')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
im_ratio = aleatoric_img.shape[0] / aleatoric_img.shape[1]
fig.colorbar(fig_img, ax=ax, fraction=0.046*im_ratio, pad=0.04*im_ratio)

fig, ax = plt.subplots()
fig_img = ax.imshow(epistemic_img, cmap='plasma')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
im_ratio = epistemic_img.shape[0] / epistemic_img.shape[1]
fig.colorbar(fig_img, ax=ax, fraction=0.046*im_ratio, pad=0.04*im_ratio)

# ----------------------------------------------------------------
# Plot CIR image
from CPR.utils import tif_stacker
from PIL import Image, ImageEnhance

print('Stacking image')
band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
tif_stacker(data_path, img, band_list, features=False, overwrite=False)
spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'


# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    return ((array - array_min) / (array_max - array_min))


print('Processing CIR image')
with rasterio.open(spectra_stack_path, 'r') as f:
    nir, red, green = f.read(5), f.read(4), f.read(3)
    nir[nir == -999999] = np.nan
    red[red == -999999] = np.nan
    green[green == -999999] = np.nan
    nirn = normalize(nir)
    redn = normalize(red)
    greenn = normalize(green)
    cir = np.dstack((nirn, redn, greenn))
    cir[np.isnan(prediction_img)] = np.nan

# Convert to PIL image, enhance, and save
cir_img = Image.fromarray((cir * 255).astype(np.uint8()))
cir_img = ImageEnhance.Contrast(cir_img).enhance(1.5)
cir_img = ImageEnhance.Sharpness(cir_img).enhance(2)
cir_img = ImageEnhance.Brightness(cir_img).enhance(2)

fig, ax = plt.subplots()
ax.imshow(cir_img)

