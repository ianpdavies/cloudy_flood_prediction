import __init__
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, \
    concatenate, BatchNormalization, Activation
from tensorflow_probability import distributions
from tensorflow.keras import models
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Layer
from tensorflow.keras import Model
import os
import sys

# Set some optimized config parameters
NUM_PARALLEL_EXEC_UNITS = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)
# tf.config.experimental.set_visible_devices(NUM_PARALLEL_EXEC_UNITS, 'CPU')
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

# ======================================================================================================================
# Models come from Kyle Dorman: https://github.com/kyle-dorman/bayesian-neural-network-blogpost
def get_aleatoric_uncertainty_model(epochs, X_train, Y_train, input_shape, T, D):
    tf.keras.backend.clear_session()
    inp = Input(shape=input_shape[1:])
    x = inp
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(24, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x, training=True)
    # x = Dense(12, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(x, training=True)

    means = Dense(D, name='means')(x)
    log_var_pre = Dense(1, name='log_var_pre')(x)
    log_var = Activation('softplus', name='log_var')(log_var_pre)  # What does this do?
    logits_variance = concatenate([means, log_var], name='logits_variance')
    softmax_output = Activation('softmax', name='softmax_output')(means)
    model = models.Model(inputs=inp, outputs=[logits_variance, softmax_output])
    # model = models.Model(inputs=inp, outputs=logits_variance)

    iterable = K.variable(np.ones(T))
    def heteroscedastic_categorical_crossentropy(true, pred):
        mean = pred[:, :D]
        log_var = pred[:, D:]
        log_std = K.sqrt(log_var)
        # variance depressor
        logvar_dep = K.exp(log_var) - K.ones_like(log_var)
        # undistorted loss
        undistorted_loss = K.categorical_crossentropy(mean, true, from_logits=True)
        # apply montecarlo simulation
        # T = 20
        dist = distributions.Normal(loc=K.zeros_like(log_std), scale=log_std)
        monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, mean, dist,
                                                                         undistorted_loss, D), iterable,
                                       name='monte_carlo_results')

        var_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

        return var_loss + undistorted_loss + K.sum(logvar_dep, -1)

    def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
        def map_fn(i):
            std_samples = dist.sample(1)
            distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
            diff = undistorted_loss - distorted_loss
            return -K.elu(diff)
        return map_fn

    model.compile(optimizer='Adam',
                  # loss=heteroscedastic_categorical_crossentropy)
                  loss={'logits_variance': heteroscedastic_categorical_crossentropy,
                        'softmax_output': 'categorical_crossentropy'},
                  metrics={'softmax_output': 'categorical_accuracy'},
                  loss_weights={'logits_variance': .2, 'softmax_output': 1.}
                  )
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='softmax_output_categorical_accuracy',
                                                  min_delta=0.0001, patience=10)]
    hist = model.fit(X_train,
                     {'logits_variance': Y_train, 'softmax_output': Y_train},
                     # epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.3)
                     epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    loss = hist.history['loss'][-1]
    return model#, -0.5 * loss  # return ELBO up to const.


# ======================================================================================================================

# Take a mean of the results of a TimeDistributed layer.
# Applying TimeDistributedMean()(TimeDistributed(T)(x)) to an
# input of shape (None, ...) returns putpur of same size.
class TimeDistributedMean(Layer):
    def build(self, input_shape):
        super(TimeDistributedMean, self).build(input_shape)

    # input shape (None, T, ...)
    # output shape (None, ...)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]

    def call(self, x):
        return K.mean(x, axis=1)

# Apply the predictive entropy function for input with C classes.
# Input of shape (None, C, ...) returns output with shape (None, ...)
# Input should be predictive means for the C classes.
# In the case of a single classification, output will be (None,).
class PredictiveEntropy(Layer):
    def build(self, input_shape):
        super(PredictiveEntropy, self).build(input_shape)

    # input shape (None, C, ...)
    # output shape (None, ...)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)

    # x - prediction probability for each class(C)
    def call(self, x):
        return -1 * K.sum(K.log(x) * x, axis=1)

def load_epistemic_uncertainty_model(model, epistemic_monte_carlo_simulations):
    # model = load_bayesian_model(checkpoint)
    inpt = Input(shape=(model.input_shape[1:]))
    x = RepeatVector(epistemic_monte_carlo_simulations)(inpt)
    # Keras TimeDistributed can only handle a single output from a model :(
    # and we technically only need the softmax outputs.
    hacked_model = Model(inputs=model.inputs, outputs=model.outputs[1])
    x = TimeDistributed(hacked_model, name='epistemic_monte_carlo')(x)
    # predictive probabilties for each class
    softmax_mean = TimeDistributedMean(name='epistemic_softmax_mean')(x)
    variance = PredictiveEntropy(name='epistemic_variance')(softmax_mean)
    epistemic_model = Model(inputs=inpt, outputs=[variance, softmax_mean])

    return epistemic_model

# ======================================================================================================================
# using landsat data
from CPR.utils import preprocessing
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append('../')
from CPR.configs import data_path

img = '4469_LC08_015036_20170502_1'
pctl = 50
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

epochs = 1
Q = 2  # num of features
D = 2  # num of target classes (output dim)
batch_size = 8192
l = 1e-4
T = 50  # Number of MC passes
dropout_rate = 0.2

data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, feat_list_new, test=False)
perm_index = feat_keep.index('GSW_perm')
flood_index = feat_keep.index('flooded')
data_vector_train[data_vector_train[:, perm_index] == 1, flood_index] = 0
data_vector_train = np.delete(data_vector_train, perm_index, axis=1)
shape = data_vector_train.shape
X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]
y_train = to_categorical(y_train)
INPUT_DIMS = X_train.shape[1]

from tensorflow.keras.utils import plot_model

os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['CONDA_PREFIX'] + r"\Library\bin\graphviz"
print('training')
model = get_aleatoric_uncertainty_model(epochs, X_train, y_train, input_shape=X_train.shape, T=T, D=D)
plot_model(model, to_file='graph.png', show_shapes=True)


# ======================================================================================================================
# Testing with cloud gaps
data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new, test=False)
perm_index = feat_keep.index('GSW_perm')
flood_index = feat_keep.index('flooded')
data_vector_test[
    data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove perm water column
shape = data_vector_test.shape
X_test, y_test = data_vector_test[:, 0:shape[1] - 1], data_vector_test[:, shape[1] - 1]
y_test = to_categorical(y_test)

aleatoric_results = model.predict(X_test, verbose=1)
aleatoric_uncertainties = np.reshape(aleatoric_results[0][:, D:], (-1))
logits = aleatoric_results[0][:, 0:D]
aleatoric_softmax = aleatoric_results[1]

# Epistemic uncertainty
epistemic_model = load_epistemic_uncertainty_model(model, 20)
epistemic_results = epistemic_model.predict(X_test, verbose=1)
epistemic_uncertainties = epistemic_results[0]
epistemic_softmax = epistemic_results[1]

from sklearn.metrics import accuracy_score
accuracy_score(y_true=y_test[:, 1], y_pred=np.argmax(aleatoric_softmax, axis=1))

# Plot the uncertainties
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
myDpi = 300
plt.subplots_adjust(top=0.98, bottom=0.055, left=0.024, right=0.976, hspace=0.2, wspace=0.2)

stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

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
tp = ma.masked_array(tp, mask=np.isnan(prediction_img))
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

# ----------------------------------------------------------------
# Plot epistemic uncertainty
fig, ax = plt.subplots()
fig_img = ax.imshow(epistemic_img, cmap='plasma')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
im_ratio = epistemic_img.shape[0] / epistemic_img.shape[1]
fig.colorbar(fig_img, ax=ax, fraction=0.046*im_ratio, pad=0.04*im_ratio)

aleatoric_img *= 1/np.nanmax(aleatoric_img)
epistemic_img *= 1/np.nanmax(epistemic_img)

uncertainty_img = aleatoric_img + epistemic_img
fig, ax = plt.subplots()
fig_img = ax.imshow(uncertainty_img, cmap='plasma')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
im_ratio = uncertainty_img.shape[0] / uncertainty_img.shape[1]
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

# print('Saving CIR image')
# cir_file = plot_path / '{}'.format('cir_img' + '.png')
# cir_img.save(cir_file, dpi=(300, 300))

import matplotlib.pyplot as plt
import seaborn as sns
plt.imshow(np.reshape(epistemic_uncertainties[0], (50, 60)))
# Test out uncertainties
plt.figure()
sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=epistemic_uncertainties[0])
plt.figure()
sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=aleatoric_uncertainties)
total_uncertainty = epistemic_uncertainties[0] + aleatoric_uncertainties[0]

from sklearn.metrics import accuracy_score
accuracy_score(y_true=X_test[:, 1], y_pred=np.argmax(logits, axis=1))


# print('predicting')
# MC_samples = []
# for k in range(K_test):
#     print('predicting MC_sample: ', k)
#     MC_samples.append(model.predict(X_val))
# MC_samples = np.array(MC_samples)
#
# results = []
# # get results for multiple N
# for N, nb_epoch in zip(Ns, nb_epochs):
#     # repeat exp multiple times
#     rep_results = []
#     for i in range(nb_reps):
#         # X, Y = gen_data(N + nb_val_size)
#         X_train, Y_train = X[:N], Y[:N]
#         X_val, Y_val = X[N:], Y[N:]
#         model, ELBO = fit_model(nb_epoch, X_train, Y_train, input_shape=X_train.shape, T=T, D=D)
#         print('predicting')
#         # MC_samples = np.array([model.predict(X_val) for _ in range(K_test)])
#         MC_samples = []
#         for k in range(K_test):
#             print('predicting MC_sample: ', k)
#             MC_samples.append(model.predict(X_val))
#         MC_samples = np.array(MC_samples)
#         print('testing')
#         pppp, rmse = test(Y_val, MC_samples)  # per point predictive probability
#         means = MC_samples[:, :, :D]  # K x N
#         epistemic_uncertainty = np.var(means, 0).mean(0)
#         logvar = np.mean(MC_samples[:, :, D:], 0)
#         aleatoric_uncertainty = np.exp(logvar).mean(0)
#         ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
#         # plot(X_train, Y_train, X_val, Y_val, means)
#         rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty)]
#     test_mean = np.mean([r[0] for r in rep_results])
#     test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
#     ps = np.mean([r[1] for r in rep_results], 0)
#     aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
#     epistemic_uncertainty = np.mean([r[3] for r in rep_results])
#     print(N, nb_epoch, '-', test_mean, test_std_err, ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)
#     sys.stdout.flush()
#     results += [rep_results]
