from models import get_nn_bn as model_func
import tensorflow as tf
import os
from training import training1, training2
from prediction import prediction
from evaluation import evaluation
from results_viz import VizFuncs
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Learning rate finder
# From here: https://mancap314.github.io/cyclical-learning-rates-with-tensorflow-implementation.html
# ipynb here: https://github.com/mancap314/miscellanous/blob/master/lr_optimization.ipynb

# LrRangeFinder increases the LR from start_lr to end_lr
class LrRangeFinder(tf.keras.callbacks.Callback):
    def __init__(self, start_lr, end_lr):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr

    def on_train_begin(self, logs={}):
        self.lrs = []
        self.losses = []
        tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

        n_steps = self.params['steps'] if self.params['steps'] is not None else round(
            self.params['samples'] / self.params['batch_size'])
        n_steps *= self.params['epochs']
        self.by = (self.end_lr - self.start_lr) / n_steps

    def on_batch_end(self, batch, logs={}):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.lrs.append(lr)
        self.losses.append(logs.get('loss'))
        lr += self.by
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

class SGDRScheduler(tf.keras.callbacks.Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        self.steps_per_epoch = self.params['steps'] if self.params['steps'] is not None else round(self.params['samples'] / self.params['batch_size'])
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)

# ==================================================================================
# Parameters

uncertainty = False
batch = 'v3'

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

# img_list = ['4115_LC08_021033_20131227_test']
img_list = ['4115_LC08_021033_20131227_1']
# img_list = ['4101_LC08_027038_20131103_1',
#             '4101_LC08_027038_20131103_2',
#             '4101_LC08_027039_20131103_1',
#             '4115_LC08_021033_20131227_1',
#             '4115_LC08_021033_20131227_2',
#             '4337_LC08_026038_20160325_1',
#             '4444_LC08_043034_20170303_1',
#             '4444_LC08_043035_20170303_1',
#             '4444_LC08_044032_20170222_1',
#             '4444_LC08_044033_20170222_1',
#             '4444_LC08_044033_20170222_2',
#             '4444_LC08_044033_20170222_3',
#             '4444_LC08_044033_20170222_4',
#             '4444_LC08_044034_20170222_1',
#             '4444_LC08_045032_20170301_1',
#             '4468_LC08_022035_20170503_1',
#             '4468_LC08_024036_20170501_1',
#             '4468_LC08_024036_20170501_2',
#             '4469_LC08_015035_20170502_1',
#             '4469_LC08_015036_20170502_1',
#             '4477_LC08_022033_20170519_1',
#             '4514_LC08_027033_20170826_1']


# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
BATCH_SIZE = 10000
EPOCHS = 1
DROPOUT_RATE = 0.3  # Dropout rate for MCD
HOLDOUT = 0.3  # Validation data size
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1)

lrRangeFinder = LrRangeFinder(start_lr=0.1, end_lr=2)

model_params = {'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'verbose': 1,
                'callbacks': [lrRangeFinder, es],
                'use_multiprocessing': True}

# ==================================================================================
# Training
from CPR.utils import preprocessing, train_val
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
img = img_list[0]
pctl = pctls[3]

data_train, data_vector_train, data_ind_train = preprocessing(data_path, img, pctl, gaps=False)
perm_index = feat_list_new.index('GSW_perm')
flood_index = feat_list_new.index('flooded')
data_vector_train[data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column

training_data, validation_data = train_val(data_vector_train, holdout=HOLDOUT)
X_train, y_train = training_data[:, 0:14], training_data[:, 14]
X_val, y_val = validation_data[:, 0:14], validation_data[:, 14]
INPUT_DIMS = X_train.shape[1]

model = model_func(INPUT_DIMS)  # Model without uncertainty
model.fit(X_train, y_train, **model_params, validation_data=(X_val, y_val))

print('{} batches recorded'.format(len(lrRangeFinder.losses)))

# Visualize learning rate vs. loss
plt.plot(lrRangeFinder.lrs, lrRangeFinder.losses)
plt.title('Model Losses Batch after Batch')
plt.ylabel('loss')
plt.xlabel('learning rate')
plt.show()

def smooth(y, box_pts):
  """smoothes an array by taking the average of the `box_pts` point around each point"""
  box = np.ones(box_pts)/box_pts
  y_smooth = np.convolve(y, box, mode='same')
  return y_smooth

smoothed_losses = smooth(lrRangeFinder.losses, 20)
plt.figure()
plt.plot(lrRangeFinder.lrs, smoothed_losses)
plt.title('Smoothed Model Losses Batch after Batch')
plt.ylabel('loss')
plt.xlabel('learning rate')
plt.show()

# Find LR range
min_ = np.argmin(smoothed_losses)
max_ = np.argmax(smoothed_losses)

smoothed_losses_ = smoothed_losses[min_: max_]
smoothed_diffs = smooth(np.diff(smoothed_losses), 20)
plt.figure()
plt.plot(lrRangeFinder.lrs[:-1], smoothed_diffs)
plt.title('Differences')
plt.ylabel('loss difference')
plt.xlabel('learning rate')
plt.show()

min_ = np.argmax(smoothed_diffs <= 0)  # where the (smoothed) loss starts to decrease
max_ = np.argmax(smoothed_diffs >= 0)  # where the (smoothed) loss restarts to increase
max_ = max_ if max_ > 0 else smoothed_diffs.shape[0]  # because max_ == 0 if it never restarts to increase

smoothed_losses_ = smoothed_losses[min_: max_]  # restrain the window to the min_, max_ interval
# Take min and max loss in this restrained window
min_smoothed_loss_ = min(smoothed_losses_[:-1])
max_smoothed_loss_ = max(smoothed_losses_[:-1])
delta = max_smoothed_loss_ - min_smoothed_loss_


lr_arg_max = np.argmax(smoothed_losses_ <= min_smoothed_loss_ + .05 * delta)
lr_arg_min = np.argmax(smoothed_losses_ <= min_smoothed_loss_ + .5 * delta)

lr_arg_min += min_
lr_arg_max += min_

lrs = lrRangeFinder.lrs[lr_arg_min: lr_arg_max]
lr_min, lr_max = min(lrs), max(lrs)

print('lr range: [{}, {}]'.format(lr_min, lr_max))

plt.figure()
ax = plt.axes()
ax.plot(lrRangeFinder.lrs, smoothed_losses)
ax.set_title('Smoothed Model Losses Batch after Batch')
ax.set_ylabel('loss')
ax.set_xlabel('learning rate')
ax.set_ylim(0, 1.05 * np.max(smoothed_losses))
ax.set_xlim(0, max(lrRangeFinder.lrs))
ax.vlines(x=[lr_min, lr_max], ymin=[0, 0], ymax=[smoothed_losses[lr_arg_min], smoothed_losses[lr_arg_max]], color='r', linestyle='--', linewidth=.8)
ax.plot(lrs, smoothed_losses[lr_arg_min: lr_arg_max], linewidth=2)
x_arrow_arg = int((lr_arg_min + lr_arg_max) / 2)
x_arrow = lrRangeFinder.lrs[x_arrow_arg]
y_arrow = smoothed_losses[x_arrow_arg]
ax.annotate('best piece of slope', xy=(x_arrow, y_arrow), xytext=(lr_max, smoothed_losses[lr_arg_min]), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate ('', (lr_min, smoothed_losses[lr_arg_max] / 5), (lr_max, smoothed_losses[lr_arg_max] / 5), arrowprops={'arrowstyle':'<->'})
ax.text((lr_min + lr_max) / 2, 3 * smoothed_losses[lr_arg_max] / 5, 'lr range', horizontalalignment='center', verticalalignment='center', weight='bold')

plt.show()
# ==================================================================================
# Cyclical learning rate scheduler
scheduler = SGDRScheduler(min_lr=lr_min, max_lr=lr_max, lr_decay=.9, cycle_length=3, mult_factor=1.5)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
             tf.keras.callbacks.ModelCheckpoint(filepath='best_model_with_cycling.h5', monitor='val_loss', save_best_only=True),
            scheduler]

epochs = 100

model = model_func(INPUT_DIMS)

model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=BATCH_SIZE, callbacks=callbacks)

# ==================================================================================
# # Prediction
# data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True)
#
# perm_index = feat_list_new.index('GSW_perm')
# flood_index = feat_list_new.index('flooded')
# data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
# data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
# data_shape = data_vector_test.shape
# X_test, y_test = data_vector_test[:, 0:data_shape[1] - 1], data_vector_test[:, data_shape[1] - 1]
# INPUT_DIMS = X_test.shape[1]
#
# print('Predicting for {} at {}% cloud cover'.format(img, pctl))
#
# trained_model = model  # Get untrained model to add trained weights into
# preds = trained_model.predict(X_test, batch_size=BATCH_SIZE, use_multiprocessing=True)
# preds = np.argmax(preds, axis=1)  # Display most probable value
# # ==================================================================================
# # Evaluation
#
# # Metrics including perm water
# print('Evaluating with perm water')
# accuracy = accuracy_score(y_test, preds)
# precision = precision_score(y_test, preds)
# recall = recall_score(y_test, preds)
# f1 = f1_score(y_test, preds)
# print(accuracy, precision, recall, f1)
