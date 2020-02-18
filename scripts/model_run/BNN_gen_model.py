import __init__
from models import get_nn_bn2_kwon_v2 as model_func
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import h5py
import pandas as pd
import numpy as np
import random
from CPR.utils import timer, preprocessing_gen_model, preprocessing
import time
import os
from results_viz import VizFuncs
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Parameters

batch = 'BNN_gen_model'
pctls = [10, 30, 50, 70, 90]
batch_size = 8192
epochs = 50
MC_passes = 50
dropout_rate = 0.2

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
img_list.remove('4115_LC08_021033_20131227_test')
img_list = img_list[0:3]
pctls = [50]
# random.seed(32)
# random.shuffle(img_list)

num_train = np.floor(len(img_list) * (2/3)).astype('int')
img_list_train = img_list[0:num_train]
img_list_test = img_list[num_train:len(img_list)]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

model_params = {'epochs': epochs,
                'batch_size': batch_size}

viz_params = {'img_list': img_list_test,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}

NUM_PARALLEL_EXEC_UNITS = 8
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=4,
                                  allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ['MKL_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ['GOTO_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)

os.environ['MKL_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ['GOTO_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)

# ======================================================================================================================

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
    """
    Cosine annealing learning rate scheduler with periodic restarts.
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
    """
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


def smooth(y, box_pts):
    """smoothes an array by taking the average of the `box_pts` point around each point"""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def lr_plots(lrRangeFinder, lr_plots_path):
    # Make plots of learning rate vs. loss
    plt.ioff()

    # LR vs. loss (smooth)
    smoothed_losses = smooth(lrRangeFinder.losses, len(lrRangeFinder.lrs))
    plt.figure()
    plt.plot(lrRangeFinder.lrs, smoothed_losses)
    plt.title('Smoothed Model Losses Batch after Batch')
    plt.ylabel('loss')
    plt.xlabel('learning rate')
    plt.savefig(lr_plots_path / 'lr_loss_smooth.png')

    # Find LR range
    min_ = np.argmin(smoothed_losses)
    max_ = np.argmax(smoothed_losses)

    smoothed_losses_ = smoothed_losses[min_: max_]
    smoothed_diffs = smooth(np.diff(smoothed_losses), len(np.diff(smoothed_losses)))
    plt.figure()
    plt.plot(lrRangeFinder.lrs[:-1], smoothed_diffs)
    plt.title('Differences')
    plt.ylabel('loss difference')
    plt.xlabel('learning rate')

    min_ = np.argmax(smoothed_diffs <= 0)  # where the (smoothed) loss starts to decrease
    max_ = np.argmax(smoothed_diffs >= 0)  # where the (smoothed) loss restarts to increase
    max_ = max_ if max_ > 0 else smoothed_diffs.shape[0]  # because max_ == 0 if it never restarts to increase


    smoothed_losses_ = smoothed_losses[min_: max_]  # restrain the window to the min_, max_ interval
    # Take min and max loss in this restrained window
    try:
        min_smoothed_loss_ = min(smoothed_losses_[:-1])
        max_smoothed_loss_ = max(smoothed_losses_[:-1])
        delta = max_smoothed_loss_ - min_smoothed_loss_

        lr_arg_max = np.argmax(smoothed_losses_ <= min_smoothed_loss_ + .05 * delta)
        lr_arg_min = np.argmax(smoothed_losses_ <= min_smoothed_loss_ + .5 * delta)

        lr_arg_min += min_
        lr_arg_max += min_
        lrs = lrRangeFinder.lrs[lr_arg_min: lr_arg_max]
        lr_min, lr_max = min(lrs), max(lrs)
    except ValueError:
        lr_min, lr_max = 0.8, 1.5


    print('lr range: [{}, {}]'.format(lr_min, lr_max))

    plt.figure()
    ax = plt.axes()
    ax.plot(lrRangeFinder.lrs, smoothed_losses)
    ax.set_title('Smoothed Model Losses Batch after Batch')
    ax.set_ylabel('loss')
    ax.set_xlabel('learning rate')
    ax.set_ylim(0, 1.05 * np.max(smoothed_losses))
    ax.set_xlim(0, max(lrRangeFinder.lrs))
    ax.vlines(x=[lr_min, lr_max], ymin=[0, 0], ymax=[smoothed_losses[lr_arg_min], smoothed_losses[lr_arg_max]],
              color='r', linestyle='--', linewidth=.8)
    ax.plot(lrs, smoothed_losses[lr_arg_min: lr_arg_max], linewidth=2)
    x_arrow_arg = int((lr_arg_min + lr_arg_max) / 2)
    x_arrow = lrRangeFinder.lrs[x_arrow_arg]
    y_arrow = smoothed_losses[x_arrow_arg]
    ax.annotate('best piece of slope', xy=(x_arrow, y_arrow), xytext=(lr_max, smoothed_losses[lr_arg_min]),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', (lr_min, smoothed_losses[lr_arg_max] / 5), (lr_max, smoothed_losses[lr_arg_max] / 5),
                arrowprops={'arrowstyle': '<->'})
    ax.text((lr_min + lr_max) / 2, 3 * smoothed_losses[lr_arg_max] / 5, 'lr range',
            horizontalalignment='center', verticalalignment='center', weight='bold')
    plt.savefig(lr_plots_path / '_lrRange.png')
    plt.close('all')
    return lr_min, lr_max, lrRangeFinder.lrs, lrRangeFinder.losses


def training_BNN_gen_model(img_list_train, feat_list_new, model_func, data_path, batch, dropout_rate, **model_params):
    get_model = model_func
    times = []
    lr_mins = []
    lr_maxes = []

    print('Preprocessing')
    tf.keras.backend.clear_session()
    data_vector_train = preprocessing_gen_model(data_path, img_list_train)
    perm_index = feat_list_new.index('GSW_perm')
    flood_index = feat_list_new.index('flooded')
    print(data_vector_train.shape)
    data_vector_train[data_vector_train[:, perm_index] == 1, flood_index] = 0
    data_vector_train = np.delete(data_vector_train, perm_index, axis=1)
    shape = data_vector_train.shape
    X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]
    input_dims = X_train.shape[1]

    model_path = data_path / batch / 'models'
    metrics_path = data_path / batch / 'metrics' / 'training'

    lr_plots_path = metrics_path / 'lr_plots'
    lr_vals_path = metrics_path/ 'lr_vals'
    try:
        metrics_path.mkdir(parents=True)
        model_path.mkdir(parents=True)
        lr_plots_path.mkdir(parents=True)
        lr_vals_path.mkdir(parents=True)
    except FileExistsError:
        pass

    # ---------------------------------------------------------------------------------------------------
    # Determine learning rate by finding max loss decrease during single epoch training
    lrRangeFinder = LrRangeFinder(start_lr=0.1, end_lr=2)

    lr_model_params = {'batch_size': model_params['batch_size'],
                       'epochs': 1,
                       'verbose': 2,
                       'callbacks': [lrRangeFinder],
                       'use_multiprocessing': True}

    model = model_func(input_dims, dropout_rate)

    print('Finding learning rate')
    model.fit(X_train, y_train, **lr_model_params)
    lr_min, lr_max, lr, losses = lr_plots(lrRangeFinder, lr_plots_path)
    lr_mins.append(lr_min)
    lr_maxes.append(lr_max)
    # ---------------------------------------------------------------------------------------------------
    # Training the model with cyclical learning rate scheduler
    model_path = model_path / 'gen_model.h5'
    scheduler = SGDRScheduler(min_lr=lr_min, max_lr=lr_max, lr_decay=0.9, cycle_length=3, mult_factor=1.5)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0.001, patience=10),
                 tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor='loss',
                                                    save_best_only=True),
                 CSVLogger(metrics_path / 'training_log.log'),
                 scheduler]

    model = get_model(input_dims, dropout_rate)

    print('Training full model with best LR')
    start_time = time.time()
    model.fit(X_train, y_train, **model_params, callbacks=callbacks)
    end_time = time.time()
    times.append(timer(start_time, end_time, False))

    metrics_path = metrics_path.parent
    times = [float(i) for i in times]
    times_df = pd.DataFrame(times, columns=['training_time'])
    times_df.to_csv(metrics_path / 'training_times.csv', index=False)

    lr_range = np.column_stack([lr_mins, lr_maxes])
    lr_avg = np.mean(lr_range, axis=1)
    lr_range = np.column_stack([lr_range, lr_avg])
    lr_range_df = pd.DataFrame(lr_range, columns=['lr_min', 'lr_max', 'lr_avg'])
    lr_range_df.to_csv((lr_vals_path).with_suffix('.csv'), index=False)

    losses_path = lr_vals_path / 'gen_model_losses.csv'
    try:
        losses_path.parent.mkdir(parents=True)
    except FileExistsError:
        pass
    lr_losses = np.column_stack([lr, losses])
    lr_losses = pd.DataFrame(lr_losses, columns=['lr', 'losses'])
    lr_losses.to_csv(losses_path, index=False)


def prediction_BNN_gen_model(img_list, pctls, feat_list_new, data_path, batch, MC_passes, **model_params):
    for j, img in enumerate(img_list):
        times = []
        accuracy, precision, recall, f1 = [], [], [], []
        preds_path = data_path / batch / 'predictions' / img
        bin_file = preds_path / 'predictions.h5'
        model_path = data_path / batch / 'models' / 'gen_model.h5'

        uncertainties_path = data_path / batch / 'uncertainties' / img
        aleatoric_bin_file = uncertainties_path / 'aleatoric_uncertainties.h5'
        epistemic_bin_file = uncertainties_path / 'epistemic_uncertainties.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for i, pctl in enumerate(pctls):
            print('Preprocessing', img, pctl, '% cloud cover')
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                                  test=True)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[
                data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
            data_shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:data_shape[1] - 1], data_vector_test[:, data_shape[1] - 1]

            print('Predicting for {} at {}% cloud cover'.format(img, pctl))
            start_time = time.time()
            model = tf.keras.models.load_model(model_path)
            p_hat = []
            for t in range(MC_passes):
                p_hat.append(
                    model.predict(X_test, batch_size=model_params['batch_size'], use_multiprocessing=True)[:, 1])
            p_hat = np.array(p_hat)
            preds = np.round(np.mean(p_hat, axis=0))
            aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
            epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

            try:
                preds_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier mean predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=preds)

            try:
                uncertainties_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(epistemic_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier epistemic predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=epistemic)

            with h5py.File(aleatoric_bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier epistemic predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=aleatoric)

            times.append(timer(start_time, time.time(), False))

            print('Evaluating predictions')
            accuracy.append(accuracy_score(y_test, preds))
            precision.append(precision_score(y_test, preds))
            recall.append(recall_score(y_test, preds))
            f1.append(f1_score(y_test, preds))

            del preds, p_hat, aleatoric, epistemic, X_test, y_test, model, data_test, data_vector_test, data_ind_test

        metrics = pd.DataFrame(np.column_stack([pctls, accuracy, precision, recall, f1]),
                               columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        times = [float(i) for i in times]
        times_df = pd.DataFrame(np.column_stack([pctls, times]),
                                columns=['cloud_cover', 'testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)


# ======================================================================================================================
training_BNN_gen_model(img_list_train, feat_list_new, model_func, data_path, batch, dropout_rate, **model_params)
prediction_BNN_gen_model(img_list_test, pctls, feat_list_new, data_path, batch, MC_passes, **model_params)
viz = VizFuncs(viz_params)
viz.metric_plots()
viz.metric_plots_multi()
viz.time_plot()
viz.false_map(probs=False, save=False)
viz.false_map_borders()
viz.fpfn_map()
viz.uncertainty_map_NN()
