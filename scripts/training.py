import tensorflow as tf
import pandas as pd
import time
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import time
import os
# Import custom functions
import sys

sys.path.append('../')
# from CPR.configs import data_path
from CPR.utils import tif_stacker, cloud_generator, preprocessing, train_val, timer


# from models import get_nn_uncertainty1 as get_model
# from models import get_nn1 as model_func


# ==================================================================================

def training1(img_list, pctls, model_func, feat_list_new, uncertainty, data_path, batch,
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
                model_path = data_path / batch / 'models' / 'nn_mcd' / img
                metrics_path = data_path / batch / 'metrics' / 'training_nn_mcd' / img / '{}'.format(
                    img + '_clouds_' + str(pctl))
            else:
                model_path = data_path / batch / 'models' / 'nn' / img
                metrics_path = data_path / batch / 'metrics' / 'training_nn' / img / '{}'.format(
                    img + '_clouds_' + str(pctl))

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

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times_df = pd.DataFrame(times, columns=['times'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)


# ============================================================================================


def training2(img_list, pctls, model_func, feat_list_new, uncertainty, data_path, batch,
              DROPOUT_RATE=0, HOLDOUT=0.3, **model_params):
    '''
    Removes flood water that is permanent water
    '''

    get_model = model_func

    for j, img in enumerate(img_list):
        times = []
        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=True)
        cloud_generator(img, data_path, overwrite=False)

        for i, pctl in enumerate(pctls):
            data_train, data_vector_train, data_ind_train = preprocessing(data_path, img, pctl, gaps=False)
            perm_index = feat_list_new.index('GSW_perm')
            flood_index = feat_list_new.index('flooded')
            data_vector_train[
                data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column

            training_data, validation_data = train_val(data_vector_train, holdout=HOLDOUT)
            X_train, y_train = training_data[:, 0:14], training_data[:, 14]
            X_val, y_val = validation_data[:, 0:14], validation_data[:, 14]
            INPUT_DIMS = X_train.shape[1]

            if uncertainty:
                model_path = data_path / batch / 'models' / 'nn_mcd' / img
                metrics_path = data_path / batch / 'metrics' / 'training_nn_mcd' / img / '{}'.format(
                    img + '_clouds_' + str(pctl))
            else:
                model_path = data_path / batch / 'models' / 'nn' / img
                metrics_path = data_path / batch / 'metrics' / 'training_nn' / img / '{}'.format(
                    img + '_clouds_' + str(pctl))

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

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([pctls, times])
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)


# ============================================================================================
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


def lr_plots(lrRangeFinder, lr_plots_path, img, pctl):
    # Make plots of learning rate vs. loss
    plt.ioff()

    # LR vs. loss
    plt.plot(lrRangeFinder.lrs, lrRangeFinder.losses)
    plt.title('Model Losses Batch after Batch')
    plt.ylabel('loss')
    plt.xlabel('learning rate')
    plt.savefig(lr_plots_path / '{}'.format(img + '_clouds_' + str(pctl) + '_smooth.png'))

    # LR vs. loss (smooth)
    smoothed_losses = smooth(lrRangeFinder.losses, 20)
    plt.figure()
    plt.plot(lrRangeFinder.lrs, smoothed_losses)
    plt.title('Smoothed Model Losses Batch after Batch')
    plt.ylabel('loss')
    plt.xlabel('learning rate')
    plt.savefig(lr_plots_path / '{}'.format(img + '_clouds_' + str(pctl) + '_smooth.png'))

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
    plt.savefig(lr_plots_path / '{}'.format(img + '_clouds_' + str(pctl) + '_lrRange.png'))
    plt.close('all')
    return lr_min, lr_max, lrRangeFinder.lrs, lrRangeFinder.losses


def training3(img_list, pctls, model_func, feat_list_new, uncertainty, data_path, batch,
              DROPOUT_RATE=None, HOLDOUT=0.2, **model_params):
    '''
    1. Removes flood water that is permanent water
    2. Finds the optimum learning rate and uses cyclic LR scheduler
    to train the model
    '''
    get_model = model_func
    for j, img in enumerate(img_list):
        print(img + ': stacking tif, generating clouds')
        times = []
        lr_mins = []
        lr_maxes = []
        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
        cloud_generator(img, data_path, overwrite=False)

        for i, pctl in enumerate(pctls):
            print(img, pctl, '% CLOUD COVER')
            print('Preprocessing')
            data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, gaps=False)
            feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
            perm_index = feat_list_keep.index('GSW_perm')
            flood_index = feat_list_keep.index('flooded')
            data_vector_train[data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
            training_data, validation_data = train_val(data_vector_train, holdout=HOLDOUT)
            shape = data_vector_train.shape
            X_train, y_train = training_data[:, 0:shape[1]-1], training_data[:, shape[1]-1]
            X_val, y_val = validation_data[:, 0:shape[1]-1], validation_data[:, shape[1]-1]
            INPUT_DIMS = X_train.shape[1]

            model_path = data_path / batch / 'models' / 'nn' / img
            metrics_path = data_path / batch / 'metrics' / 'training_nn' / img / '{}'.format(
                img + '_clouds_' + str(pctl))

            lr_plots_path = metrics_path.parents[1] / 'lr_plots'
            lr_vals_path = metrics_path.parents[1] / 'lr_vals'
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

            if uncertainty:
                model = model_func(INPUT_DIMS, DROPOUT_RATE)
            else:
                model = model_func(INPUT_DIMS)

            print('Finding learning rate')
            model.fit(X_train, y_train, **lr_model_params, validation_data=(X_val, y_val))
            lr_min, lr_max, lr, losses = lr_plots(lrRangeFinder, lr_plots_path, img, pctl)
            lr_mins.append(lr_min)
            lr_maxes.append(lr_max)
            # ---------------------------------------------------------------------------------------------------
            # Training the model with cyclical learning rate scheduler
            model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
            scheduler = SGDRScheduler(min_lr=lr_min, max_lr=lr_max, lr_decay=0.9, cycle_length=3, mult_factor=1.5)

            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),
                         tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor='val_loss',
                                                            save_best_only=True),
                         CSVLogger(metrics_path / 'training_log.log'),
                         scheduler]

            if uncertainty:
                model = get_model(INPUT_DIMS, DROPOUT_RATE)  # Model with uncertainty
            else:
                model = get_model(INPUT_DIMS)  # Model without uncertainty

            print('Training full model with best LR')
            start_time = time.time()
            model.fit(X_train, y_train, **model_params, validation_data=(X_val, y_val), callbacks=callbacks)
            end_time = time.time()
            times.append(timer(start_time, end_time, False))
            # model.save(model_path)

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([pctls, times])
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)

        lr_range = np.column_stack([pctls, lr_mins, lr_maxes])
        lr_avg = np.mean(lr_range[:, 1:2], axis=1)
        lr_range = np.column_stack([lr_range, lr_avg])
        lr_range_df = pd.DataFrame(lr_range, columns=['cloud_cover', 'lr_min', 'lr_max', 'lr_avg'])
        lr_range_df.to_csv((lr_vals_path / img).with_suffix('.csv'), index=False)

        losses_path = lr_vals_path / img / '{}'.format('losses_'+str(pctl)+'.csv')
        try:
            losses_path.parent.mkdir(parents=True)
        except FileExistsError:
            pass
        lr_losses = np.column_stack([lr, losses])
        lr_losses = pd.DataFrame(lr_losses, columns=['lr', 'losses'])
        lr_losses.to_csv(losses_path, index=False)

# ============================================================================================
def training4(img_list, pctls, model_func, feat_list_new, uncertainty, data_path, batch,
              DROPOUT_RATE=0, **model_params):
    '''
    1. Removes flood water that is permanent water
    2. Finds the optimum learning rate and uses cyclic LR scheduler
    to train the model
    3. No validation set for training
    '''
    get_model = model_func
    for j, img in enumerate(img_list):
        print(img + ': stacking tif, generating clouds')
        times = []
        lr_mins = []
        lr_maxes = []
        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=False)
        cloud_generator(img, data_path, overwrite=False)

        for i, pctl in enumerate(pctls):
            print(img, pctl, '% CLOUD COVER')
            print('Preprocessing')
            tf.keras.backend.clear_session()
            data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, gaps=False)
            feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
            perm_index = feat_list_keep.index('GSW_perm')
            flood_index = feat_list_keep.index('flooded')
            data_vector_train[
                data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
            shape = data_vector_train.shape
            X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]
            INPUT_DIMS = X_train.shape[1]

            model_path = data_path / batch / 'models' / 'nn' / img
            metrics_path = data_path / batch / 'metrics' / 'training_nn' / img / '{}'.format(
                img + '_clouds_' + str(pctl))

            lr_plots_path = metrics_path.parents[1] / 'lr_plots'
            lr_vals_path = metrics_path.parents[1] / 'lr_vals'
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

            if uncertainty:
                model = model_func(INPUT_DIMS, DROPOUT_RATE)
            else:
                model = model_func(INPUT_DIMS)

            print('Finding learning rate')
            model.fit(X_train, y_train, **lr_model_params)
            lr_min, lr_max, lr, losses = lr_plots(lrRangeFinder, lr_plots_path, img, pctl)
            lr_mins.append(lr_min)
            lr_maxes.append(lr_max)
            # ---------------------------------------------------------------------------------------------------
            # Training the model with cyclical learning rate scheduler
            model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
            scheduler = SGDRScheduler(min_lr=lr_min, max_lr=lr_max, lr_decay=0.9, cycle_length=3, mult_factor=1.5)

            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=10),
                         tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor='loss',
                                                            save_best_only=True),
                         CSVLogger(metrics_path / 'training_log.log'),
                         scheduler]

            if uncertainty:
                model = get_model(INPUT_DIMS, DROPOUT_RATE)  # Model with uncertainty
            else:
                model = get_model(INPUT_DIMS)  # Model without uncertainty

            print('Training full model with best LR')
            start_time = time.time()
            model.fit(X_train, y_train, **model_params, callbacks=callbacks)
            end_time = time.time()
            times.append(timer(start_time, end_time, False))
            # model.save(model_path)

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([pctls, times])
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)

        lr_range = np.column_stack([pctls, lr_mins, lr_maxes])
        lr_avg = np.mean(lr_range[:, 1:2], axis=1)
        lr_range = np.column_stack([lr_range, lr_avg])
        lr_range_df = pd.DataFrame(lr_range, columns=['cloud_cover', 'lr_min', 'lr_max', 'lr_avg'])
        lr_range_df.to_csv((lr_vals_path / img).with_suffix('.csv'), index=False)

        losses_path = lr_vals_path / img / '{}'.format('losses_' + str(pctl) + '.csv')
        try:
            losses_path.parent.mkdir(parents=True)
        except FileExistsError:
            pass
        lr_losses = np.column_stack([lr, losses])
        lr_losses = pd.DataFrame(lr_losses, columns=['lr', 'losses'])
        lr_losses.to_csv(losses_path, index=False)
