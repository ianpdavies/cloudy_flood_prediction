from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, BatchNormalization, Activation, RepeatVector, \
    TimeDistributed, Layer
from tensorflow_probability import distributions
from tensorflow.keras import Model, models
import tensorflow.keras.backend as K
import numpy as np

1
def bayesian_categorical_crossentropy(T, D, iterable):
    def bayesian_categorical_crossentropy_internal(true, pred_var):
        # shape: (N,)
        std = K.sqrt(pred_var[:, D:])
        # shape: (N,)
        variance = pred_var[:, D]
        variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:D]
        # shape: (N,)
        undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
        # shape: (T,)
        # iterable = K.variable(np.ones(T))
        dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
        monte_carlo_results = K.map_fn(
            gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, D), iterable,
            name='monte_carlo_results')
        variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_categorical_crossentropy_internal


def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
    def map_fn(i):
        std_samples = dist.sample(1)
        distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -K.elu(diff)

    return map_fn
