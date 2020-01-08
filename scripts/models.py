import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.models

# # ==================================================================================

# Using Functional API
# def get_NN_MCD():
#     tf.keras.backend.clear_session()
#     inputs = tf.keras.layers.Input(shape=(INPUT_DIMS))
#     x = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(inputs, training=True)
#     x = tf.keras.layers.Dense(units=29, activation='relu')(x)
#     x = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(x, training=True)  
#     x = tf.keras.layers.Dense(units=15, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',      
# #                   metrics=[tf.keras.metrics.Recall()])
# #                   metrics=[tf.keras.metrics.F1()])
#                   metrics=['sparse_categorical_accuracy'])
#     return model

# Using Sequential API

# Custom metric functions

# ==================================================================================
# 2 dense layer NN with dropout before ReLU layers

def get_nn_mcd1(INPUT_DIMS, DROPOUT_RATE):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS, name="Input")),
    model.add(tf.keras.layers.Lambda(lambda x: K.dropout(x, level=DROPOUT_RATE))),
    model.add(tf.keras.layers.Dense(units=12, name="Dense1")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm1")),
    model.add(tf.keras.layers.Lambda(lambda x: K.dropout(x, level=DROPOUT_RATE))),
    model.add(tf.keras.layers.Dense(units=12, name="Dense2")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm2")),
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

# ==================================================================================
# 2 dense layer NN with dropout before ReLU layers, and L2 regularization before batch normalization.
# L2 is used because it is a coefficient in Gal's inverse precision (tau) during MCD variance estimation


def get_nn_mcd2(INPUT_DIMS, DROPOUT_RATE, weight_decay=0.005):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS, name="Input")),
    model.add(tf.keras.layers.Lambda(lambda x: K.dropout(x, level=DROPOUT_RATE))),
    model.add(tf.keras.layers.Dense(units=12, name="Dense1",
                                    activity_regularizer=tf.keras.regularizers.l2(weight_decay))),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm1")),
    model.add(tf.keras.layers.Lambda(lambda x: K.dropout(x, level=DROPOUT_RATE))),
    model.add(tf.keras.layers.Dense(units=12, name="Dense2",
                                    activity_regularizer=tf.keras.regularizers.l2(weight_decay))),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm2")),
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

# # ==================================================================================

def get_nn1(INPUT_DIMS):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS)),
    model.add(tf.keras.layers.Dense(units=24,
                                    activation='relu')),
    model.add(tf.keras.layers.Dense(units=12,
                                    activation='relu')),
#     model.add(tf.keras.layers.Dropout(rate=dropout_rate)(training=True)),
#     model.add(tf.keras.layers.Flatten()),
    model.add(tf.keras.layers.Dense(2,
                                    activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
#                   metrics=[tf.keras.metrics.Recall()])
#                   metrics=[tf.keras.metrics.F1()])
                  metrics=['sparse_categorical_accuracy'])
    return model

# ==================================================================================
# NN with batch normalization - OOPS this model has activation after input layer and only one dense layer!
def get_nn_bn(INPUT_DIMS):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS, name="Input")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm1")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense1")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm2")),
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  # optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

# ==================================================================================
# NN with batch normalization, now with two dense layers
def get_nn_bn2(INPUT_DIMS):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS, name="Input")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense1")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm1")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense2")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm2")),
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  # optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

# ==================================================================================
# NN with batch normalization, with 3 dense layers
def get_nn_bn3(INPUT_DIMS):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS, name="Input")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense1")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm1")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense2")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm2")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense3")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.BatchNormalization(name="BatchNorm3")),
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  # optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

# ======================================================================================================================
# Concrete dropout with aleatoric and epistemic uncertainty
from tensorflow.keras.layers import Input, Dense, Dropout, \
    concatenate, BatchNormalization, Activation
from tensorflow_probability import distributions
from tensorflow.keras import models
import tensorflow.keras.backend as K
import numpy as np


def get_aleatoric_uncertainty_model(epochs, X_train, Y_train, input_shape, T, D, batch_size, dropout_rate, callbacks):
    # optimizer = tf.keras.optimizers.Adam()
    tf.keras.backend.clear_session()
    inp = Input(shape=input_shape[1:])
    x = inp
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x, training=True)
    x = Dense(24, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x, training=True)
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
    hist = model.fit(X_train,
                     {'logits_variance': Y_train, 'softmax_output': Y_train},
                     # epochs=nb_epoch, batch_size=batch_size, verbose=1, validation_split=0.3)
                     epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    loss = hist.history['loss'][-1]
    return model#, -0.5 * loss  # return ELBO up to const.

# -----------------------------------------------------------
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Layer
from tensorflow.keras import Model

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

def get_epistemic_uncertainty_model(model, epistemic_monte_carlo_simulations):
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

