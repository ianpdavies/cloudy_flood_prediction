import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.models


# ======================================================================================================================
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


# ======================================================================================================================

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


# ======================================================================================================================
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


# ======================================================================================================================
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

# ======================================================================================================================
# NN with batch normalization, now with two dense layers
from loss_functions import macro_double_soft_f1
def get_nn_bn2_f1(INPUT_DIMS):
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
                  loss=macro_double_soft_f1)
    return model

# ======================================================================================================================
# NN with batch normalization, now with two dense layers
def get_nn_bn2_noBN(INPUT_DIMS):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS, name="Input")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense1")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.Dense(units=12, name="Dense2")),
    model.add(tf.keras.layers.Activation("relu")),
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  # optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

# ======================================================================================================================
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

# Models come from Kyle Dorman: https://github.com/kyle-dorman/bayesian-neural-network-blogpost

from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, BatchNormalization, Activation, RepeatVector, \
    TimeDistributed, Layer
from tensorflow_probability import distributions
from tensorflow.keras import Model, models
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects
from loss_functions import bayesian_categorical_crossentropy
import numpy as np


def load_bayesian_model(checkpoint, T, D, iterable):
    get_custom_objects().update({"bayesian_categorical_crossentropy_internal": bayesian_categorical_crossentropy(T, D, iterable)})
    return tf.keras.models.load_model(checkpoint)


def get_aleatoric_uncertainty_model(X_train, Y_train, epochs, T, D, batch_size, dropout_rate, callbacks):
    tf.keras.backend.clear_session()
    inp = Input(shape=X_train.shape[1:])
    x = inp
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x, training=True)
    x = Dense(24, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x, training=True)

    means = Dense(D, name='means')(x)
    log_var_pre = Dense(1, name='log_var_pre')(x)
    log_var = Activation('softplus', name='log_var')(log_var_pre)  # What does this do?
    logits_variance = concatenate([means, log_var], name='logits_variance')
    softmax_output = Activation('softmax', name='softmax_output')(means)
    model = models.Model(inputs=inp, outputs=[logits_variance, softmax_output])

    iterable = K.variable(np.ones(T))
    model.compile(optimizer='Adam',
                  loss={'logits_variance': bayesian_categorical_crossentropy(T, D, iterable),
                        'softmax_output': 'categorical_crossentropy'},
                  metrics={'softmax_output': 'categorical_accuracy'},
                  loss_weights={'logits_variance': .2, 'softmax_output': 1.})

    model.fit(X_train,
              {'logits_variance': Y_train, 'softmax_output': Y_train}, epochs=epochs,
              batch_size=batch_size, verbose=2, callbacks=callbacks)
    return model


# Take a mean of the results of a TimeDistributed layer.
# Applying TimeDistributedMean()(TimeDistributed(T)(x)) to an
# input of shape (None, ...) returns output of same size.
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


def get_epistemic_uncertainty_model(checkpoint, T, D):
    iterable = K.variable(np.ones(T))
    model = load_bayesian_model(checkpoint, T, D, iterable)
    inpt = Input(shape=(model.input_shape[1:]))
    x = RepeatVector(T)(inpt)
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
# BNN with dropout, but modified uncertainty estimation from Kwon
# https://github.com/ykwon0407/UQ_BNN
# https://gitlab.com/wdeback/dl-keras-tutorial/blob/master/notebooks/3-cnn-segment-retina-uncertainty.ipynb


def get_nn_bn1_kwon(input_dims, dropout_rate):
    tf.keras.backend.clear_session()
    inp = Input(shape=input_dims, name='Input')
    x = inp
    x = Dense(24, activation='relu', name='Dense_1')(x)
    x = BatchNormalization(name='BatchNormalization_1')(x)
    x = Dropout(dropout_rate, name='Dropout_1')(x, training=True)
    output = Dense(2, activation='softmax', name='output')(x)
    model = models.Model(inputs=inp, outputs=output)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    return model


def get_nn_bn2_kwon(input_dims, dropout_rate):
    tf.keras.backend.clear_session()
    inp = Input(shape=input_dims, name='Input')
    x = inp
    x = Dense(24, activation='relu', name='Dense_1')(x)
    x = BatchNormalization(name='BatchNormalization_1')(x)
    x = Dropout(dropout_rate, name='Dropout_1')(x, training=True)
    x = Dense(12, activation='relu', name='Dense_2')(x)
    x = BatchNormalization(name='BatchNormalization_2')(x)
    x = Dropout(dropout_rate, name='Dropout_2')(x, training=True)
    output = Dense(2, activation='softmax', name='output')(x)
    model = models.Model(inputs=inp, outputs=output)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    return model


def get_nn_bn2_kwon_v2(input_dims, dropout_rate):
    tf.keras.backend.clear_session()
    inp = Input(shape=input_dims, name='Input')
    x = inp
    x = Dense(12, activation='relu', name='Dense_1')(x)
    x = BatchNormalization(name='BatchNormalization_1')(x)
    x = Dropout(dropout_rate, name='Dropout_1')(x, training=True)
    x = Dense(6, activation='relu', name='Dense_2')(x)
    x = BatchNormalization(name='BatchNormalization_2')(x)
    x = Dropout(dropout_rate, name='Dropout_2')(x, training=True)
    output = Dense(2, activation='softmax', name='output')(x)
    model = models.Model(inputs=inp, outputs=output)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    return model

