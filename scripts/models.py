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
    model.add(tf.keras.layers.Lambda(lambda x: K.dropout(x, level=DROPOUT_RATE))),0
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