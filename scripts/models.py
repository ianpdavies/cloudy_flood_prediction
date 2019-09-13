import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# ---------------------------------------------------------------

tf.keras.backend.clear_session()

# Using Functional API
# def get_NN_MCD():
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

# ---------------------------------------------------------------
# NN with dropout before ReLU layers 

tf.keras.backend.clear_session()

def get_NN_MCD1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_DIMS)),
    model.add(tf.keras.layers.Lambda(lambda x: K.dropout(x, level=DROPOUT_RATE))),
    model.add(tf.keras.layers.Dense(units=24,
                                    activation='relu')),
    model.add(tf.keras.layers.Lambda(lambda x: K.dropout(x, level=DROPOUT_RATE))),
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

# ---------------------------------------------------------------