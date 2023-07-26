import fasttext.util
import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import kerastuner as kt


def create_model(hp):
    model1 = keras.Sequential()
    # model.add(layers.Embedding(input_dim=300, output_dim=1))

    lstm_layer_1_units = hp.Int("lstm_layer_1_units", min_value=128, max_value=128 * 19, step=256+128)
    lstm_layer_2_units = hp.Int("lstm_layer_2_units", min_value=128, max_value=128 * 19, step=256+128)

    numeric_layer_1_units = hp.Int("numeric_layer_1_units", min_value=128, max_value=128 * 19, step=256+128)
    numeric_layer_2_units = hp.Int("numeric_layer_2_units", min_value=128, max_value=128 * 19, step=256+128)

    merged_layer_1_units = hp.Int("merged_layer_1_units", min_value=128, max_value=128 * 19, step=256+128)
    merged_layer_2_units = hp.Int("merged_layer_2_units", min_value=128, max_value=128 * 19, step=256+128)
    merged_layer_3_units = hp.Int("merged_layer_3_units", min_value=128, max_value=128 * 19, step=256+128)

    learning_rate = hp.Choice('learning_rate', values=[0.0005, 0.0001])

    model1.add(layers.LSTM(lstm_layer_1_units, return_sequences=True, input_shape=(None, 300)))
    model1.add(layers.LSTM(lstm_layer_2_units, return_sequences=False))
    # model.add(layers.TimeDistributed(layers.Dense(32, activation='sigmoid')))

    model2 = keras.Sequential()
    model2.add(layers.Dense(numeric_layer_1_units, input_shape=(training_x2.shape[1],)))
    model2.add(layers.Dense(numeric_layer_2_units))

    merged = Concatenate()([model1.output, model2.output])
    z = layers.Dense(merged_layer_1_units)(merged)
    z = layers.Dense(merged_layer_2_units)(z)
    z = layers.Dense(merged_layer_3_units)(z)
    z = layers.Dense(1)(z)

    model = Model(inputs=[model1.input, model2.input], outputs=z)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=60,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(loss=root_mean_squared_error,
                  # optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
                  #   optimizer=keras.optimizers.Adam(learning_rate=0.001))
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



    # tuner = kt.Hyperband(create_model,
    #                      objective='val_loss',
    #                      max_epochs=500,
    #                      factor=3,
    #                      directory='keras_tuner',
    #                      project_name='experiment')
    #
    # tuner.search(
    #     train_generator(training_x1, training_x2, training_y),
    #     validation_data=train_generator(validation_x1, validation_x2, validation_y),
    #     steps_per_epoch=32, validation_steps=300, validation_freq=1, epochs=500, verbose=1, callbacks=callbacks
    # )


    # best_hp = tuner.get_best_hyperparameters()[0]
    # model = tuner.hypermodel.build(best_hp)
