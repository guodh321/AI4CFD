"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the tool .py file including PredANN model building functions

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""

# __all__ = ['']

# Imports
import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass

# Common imports
import math
import numpy as np
# import vtktools
import time


# model structure
# Build ANN network
def make_nn_ann(ntimes, ncoeffs):
    model = tf.keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            64, (3, 3), strides=(
                1, 1), padding='same', input_shape=[
                ntimes - 1, ncoeffs, 1]))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(ncoeffs))
    return model


def make_discriminator_ann(ncoeffs):
    model = tf.keras.Sequential()
    model.add(
        keras.layers.Dense(
            256,
            activation="relu",
            input_shape=(
                ncoeffs,
            )))
    # model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(128, activation="relu"))
    # model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(64, activation="relu"))
    # model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(1))
    return model

# loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()


def compute_nn_loss(x, x_logit):
    reconstruction_loss = mse(x, x_logit)
    return reconstruction_loss


def compute_discriminator_loss(fake_output, real_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
