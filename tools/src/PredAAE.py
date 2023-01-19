"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the tool .py file including PredAAE model building functions

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
import vtk
# import vtktools
import time

import matplotlib.pyplot as plt


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()


# model structure

def make_encoder_aae(ntimes, ncoeffs, latent_dim):
    model = tf.keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            16, (3, 3), strides=(
                1, 1), padding='same', input_shape=[
                ntimes, ncoeffs, 1]))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same'))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2D(4, (3, 3), strides=(1, 1), padding='same'))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(latent_dim))
    return model


def make_decoder_aae(ntimes, ncoeffs, latent_dim):
    model = tf.keras.Sequential()
    model.add(
        keras.layers.Dense(
            ntimes * ncoeffs * 4,
            use_bias=False,
            activation='relu',
            input_shape=(
                latent_dim,
            )))
    model.add(keras.layers.Reshape((ntimes, ncoeffs, 4)))

    model.add(
        keras.layers.Conv2DTranspose(
            8, (3, 3), strides=(
                1, 1), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            4, (3, 3), strides=(
                1, 1), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            1, (3, 3), strides=(
                1, 1), padding='same', output_padding=[
                0, 0], use_bias=False, activation='sigmoid'))

    return model


def make_encoder_aae1(ntimes, ncoeffs, latent_dim):
    model = tf.keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            128, (3, 3), strides=(
                1, 1), padding='same', input_shape=[
                ntimes, ncoeffs, 1]))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(latent_dim))
    return model


def make_decoder_aae1(ntimes, ncoeffs, latent_dim):
    model = tf.keras.Sequential()
    model.add(
        keras.layers.Dense(
            ntimes * ncoeffs * 4,
            use_bias=False,
            activation='relu',
            input_shape=(
                latent_dim,
            )))
    model.add(keras.layers.Reshape((ntimes, ncoeffs, 4)))

    model.add(
        keras.layers.Conv2DTranspose(
            64, (3, 3), strides=(
                1, 1), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            32, (3, 3), strides=(
                1, 1), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            1, (3, 3), strides=(
                1, 1), padding='same', output_padding=[
                0, 0], use_bias=False, activation='sigmoid'))

    return model


def make_discriminator_aae(latent_dim):
    model = tf.keras.Sequential()
    model.add(
        keras.layers.Dense(
            256,
            activation="relu",
            input_shape=(
                latent_dim,
            )))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(1))
    return model

# loss functions
def compute_reconstruction_loss(x, x_logit):
    reconstruction_loss = mse(x, x_logit)
    return reconstruction_loss


def compute_discriminator_loss(fake_output, real_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def compute_generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
