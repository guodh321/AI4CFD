"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the tool .py file including general functions for data processing

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""

# __all__ = ['']

# Imports
import sys
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras

# Common imports
import math
import numpy as np
import meshio
import vtk
from .vtktools import *
# import .vtktools as vtktools
# import vtktools.py as vtktools
import time


# load toy data
def loadtoydata(type='all', time_start=0, time_end=7326, freq=45):
    air_quality = np.load('../data/toydata/Monitor_air_quality.npy')
    time = np.load('../data/toydata/Monitor_time.npy')

    # smooth original data
    n0p = 54
    count = math.floor(air_quality.shape[1] / freq)
    time_smoothed = np.zeros(shape=(count, 1))
    data_smoothed = np.zeros(shape=(n0p, count))
    for j in range(n0p):
        for i in range(count):
            k = i * freq
            time_smoothed[i][:] = time[k]
            data_smoothed[j][i] = air_quality[j][k]

    # choose data type to return
    if type == 'all':
        res = data_smoothed
    elif type == 'co2':
        res = data_smoothed[:18, :]
    elif type == 'temp':
        res = data_smoothed[18:36, :]
    elif type == 'humid':
        res = data_smoothed[36:, :]

    # choose data of corresponding time period to return
    time_slice = time_smoothed[int(
        res.shape[1] * time_start / 7326):int(res.shape[1] * time_end / 7326)]
    res = res[:, int(res.shape[1] * time_start / 7326)
                     :int(res.shape[1] * time_end / 7326)]

    return res, time_slice


# scale data
def scaledata(data, type='std'):
    if type == 'std':
        scaler = StandardScaler()
    elif type == 'mm':
        scaler = MinMaxScaler((0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


# load .vtu points
def loadvtumesh():
    tic = time.time()

    extension = '.vtu'

    all_data = []

    # ---------------------------------------------------------------------
    # EXTRACT DATA
    # ---------------------------------------------------------------------
    for vtuID in range(vtu_start, vtu_end, vtu_step):
        filename = path + name_simu + '_' + str(vtuID) + extension
        print('\n  ' + str(filename))

        vtu_data = vtu(filename)
        # data = vtu_data.GetField(fieldname)
        data = meshio.read(filename)
        all_data.append(data)

    toc = time.time()  # added
    print('\n\nTime : ', toc - tic, 'sec')
    return np.array(all_data)


# load .vtu data
def loadvtufile(path, name_simu, fieldname, vtu_start, vtu_end, vtu_step):
    """
    Read in .vtu files for a particular field

    Parameters
    ----------
    path : str
        path to the folder containing the .vtu files
    name_simu : str
        name of the simulation data
    field_name : str
        field to read in e.g. CO2_ppm
    vtu_start : int
        file number to start reading from
    vtu_end : int
        file number to read up to
    vtu_step : int
        how many files to step across
    Returns
    -------
    numpy.ndarray
        Returns the data from a particular field
    """
    tic = time.time()

    extension = '.vtu'

    all_data = []

    # ---------------------------------------------------------------------
    # EXTRACT DATA
    # ---------------------------------------------------------------------
    for vtuID in range(vtu_start, vtu_end, vtu_step):
        filename = path + name_simu + '_' + str(vtuID) + extension
        print('\n  ' + str(filename))

        vtu_data = vtu(filename)
        data = vtu_data.GetField(fieldname)
        # data = meshio.read(filename)
        all_data.append(data)

    toc = time.time()  # added
    print('\n\nTime : ', toc - tic, 'sec')
    return np.array(all_data)


# concat timesteps
def concat_timesteps(X_train, ntimes, step):
    """
    Concat timesteps according to given ntimes and step

    Args:
        X_train (numpy array): the time steps to concat
        ntimes (int): the number of time steps to concat for one sample
        step (int): the gap between two concated timsteps

    Returns:
        concat samples (numpy array): the concat timesteps
    """
    X_train_concat = []
    for i in range(len(X_train) - ntimes * step):
        X_train_concat.append(X_train[i:i + ntimes * step:step])
    return np.array(X_train_concat)


# train test dataset split
def train_test_split(concat_timesteps, testFrac=0.2, seed=42):
    """
    The function to split train and test sets for concat timesteps

    Args:
        concat_timesteps (numpy array): the concated samples to be splitted
        testFrac (float, optional): the fraction to implement train and test split. Defaults to 0.2.
        seed (int, optional): the random seed to split, aming to 
        make differenting call of the function have the same result. Defaults to 42.

    Returns:
        Train and test concat_timesteps
    """
    testList = []
    trainList = []
    np.random.seed(seed)
    for i in range(len(concat_timesteps)):
        rann = np.random.random()  # randomly reassign samples
        if rann < testFrac:
            testList.append(i)
        else:
            trainList.append(i)

    nTrain = len(trainList)  # count the number in each set
    nTest = len(testList)
    print("Training samples =", nTrain, "Testing samples =", nTest)

    # convert to tensors according the random selection above
    trainIds = trainList
    testIds = testList
    train_set = concat_timesteps[trainIds]
    test_set = concat_timesteps[testIds]

    return train_set, test_set


# create dataset for model training
def create_dataset(X_train_concat, ncoeffs, ntimes, BATCH_SIZE):
    X_train_concat_flatten = X_train_concat.reshape(
        X_train_concat.shape[0], ncoeffs * ntimes)
    X_train_4d = X_train_concat.reshape(
        (X_train_concat.shape[0], ntimes, X_train_concat.shape[2], 1))

    X_train_4d = X_train_4d.astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train_4d)
    train_dataset = train_dataset.shuffle(len(X_train_concat))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset, X_train_4d


# create dataset for model training ANN
def create_dataset_ann(X_train_concat, ncoeffs, ntimes, BATCH_SIZE):
    X_train_concat_flatten = X_train_concat.reshape(
        X_train_concat.shape[0], ncoeffs * ntimes)
    X_train_4d = X_train_concat.reshape(
        (X_train_concat.shape[0], ntimes, X_train_concat.shape[2], 1))
    X_train_4d = X_train_4d.astype('float32')

    Y_train_4d = X_train_4d[:, -1, :, :]
    X_train_4d = X_train_4d[:, :-1, :, :]

    train_dataset = tf.data.Dataset(X_train_4d, Y_train_4d)
    train_dataset = train_dataset.shuffle(len(X_train_concat))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset, X_train_4d, Y_train_4d


# shuffle concat timesteps
def shuffle_along_axis(ct, axis):
    """
    Shuffle the concated timesteps from different cases according to axis

    Args:
        ct (numpy array): the concated timesteps to shuffle
        axis (int): the axis to shuffle along with

    Returns:
        shuffled concated samples
    """
    idx = np.random.rand(*ct.shape).argsort(axis=axis)
    return np.take_along_axis(ct,idx,axis=axis)