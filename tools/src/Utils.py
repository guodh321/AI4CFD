"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the tool .py file including some helper functions in this project

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


root_path = '..data/' # set the root path where data is stored

# idw
def idw(field, index_sensor=[s for s in range(0, 18)]):
    Distance_selected = np.load(root_path + 'idw_paras/Distance_selected.npy')
    Index = np.load(root_path + 'idw_paras/Index.npy')

    # index_sensor = np.array(index_sensor)
    # index_sensor = [index_sensor]
    # print(index_sensor)
    idw_results = []
    for i in index_sensor:
        i = int(i)
        distances = Distance_selected[i]
        index = Index[i]
        values = []
        for i in range(4):
            v = field[index[i]]
            values.append(v)
        values = np.array(values)
        # In IDW, weights are 1 / distance
        weights = 1.0 / distances

        # Make weights sum to one
        weights /= weights.sum(axis=0)

        idw_result = np.dot(weights.T, values)

        idw_results.append(idw_result)
    idw_results = np.array(idw_results)

    return idw_results


fields_name_vtu = ['Tracer', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Temperature', 'Humidity', 'Virus1']

# Plot functions
def plot_field(field, field_index, num_nodes=10, label='Prediction'):
    fig, ax = plt.subplots(5,2, figsize=[20,25])
    for i in range(num_nodes):
        ax.flatten()[i].plot(field[:,i], '-', label=label)  
        ax.flatten()[i].legend()
        ax.flatten()[i].set_title('Node: '+ str(i+1))
        ax.flatten()[i].set_xlabel('Time(s)')
    fig.suptitle('Resutls on Field: ' + fields_name_vtu[field_index], fontsize=16)

