"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the python script to extract and convert vtu files to npy files for multiple fields and cases

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


sys.path.append("..")
import tools as t

root_path = '..data/' # set the root path where data is stored


# Load vtu data
#--------------------------------#
#-- Choose variables           --#
#--------------------------------#
for i in range(1, 9):
    # The path where the vtu files are located
    path = root_path + 'Cotrace_fixed_720_cases/case{}/'.format(i)
    # The prefix of the file name of the vtu file
    name_simu = 'Cotrace_fixed'
    vtu_start = 1
    vtu_end   = 721  # 721
    vtu_step  = 1
    fields_list = ['Tracer ', 'Velocity', 'Temperature', 'Humidity', 'Virus1']

    times = 720

    print('loading tracer')
    tracer = t.loadvtufile(path, name_simu, 'Tracer', vtu_start, vtu_end, vtu_step)
    print('loading velocity')
    velocity = t.loadvtufile(path, name_simu, 'Velocity', vtu_start, vtu_end, vtu_step)
    print('loading temp')
    temp = t.loadvtufile(path, name_simu, 'Temperature', vtu_start, vtu_end, vtu_step)
    print('loading humid')
    humid = t.loadvtufile(path, name_simu, 'Humidity', vtu_start, vtu_end, vtu_step)
    print('loading virus')
    virus = t.loadvtufile(path, name_simu, 'Virus1', vtu_start, vtu_end, vtu_step)

    vx = velocity[:,:,0].reshape(times,-1,1)
    vy = velocity[:,:,1].reshape(times,-1,1)
    vz = velocity[:,:,2].reshape(times,-1,1)

    print(tracer.shape)
    print(velocity.shape)
    print(temp.shape)
    print(humid.shape)
    print(virus.shape)
    print(vx.shape)


    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_tracer.npy'.format(i, i), tracer)
    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_velocity.npy'.format(i, i), velocity)
    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_temp.npy'.format(i, i), temp)
    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_humid.npy'.format(i, i), humid)
    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_virus.npy'.format(i, i), virus)
    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_vx.npy'.format(i, i), vx)
    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_vy.npy'.format(i, i), vy)
    np.save(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_vz.npy'.format(i, i), vz)