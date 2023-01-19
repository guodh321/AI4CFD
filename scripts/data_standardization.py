"""
Copyright: Donghu Guo

Author: Donghu Guo

DDescription: this is the python script to scale data set into (0, 1)

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os


for c in range(1, 9):
    root_path = '..data/' # set the root path where data is stored
    scaler_path = root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_scalers'.format(c,c)
    if c>=6:
        os.mkdir(scaler_path)
    # *************train test split***********
    case = c
    tracer = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_tracer.npy'.format(case,case))
    velocity = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_velocity.npy'.format(case,case))
    temp = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_temp.npy'.format(case,case))
    humid = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_humid.npy'.format(case,case))
    virus = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_virus.npy'.format(case,case))

    vx = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_vx.npy'.format(case,case))
    vy = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_vy.npy'.format(case,case))
    vz = np.load(root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_vz.npy'.format(case,case))

    fields = [tracer, vx, vy, vz, temp, humid, virus]
    fields_list = ['Tracer', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Temperature', 'Humidity', 'Virus1']

    fields_train = []
    fields_test = []
    test_t_start = int(720*(100/120))
    test_t_end = int(720*(120/120))

    for i in range(len(fields)):
        field_train = fields[i][:test_t_start]
        field_test = fields[i][test_t_start:]
        
        fields_train.append(field_train)
        fields_test.append(field_test)

    # store train and test sets
    joblib.dump(fields, root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_fields_all.pkl'.format(case, case))
    joblib.dump(fields_train, root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_fields_train.pkl'.format(case, case))
    joblib.dump(fields_test, root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_fields_test.pkl'.format(case, case))



    for d in ['all', 'train', 'test']:
    # ######################################
        # *************values setting***********
        
        dataset_type = d  # train or test
        fields_list = [
            'Tracer ',
            'Velocity_X',
            'Velocity_Y',
            'Velocity_Z',
            'Temperature',
            'Humidity',
            'Virus1']
        # *************values setting***********

        fields = joblib.load(
            root_path +
            'Cotrace_fixed_720_cases/case{}_npys/case{}_fields_'.format(c, c) +
            dataset_type +
            '.pkl')

        # change the output file name
        fields_scaled = []
        for i in range(len(fields)):
            scaler = MinMaxScaler((0, 1))
            scaler.fit(fields[i][:, :, 0])
            field_scaled = scaler.transform(fields[i][:, :, 0])
            joblib.dump(
                scaler,
                scaler_path +'/case{}_'.format(c) +
                dataset_type +
                '_scaler01_{}.pkl'.format(
                    fields_list[i]))
            print(fields[i].shape)
            print(field_scaled.shape)
            fields_scaled.append(field_scaled)

        all_values = np.hstack(fields_scaled)
        print("Snapshot Matrix shape", all_values.shape)
        # Store the all data in the form of numpy.ndarray
        joblib.dump(
            all_values,
            root_path +
            'Cotrace_fixed_720_cases/case{}_npys/case{}_'.format(c,c) +
            dataset_type +
            '_snapshot_matrix.pkl')
