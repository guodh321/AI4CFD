"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the python script to implement normal predition and 4D-Var DA using PredANN for multiple cases

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
import os
import sys
import time
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import matplotlib.pyplot as plt


sys.path.append("..")
import tools as t
# from tools import Data_preprocessing as t


# *************values setting***********
root_path = '..data/' # set the root path where data is stored

epochs = 12000

ntimes = 9 # consecutive times for the AAE
step = 1 # step between times
ncoeffs = 249  #123
BATCH_SIZE = 64
timesteps = 720

cases_num = 7
# *************values setting***********

dataset_type = 'all'
X_com_com_list = joblib.load(root_path + 'Cotrace_fixed_720_cases/caseALL_control/X_com_com_control_all_list.pkl')
X_com_com_array = np.array(X_com_com_list)

# add time
X_com_com_time = []
X_train_4d_time_all_list = []
for i in range(6):
    times = np.linspace(1,timesteps,timesteps)
    print(times.shape)
    times = np.array(times).reshape(len(times),1)
    X_test_points = np.hstack((times,X_com_com_list[i]))
    print(X_test_points.shape)
    X_com_com_time.append(X_test_points)
    time_ct = t.concat_timesteps(X_test_points, ntimes, step)
    time_dataset, X_train_4d_time = t.create_dataset(time_ct, ncoeffs+1, ntimes, BATCH_SIZE)
    X_train_4d_time_all_list.append( X_train_4d_time)

train_ct_all = joblib.load(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/train_ct_all.pkl')
print(train_ct_all.shape)
ncoeffs = num_coeffs = 249


train_ct_all_list = []
train_dataset_all_list = []
X_train_4d_all_list = []
X_compress_all_list = []

for i in range(cases_num):
    X_comp = X_com_com_array[i]
    train_ct = t.concat_timesteps(X_comp, ntimes, step)
    # create dataset
    train_dataset, X_train_4d = t.create_dataset(
        train_ct, ncoeffs, ntimes, BATCH_SIZE)
    
    X_compress_all_list.append(X_comp)
    train_ct_all_list.append(train_ct)
    X_train_4d_all_list.append(X_train_4d)
    train_dataset_all_list.append(train_dataset)



# load the whole model (shuffled)
case = '_caseall'
epochs = 12000

nn, discriminator = load_model(root_path + 'Cotrace_fixed_720_cases/caseALL_models/ann_train_n{}_e{}_s{}{}_shuffled.h5'.format(ntimes, epochs, step, case)).layers

predann = load_model(root_path + 'Cotrace_fixed_720_cases/caseALL_models/ann_train_n{}_e{}_s{}{}_shuffled.h5'.format(ntimes, epochs, step, case), compile=False)


# Normal prediction
X_predict_all_list = []

for i in range(cases_num):
    X_compressed = X_compress_all_list[i]
    t_start = 0
    prediction_num = X_compressed.shape[0]-ntimes+1

    scaler = 1
    num_sample = 4

    nth_node = 5

    preds = []
    # choose initial input
    initial_input = X_compressed[t_start:t_start+ntimes-1,:]
    current_input = initial_input.reshape((1, ntimes-1, ncoeffs, 1))
    # current_input = tf.data.Dataset.from_tensor_slices(initial_input)
    # print(current_input.shape)
    for j in range(prediction_num):
        pred = nn.predict(current_input)

        preds.append(pred)

        new_input = np.concatenate((current_input[:,1:], pred.reshape((1, 1, ncoeffs, 1))), axis = 1)

        current_input = new_input
    preds = np.array(preds).reshape((prediction_num, ncoeffs))
    preds = np.concatenate((initial_input, preds))

    print(initial_input.shape)
    print(preds.shape)

    X_predict = np.array(preds)

    X_predict_all_list.append(X_predict)

np.save(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/ann_control_nprediction_caseALL.npy', X_predict_all_list)
print('finish normal prediction')

# 4dVar
def predict_coding(initial_pred, real_coding):
    loss = []
    initial_pred_list = []
    for epoch in range(20):
        encoder_output = encoder(initial_pred, training=False)
        decoder_output = decoder(encoder_output, training=False)
        loss.append(mse_loss(real_coding, decoder_output[:,:(ntimes - 1),:,:]).numpy())
        initial_pred[:,(ntimes - 1):,:,:] = decoder_output[:,(ntimes - 1):,:,:]
        initial_pred_list.append(initial_pred)

        # if epoch > 30: 
        #     # if (loss[-1]-loss[-2]) <= 1e-5:
        #     if(loss[-1] <= 1e-4):
        #         break

    return decoder_output, initial_pred_list[-2], loss

mse = tf.keras.losses.MeanSquaredError()
def mse_loss(inp, outp):   
    inp = tf.reshape(inp, [-1, codings_size])
    outp = tf.reshape(outp, [-1, codings_size])
    return mse(inp, outp)


# calculate the loss between real and prediction of POD coiffecients
def mse_loss(inp, outp):
    return tf.reduce_mean(tf.math.squared_difference(inp, outp))

def optimize_coding_nn_sensor(initial_condition, real_coding, epochs):
    
    optimizer = tf.keras.optimizers.Adam(1e-2)
    
    @tf.function
    def opt_step(initial_condition, real_coding):
        with tf.GradientTape() as tape:
            tape.watch(initial_condition)
            gen_output = nn(initial_condition, training=False)
            real_coding_sensor = real_coding[:,100:]
            gen_output_sensor = gen_output[:, 100:]
            
             
            loss = mse_loss(real_coding_sensor, gen_output_sensor) 
            # loss = mse_loss(real_coding, gen_output)  

        gradient = tape.gradient(loss, initial_condition)  
        optimizer.apply_gradients(zip([gradient], [initial_condition]))  

        return loss
    
    loss = []
    for epoch in range(epochs):
        loss.append(opt_step(initial_condition, real_coding).numpy())
        
    plt.plot(loss)
    #plt.grid()
    plt.show
        
    return initial_condition, loss[-1]  #returns the optimized input that generates the desired output

X_predict_4dVar_all_list = []
for c in range(1, 1+cases_num):
    print('case{} in progress'.format(c))
    X_test_for_conv = X_train_4d_all_list[c-1]
    X_compressed = X_compress_all_list[c-1]
    num_coeffs = 249
    input_timestamps = 9
    codings_size = X_compressed.shape[1]

    case = c
    X_test_4d = X_train_4d_all_list[case-1]
    X_compressed_test = X_com_com_list[case-1]

    n = 0
    X_predict_4dVar = []
    X_real = []
    latent_values = X_test_4d[n][:-1].reshape(-1, ntimes-1, ncoeffs,1)
    latent_values = tf.Variable(latent_values)
    real_coding = X_test_4d[n][-1].reshape(1,-1)
    real_coding = tf.constant(real_coding)
    real_coding = tf.cast(real_coding, dtype=tf.float32)

    prediction_num = X_compressed_test.shape[0]-ntimes

    for i in range(X_compressed_test.shape[0]-ntimes): #range(2000,len(X_train_concat)-1):
        start = time.time()
        latent_values, loss = optimize_coding_nn_sensor(latent_values, real_coding, epochs=60)
        print('Loss iteration '+str(i+1)+': '+str(loss), end=' - ')
            
        # gen_predict = ann(latent_values).numpy().flatten()
        gen_predict = nn(latent_values)
        X_predict_4dVar.append(gen_predict)
        # gen_predict[-2:] = R0s_run

        X_real.append(real_coding)

        latent_values = (np.vstack((latent_values[0, 1:, :, 0], gen_predict))).reshape(1, 8, ncoeffs, 1)
        latent_values = tf.Variable(latent_values)
        print(latent_values.shape)

        # real_coding = np.concatenate((real_coding, gen_predict.reshape(1,-1)), axis=1)[:,ncoeffs:]
        real_coding = X_test_4d[i][-1].reshape(1,-1)
        real_coding = tf.constant(real_coding)
        real_coding = tf.cast(real_coding, dtype=tf.float32)
        print ('{:.0f}s'.format( time.time()-start))
        
    X_predict_4dVar = np.array(X_predict_4dVar)
    print(X_predict_4dVar.shape)
    X_real = np.array(X_real)
    print(X_real.shape)
    plt.grid()

    X_predict = np.array(X_predict)
    X_predict_4dVar_all_list.append(X_predict)
X_predict_4dVar_all_list = np.array(X_predict_4dVar_all_list)
print(X_predict_4dVar_all_list.shape)

np.save(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/ann_control_4dvar_caseALL.npy', X_predict_4dVar_all_list)
print('finish 4d var')