"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the python script to implement normal predition and 4D-Var DA using PredAAE for multiple cases

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
import os
import sys
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
cases_num = 6
timesteps = 720
# *************values setting***********

dataset_type = 'all'
X_com_com_list = joblib.load(root_path + 'Cotrace_fixed_720_cases/caseALL_sensors/cases_pod_sensor_vp_{}.pkl'.format(dataset_type))
X_com_com_array = np.array(X_com_com_list)
print(X_com_com_array.shape)
print(X_com_com_array[0].shape)

# add time
X_com_com_time = []
X_train_4d_time_all_list = []
for i in range(cases_num):
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



# load the whole model
case = 'all'
latent_dim = 500
autoencoder = load_model(root_path + 'Cotrace_fixed_720_cases/caseALL_models/aae_ae_n{}_e{}_s{}_l{}_case{}.h5'.format(ntimes, epochs, step, latent_dim, case), compile=False)

encoder, decoder = load_model(root_path + 'Cotrace_fixed_720_cases/caseALL_models/aae_ae_n{}_e{}_s{}_l{}_case{}.h5'.format(ntimes, epochs, step, latent_dim, case)).layers


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
def mse_loss_op(inp, outp):
    return tf.reduce_mean(tf.math.squared_difference(inp, outp))

# calculate the loss between real and prediction of POD coiffecients
def mse_loss_op(inp, outp):
    return tf.reduce_mean(tf.math.squared_difference(inp, outp))


def optimize_coding_aae_sensorvp(initial_condition, real_coding, epochs):
    
    optimizer = tf.keras.optimizers.Adam(1e-2)
    
    @tf.function
    def opt_step(initial_condition, real_coding):
        with tf.GradientTape() as tape:
            tape.watch(initial_condition)
            gen_output = autoencoder(initial_condition, training=False)
            real_coding_sensor = tf.reshape(real_coding, (1, input_timestamps, ncoeffs, 1))[:,:,100:, :]
            gen_output_sensor = gen_output[:,:,100:, :]
            
             
            loss = mse_loss_op(real_coding_sensor, gen_output_sensor) 

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


X_predict_all_list = []
X_predict_4dVar_all_list = []
for c in range(1, cases_num+1):
    print('case{} in progress'.format(c))
    X_test_for_conv = X_train_4d_all_list[c-1]
    X_compressed = X_compress_all_list[c-1]
    input_timestamps = 9
    codings_size = X_compressed.shape[1]

    
    # Start from time level 0

    n = 0
    prediction_num = 720

    real_value_da = X_test_for_conv[n].reshape(1,-1)

    real_coding = real_value_da
    real_coding = tf.constant(real_coding)
    real_coding = tf.cast(real_coding, dtype=tf.float32)

    # Extract value of 0-(m-2) time levels as real value
    real_value = real_value_da[:,:num_coeffs*(input_timestamps - 1)]

    real_value = real_value.reshape((1, input_timestamps-1, X_test_for_conv.shape[2], 1))

    # Set value of time level m-1 as same as that of time level m-2
    initial_pred = np.concatenate((real_value, real_value[:,-1:,:,:]), axis=1)

    # Predict a point forward in time (time level m-1)
    prediction_values, initial_condition, loss = predict_coding(initial_pred, real_value)
    # print(initial_condition.shape)
    initial_condition = tf.Variable(initial_condition)
    initial_condition, loss = optimize_coding_aae_sensorvp(initial_condition, real_coding, epochs=60)

    prediction_values = autoencoder(initial_condition)

    # Update real value and initial guess
    X_predict = list(prediction_values.numpy().reshape(-1,num_coeffs))
    X_predict_4dvar = list(prediction_values.numpy().reshape(-1,num_coeffs))

    # prediction of time level m-1
    gen_predict = prediction_values[:,(input_timestamps - 1):,:,:]

    # Add the predicted value to the real value (time levels 1-(m-1))
    real_value = np.concatenate((real_value[:,1:,:,:], gen_predict), axis=1)

    # Set value of time level m as same as that of time level m-1
    initial_pred = np.concatenate((real_value, real_value[:,-1:,:,:]), axis=1)

    # Predict 10 points forward in time

    for i in range(710):

        real_value_da = X_test_for_conv[i+1].reshape(1,-1)
        real_coding = real_value_da
        real_coding = tf.constant(real_coding)
        real_coding = tf.cast(real_coding, dtype=tf.float32)

        # Normal prediction
        prediction_values, initial_condition, loss = predict_coding(initial_pred, real_value)
        gen_predict = prediction_values[:,(input_timestamps - 1):,:,:].numpy()
        X_predict.append(gen_predict.flatten())

        # Implement 4dVar within one timestep
        initial_condition = tf.Variable(initial_condition)
        initial_condition, loss = optimize_coding_aae_sensorvp(initial_condition, real_coding, epochs=60)
        prediction_values = autoencoder(initial_condition)
        
        # Update real value and initial guess
        gen_predict_4dvar = prediction_values[:,(input_timestamps - 1):,:,:].numpy()
        X_predict_4dvar.append(gen_predict_4dvar.flatten())
        real_value = np.concatenate((real_value[:,1:,:,:], gen_predict_4dvar), axis=1)
        initial_pred = np.concatenate((real_value, real_value[:,-1:,:,:]), axis=1)

    X_predict = np.array(X_predict)
    X_predict_4dvar = np.array(X_predict_4dvar)

    X_predict_all_list.append(X_predict)
    X_predict_4dVar_all_list.append(X_predict_4dvar)
X_predict_all_list = np.array(X_predict_all_list)
X_predict_4dVar_all_list = np.array(X_predict_4dVar_all_list)

np.save(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/aae_control_nprediction_caseALL.npy', X_predict_all_list)
np.save(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/aae_control_4dvar_caseALL.npy', X_predict_4dVar_all_list)