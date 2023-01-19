"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the python script to train PredANN

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
import sys
import os
sys.path.append("..")
import tools as t

import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt





# from tools import Data_preprocessing as t

# *************values setting***********
root_path = '..data/' # set the root path where data is stored
ncoeffs = 249
ntimes = 9  # consecutive times for the AAE
step = 1  # step between times
BATCH_SIZE = 64 # 64

epochs = 12000

learning_rate_nn = 0.0001
learning_rate_d = 0.0005
# *************values setting***********

# *************normal use***********
# PCA model
# pca_compress = joblib.load(
#     root_path +
#     'Cotrace_fixed_720_npys/train_pca_compress.pkl')
# # POD coefficients
# X_compressed = joblib.load(
#     root_path +
#     'Cotrace_fixed_720_npys/train_pod_coefficients.pkl')

# scaler_minmax_train = MinMaxScaler((0, 1))
# X_compressed = scaler_minmax_train.fit_transform(X_compressed)


# ncoeffs = X_compressed.shape[1]  # number of POD coefficients
# data_ct = t.concat_timesteps(X_compressed, ntimes, step)

# # train and valid split
# # train_ct, valid_ct = t.train_test_split(data_ct)
# train_ct = data_ct


# *************used for cases***********
# case = 1
# train_ct = joblib.load(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/train_ct_case{}.pkl'.format(case))
case = 'all'
train_ct = joblib.load(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/train_ct_{}.pkl'.format(case))
# train_ct = t.shuffle_along_axis(train_ct, axis=1)
# *************used for cases***********

# create dataset
train_dataset, X_train_4d = t.create_dataset(
    train_ct, ncoeffs, ntimes, BATCH_SIZE)
# valid_dataset, X_valid_4d = t.create_dataset(
#     valid_ct, ncoeffs, ntimes, BATCH_SIZE)


# Build ANN network
nn = t.make_nn_ann(ntimes, ncoeffs)
discriminator = t.make_discriminator_ann(ncoeffs)

ann = keras.models.Sequential([nn, discriminator])


reconstruction_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
discriminator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)

r_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_nn)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_d)

# Notice the use of `tf.function` for speeding up calculation
# This annotation causes the function to be "compiled".


@tf.function
def train_step(batch):
    batch_X = batch[:, :-1]
    batch_Y = batch[:, -1, :, 0]
    # NN update
    with tf.GradientTape() as ae_tape:
        encoder_output = nn(batch_X, training=True)
        reconstruction_loss = t.compute_nn_loss(
            batch_Y, encoder_output)

    r_gradients = ae_tape.gradient(
        reconstruction_loss,
        nn.trainable_variables)
    r_optimizer.apply_gradients(zip(r_gradients, nn.trainable_variables))

    # Discriminator update
    with tf.GradientTape() as d_tape:
        z = nn(batch_X, training=True)
        true_z = batch_Y
        fake_output = discriminator(z, training=True)
        true_output = discriminator(true_z, training=True)
        discriminator_loss = t.compute_discriminator_loss(
            fake_output, true_output)
    d_gradients = d_tape.gradient(
        discriminator_loss,
        discriminator.trainable_variables)
    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables))

    reconstruction_mean_loss(reconstruction_loss)
    discriminator_mean_loss(discriminator_loss)


def train(dataset, epochs):
    hist = []
    reconstruction = []
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        # train_X_dataset, train_Y_dataset = dataset
        for batch in dataset:
            # print(batch.size)
            train_step(batch)

        hist.append([reconstruction_mean_loss.result().numpy(),
                    discriminator_mean_loss.result().numpy()])
        reconstruction.append(reconstruction_mean_loss.result().numpy())

        # Resets all of the metric state variables.
        # This function is called between epochs/steps, when a metric is evaluated during training.
        # generator_mean_loss.reset_states()
        discriminator_mean_loss.reset_states()
        reconstruction_mean_loss.reset_states()

        print("nn loss: ", hist[-1][0]," - ", "discriminator loss: ", hist[-1][1])

    return hist, reconstruction


hist, reconstruction = train(train_dataset, epochs=epochs)

print('nn loss in last epochs: ', reconstruction[-1])

fig, ax = plt.subplots(1, 1, figsize=[16, 8])
ax.plot(hist)
ax.legend(['loss_nn', 'loss_disc'])
# ax.set_yscale('log')
ax.grid()
# save the figure to file
fig.savefig(root_path + 'Cotrace_fixed_720_cases/caseALL_figs/ann_training_loss_n{}_e{}_s{}_case{}_shuffled.png'.format(ntimes, epochs, step, case))
plt.close(fig)    # close the figure window

# save trained model
# joblib.dump(scaler_minmax_train, root_path +
#             'Cotrace_fixed_720_npys/scalers/ann_training_scaler_minmax.pkl')
ann.save(root_path + 'Cotrace_fixed_720_cases/caseALL_models/ann_train_n{}_e{}_s{}_case{}_shuffled.h5'.format(ntimes, epochs, step, case))
print('trained model saved.')
