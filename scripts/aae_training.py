"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the python script to train PredAAE

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
import sys
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
sys.path.append("..")
import tools as t


# from tools import Data_preprocessing as t

# *************values setting***********
root_path = '..data/' # set the root path where data is stored
ncoeffs = 249 # 123
seed = 42
ntimes = 9  # consecutive times for the AAE
step = 1  # step between times
BATCH_SIZE = 64

epochs = 12000
latent_dim = 500 #200

learning_rate_ae = 0.0001
learning_rate_d = 0.0005
# *************values setting***********


# # PCA model
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
# # train_ct, valid_ct = t.train_test_split(concat_timesteps=data_ct, seed=seed)
# train_ct = data_ct


# *************used for cases***********
case = 'all'
# cases_pod_sensor_vp_train_scaled = joblib.load(root_path + 'Cotrace_fixed_720_cases/caseALL_sensors/cases_pod_sensor_vp_train_scaled.pkl')
# X_compressed = cases_pod_sensor_vp_train_scaled[case-1]
# scaler_minmax_train = MinMaxScaler((0, 1))
# X_compressed = scaler_minmax_train.fit_transform(X_compressed)
# ncoeffs = X_compressed.shape[1]  # number of POD coefficients
# data_ct = t.concat_timesteps(X_compressed, ntimes, step)

# train and valid split
# train_ct, valid_ct = t.train_test_split(concat_timesteps=data_ct, seed=seed)
# train_ct = data_ct
train_ct = joblib.load(root_path + 'Cotrace_fixed_720_cases/caseALL_npys/train_ct_{}.pkl'.format(case))
# *************used for cases***********


# create dataset
train_dataset, X_train_4d = t.create_dataset(
    train_ct, ncoeffs, ntimes, BATCH_SIZE)
# valid_dataset, X_valid_4d = t.create_dataset(
#     valid_ct, ncoeffs, ntimes, BATCH_SIZE)



# Build PredAAE network
encoder = t.make_encoder_aae1(ntimes, ncoeffs, latent_dim)
decoder = t.make_decoder_aae1(ntimes, ncoeffs, latent_dim)
discriminator = t.make_discriminator_aae(latent_dim)

autoencoder = keras.models.Sequential([encoder, decoder])
enc_disc = keras.models.Sequential([encoder, discriminator])


reconstruction_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
discriminator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)

r_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_ae)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_ae)

d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_d)

# Notice the use of `tf.function` for speeding up calculation
# This annotation causes the function to be "compiled".
@tf.function
def train_step(batch):
    # Autoencoder update
    with tf.GradientTape() as ae_tape:
        encoder_output = encoder(batch, training=True)
        decoder_output = decoder(encoder_output, training=True)
        reconstruction_loss = t.compute_reconstruction_loss(
            batch, decoder_output)

    r_gradients = ae_tape.gradient(
        reconstruction_loss,
        encoder.trainable_variables +
        decoder.trainable_variables)
    r_optimizer.apply_gradients(
        zip(r_gradients, encoder.trainable_variables + decoder.trainable_variables))

    # Discriminator update
    with tf.GradientTape() as d_tape:
        z = encoder(batch, training=True)
        true_z = tf.random.normal(shape=(z.shape))
        fake_output = discriminator(z, training=True)
        true_output = discriminator(true_z, training=True)
        discriminator_loss = t.compute_discriminator_loss(
            fake_output, true_output)
    d_gradients = d_tape.gradient(
        discriminator_loss,
        discriminator.trainable_variables)
    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables))

    # Generator update
    with tf.GradientTape() as g_tape:
        z = encoder(batch, training=True)
        fake_output = discriminator(z, training=True)
        generator_loss = t.compute_generator_loss(fake_output)
    g_gradients = g_tape.gradient(generator_loss, encoder.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, encoder.trainable_variables))

    reconstruction_mean_loss(reconstruction_loss)
    generator_mean_loss(generator_loss)
    discriminator_mean_loss(discriminator_loss)


def train(dataset, epochs):
    hist = []
    reconstruction = []
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        for batch in dataset:
            train_step(batch)

        hist.append([generator_mean_loss.result().numpy(),
                    discriminator_mean_loss.result().numpy()])
        reconstruction.append(reconstruction_mean_loss.result().numpy())

        generator_mean_loss.reset_states()
        discriminator_mean_loss.reset_states()
        reconstruction_mean_loss.reset_states()

        print("encoder loss: ", hist[-1][0]," - ", "discriminator loss: ", hist[-1][1])
        print("autoencoder loss: ", reconstruction[-1])

    return hist, reconstruction


hist, reconstruction = train(train_dataset, epochs=epochs)
print(reconstruction[-1])

# save loss_enc and loss_dis plot
fig, ax = plt.subplots(1,1, figsize=[20,10])
ax.plot(hist)
ax.legend(['loss_enc', 'loss_disc'])
#ax.set_yscale('log')
ax.grid()
fig.savefig(root_path + 'Cotrace_fixed_720_cases/caseALL_figs/aae_encdis_loss_n{}_e{}_s{}_l{}_case{}.png'.format(ntimes, epochs, step, latent_dim, case))
plt.close(fig)    # close the figure window

# save loss_reconstruction plot
fig, ax = plt.subplots(1,1, figsize=[16,8])
ax.plot(reconstruction)
ax.legend(['loss_reconstruction'])
ax.set_yscale('log')
ax.grid()
fig.savefig(root_path + 'Cotrace_fixed_720_cases/caseALL_figs/aae_recon_loss_n{}_e{}_s{}_l{}_case{}.png'.format(ntimes, epochs, step, latent_dim, case))
plt.close(fig)    # close the figure window

# save trained model
autoencoder.save(root_path +
                 'Cotrace_fixed_720_cases/caseALL_models/aae_ae_n{}_e{}_s{}_l{}_case{}.h5'.format(ntimes, epochs, step, latent_dim, case))
enc_disc.save(root_path +
              'Cotrace_fixed_720_cases/caseALL_models/aae_enc_disc_n{}_e{}_s{}_l{}_case{}.h5'.format(ntimes, epochs, step, latent_dim, case))
