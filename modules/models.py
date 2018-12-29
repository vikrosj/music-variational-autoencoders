### keras ###
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, Lambda
from keras.layers import LSTM, Bidirectional, RepeatVector, Add, TimeDistributed, Reshape, Concatenate
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import kullback_leibler_divergence
import tensorflow as tf
import keras

### other ####
import numpy as np
import os
import time
from modules import mdn

# returns train, inference_encoder and inference_decoder models
def define_VAE(enc_units,
                dec_units,
                latent_units,
                timesteps,
                features,
                dropout,
                beta,
                learning_rate,
                epsilon_std):

    # define training encoder
    encoder_inputs = Input(shape=(timesteps, features), name="encoder_inputs")

    encoder0 = Bidirectional(LSTM(enc_units,
                                  dropout=dropout,
                                  return_sequences=True,
                                  unit_forget_bias=True,
                                  name="bidirectional_encoder0"))

    encoder1 = Bidirectional(LSTM(enc_units,
                                  unit_forget_bias=True,
                                  return_state=True,
                                  name="bidirectional_encoder1"))

    # intermediate outputs
    encoder_im_outputs = encoder0(encoder_inputs)

    # final outputs
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder1(encoder_im_outputs)


    # concatenating states
    state = Add(name='add_states')([forward_h, forward_c, backward_h, backward_c])


    # creating latent vectors
    z_mean = Dense(latent_units,
                   name="z_mean",
                   kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                   bias_initializer='zeros')(state)

    z_log_var = Dense(latent_units,
                          name="z_log_var",
                          activation=tf.math.softplus,
                          kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                          bias_initializer='zeros')(state)


    # sampling layer
    def sampling(args):
        """Sampling z from isotropic Gaussian"""
        z_mean, z_log_var = args

        eps = K.random_normal(shape=(K.shape(z_mean)[0], latent_units), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var)*eps

    # sampling z
    z = Lambda(sampling, name="z_sample")([z_mean, z_log_var])


    # Initial states for decoder is from z
    state_decoder_h = Dense(dec_units, activation='tanh', name="state_decoder_h")(z)
    state_decoder_c = Dense(dec_units, activation='tanh', name="state_decoder_c")(z)

    # Input to decoder lstm is concatenation of z and inputs
    z_repeated = RepeatVector(timesteps, name="z_repeated")(z)
    decoder_inputs = Input(shape=(timesteps, features), name="input_layer_decoder")
    decoder_train_input = Concatenate(axis=2, name="decoder_train_input")([decoder_inputs, z_repeated])

    # training decoder
    decoder_lstm0 = LSTM(dec_units,
                         unit_forget_bias=True,
                         dropout=dropout,
                         return_sequences=True,
                         name="decoder_lstm0")

    decoder_lstm1 = LSTM(dec_units,
                         unit_forget_bias=True,
                         return_sequences=True,
                         return_state=True,
                         name="decoder_lstm1")

    # intermediate outputs
    decoder_im_outputs = decoder_lstm0(decoder_train_input, initial_state=[state_decoder_h, state_decoder_c])
    decoder_outputs, _, _ = decoder_lstm1(decoder_im_outputs)

    decoder_dense = TimeDistributed(Dense(features, activation='softmax'), name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    # define inference encoder
    encoder_model = Model(encoder_inputs, [state_decoder_h, state_decoder_c, z])



    # define inference decoder
    decinf_input = Input(shape=(1, features+latent_units), name="decinf_inputs")

    # defime input states
    dec_input_state_c = Input(shape=(1, dec_units), name="d_input_state_c")
    dec_input_state_h = Input(shape=(1, dec_units), name="d_input_state_h")
    dec_input_states = [dec_input_state_h, dec_input_state_c]

    # intermediate lstm outputs
    decinf_im_outputs = decoder_lstm0(decinf_input,
                                      initial_state=dec_input_states)

    # output is a vector of sequences, needs reshaping
    decinf_im_outputs = Reshape((1,dec_units))(decinf_im_outputs)

    # lstm outputs
    decinf_outputs, state_h, state_c = decoder_lstm1(decinf_im_outputs)
    decoder_states = [state_h, state_c]

    # During inference the decoder output one element at the time
    decoder_inference_dense = Dense(features, activation='softmax', name="decoder_inference_dense")
    decinf_outputs = decoder_inference_dense(decinf_outputs)
    decoder_model = Model([decinf_input, dec_input_state_c, dec_input_state_h], [decinf_outputs] + decoder_states)


    def vae_loss(encoder_inputs, decoder_outputs):
        xent_loss = K.categorical_crossentropy(encoder_inputs, decoder_outputs)
        kl_loss = beta * kullback_leibler_divergence(encoder_inputs, decoder_outputs)
        loss = xent_loss + kl_loss
        return loss

    optimizer = Adam(lr=learning_rate, amsgrad=True, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=vae_loss, metrics=['acc'])

    return model, encoder_model, decoder_model


def define_MM(seq_len,
            vae_latent_units,
             hidden_units,
             number_mixtures):

    inputs = keras.layers.Input(shape=(seq_len,
                                       vae_latent_units), name='inputs')

    lstm1_out = keras.layers.LSTM(hidden_units, name='lstm1',
                                  return_sequences=True)(inputs)

    lstm2_out = keras.layers.LSTM(hidden_units, name='lstm2',
                                  return_sequences=True)(lstm1_out)

    mdn_out = keras.layers.TimeDistributed(mdn.MDN(vae_latent_units,
                                                   number_mixtures,
                                                   name='mdn_outputs'),
                                           name='td_mdn')(lstm2_out)

    model = keras.models.Model(inputs=inputs,
                               outputs=mdn_out)

    optimizer = Adam(clipnorm=1.,
                     lr=0.0001)

    model.compile(loss=mdn.get_mixture_loss_func(vae_latent_units,
                                                 number_mixtures),
                  optimizer=optimizer)

    decoder = keras.Sequential()

    decoder.add(keras.layers.LSTM(hidden_units, batch_input_shape=(1,1,vae_latent_units),
                                  return_sequences=True,
                                  stateful=True))

    decoder.add(keras.layers.LSTM(hidden_units, stateful=True))

    decoder.add(mdn.MDN(vae_latent_units, number_mixtures))

    decoder.compile(loss=mdn.get_mixture_loss_func(vae_latent_units,
                                                   number_mixtures), optimizer=keras.optimizers.Adam())

    return model, decoder
