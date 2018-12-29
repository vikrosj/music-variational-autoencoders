import numpy as np
from keras.utils import multi_gpu_model
from numpy import array
from keras.models import Model, Sequential
from keras.layers import Input, Flatten
from keras.layers import LSTM, Bidirectional, RepeatVector, Add, TimeDistributed, Reshape,Concatenate, Activation
from keras.layers import Dense, Lambda
import keras.backend as K
from keras.models import model_from_json
from keras.losses import kullback_leibler_divergence
import os
import tensorflow as tf
from my_classes import DataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping, TensorBoard
from keras.optimizers import Adam
from numba import jit

np.set_printoptions(threshold=10e6)

cardinality = 131
start_of_sequence = np.zeros(cardinality)
start_of_sequence[cardinality-1] = 1


def sample(preds, temperature=1.0):
    """ helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return probas

# generate target given source sequence
def predict_sequence(infenc,
                     infdec,
                     source,
                     latent_dim,
                     seq_len,
                     n_decoder_units,
                     timesteps,
                     temperature):


    # start of sequence input
    target_seq = start_of_sequence
    target_seq = target_seq.reshape(1, 1, cardinality)

    # collect predictions
    output = list()
    
    for i in range(seq_len // timesteps):
        
         # encode
        encoder_output = infenc.predict(source)

        # ensure 3 dimensions
        state_h = np.array(encoder_output[0]).reshape(1, 1, len(encoder_output[0][0]))
        state_c = np.array(encoder_output[1]).reshape(1, 1, len(encoder_output[0][0]))

        z = encoder_output[2]
        z = z.reshape(1, 1, latent_dim)

        # placeholder
        new_source = []

        for t in range(timesteps):
            # predict next char
            concatenated_input = np.concatenate((target_seq,z), axis=2).reshape(1,1,cardinality+latent_dim)

            yhat, h, c = infdec.predict([concatenated_input, state_h, state_c])

            # store prediction
            # yhat is a nested list, therefore slicing is necessary    
            sampled_output = sample(yhat[0,0,:], temperature)
            output.append(sampled_output)

            # update state
            state_h = h.reshape(1,1,n_decoder_units)
            state_c = c.reshape(1,1,n_decoder_units)

            # update target sequence
            target_seq = sampled_output.reshape(1,1,cardinality)

            new_source.append(target_seq)
        
        new_source = np.array(new_source).reshape(1,timesteps,cardinality)
        source = new_source 
        
    return array(output)

# generate target given source sequence
def mdn_predict_sequence(infdec,
                     z,
                     latent_dim,
                     seq_len,
                     temperature,
                     n_decoder_units):
    
   
    states_from_z = Sequential()
    states_from_z.add(Dense(n_decoder_units, activation='tanh'))
    states_from_z.compile(optimizer="adam")
    
    # Initial states for decoder is from z
    state_decoder_h = states_from_z.predict(z)
    state_decoder_c = states_from_z.predict(z)

    state_h = state_decoder_h.reshape(1, 1, n_decoder_units)
    state_c = state_decoder_c.reshape(1, 1, n_decoder_units)
    
    z = z.reshape(1, 1, latent_dim)

    # start of sequence input
    target_seq = start_of_sequence
    target_seq = target_seq.reshape(1, 1, cardinality)

    # collect predictions
    output = list()

    for t in range(seq_len):
        # predict next char
        concatenated_input = np.concatenate((target_seq,z), axis=2).reshape(1,1,cardinality+latent_dim)

        yhat, h, c = infdec.predict([concatenated_input, state_h, state_c])

        # store prediction
        # yhat is a nested list, therefore slicing is necessary    
        sampled_output = sample(yhat[0,0,:], temperature)
        output.append(sampled_output)
        
        # update state
        state_h = h.reshape(1,1,n_decoder_units)
        state_c = c.reshape(1,1,n_decoder_units)
        
        # update target sequence
        target_seq = sampled_output.reshape(1,1,cardinality)
        
    return array(output)
    

# generate target given source sequence
def predict_sequence_from_zs(infdec,
                     z,
                     latent_dim,
                     seq_len,
                     temperature,
                     n_decoder_units):
    
   
    
    states_from_z = Sequential()
    states_from_z.add(Dense(n_decoder_units, activation='tanh'))
    states_from_z.compile(optimizer="adam")
    
    # Initial states for decoder is from z
    state_decoder_h = states_from_z.predict(z)
    state_decoder_c = states_from_z.predict(z)

    state_h = state_decoder_h.reshape(1, 1, n_decoder_units)
    state_c = state_decoder_c.reshape(1, 1, n_decoder_units)
    
    z = z.reshape(1, 1, latent_dim)

    # start of sequence input
    target_seq = start_of_sequence
    target_seq = target_seq.reshape(1, 1, cardinality)

    # collect predictions
    output = list()

    for t in range(seq_len):
        # predict next char
        concatenated_input = np.concatenate((target_seq,z), axis=2).reshape(1,1,cardinality+latent_dim)

        yhat, h, c = infdec.predict([concatenated_input, state_h, state_c])

        # store prediction
        # yhat is a nested list, therefore slicing is necessary    
        sampled_output = sample(yhat[0,0,:], temperature)
        output.append(sampled_output)
        
        # update state
        state_h = h.reshape(1,1,n_decoder_units)
        state_c = c.reshape(1,1,n_decoder_units)
        
        # update target sequence
        target_seq = sampled_output.reshape(1,1,cardinality)
    return array(output)

# generate target given source sequence
def predict_z(infenc,source,latent_dim):
    
    # encode
    encoder_output = infenc.predict(source)
    
    z = encoder_output[2]
    z = z.reshape(1, 1, latent_dim)
    
    return z

# returns train, inference_encoder and inference_decoder models
def define_models(n_encoder_units, 
                  n_decoder_units, 
                  latent_dim, 
                  timesteps, 
                  n_features, 
                  learning_rate,
                  dropout,
                  beta,
                  epsilon_std):
    
    # define training encoder
    encoder_inputs = Input(shape=(timesteps, n_features), name="encoder_inputs")
    
    encoder0 = Bidirectional(LSTM(n_encoder_units, 
                                  dropout=dropout,
                                  return_sequences=True, 
                                  unit_forget_bias=True,
                                  name="bidirectional_encoder0"))
                             
    encoder1 = Bidirectional(LSTM(n_encoder_units,
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
    z_mean = Dense(latent_dim, 
                   name="z_mean",
                   kernel_initializer=tf.random_normal_initializer(stddev=0.001), 
                   bias_initializer='zeros')(state)
    
    z_log_var = Dense(latent_dim, 
                          name="z_log_var",
                          activation=tf.math.softplus,
                          kernel_initializer=tf.random_normal_initializer(stddev=0.001), 
                          bias_initializer='zeros')(state)
    
    
    # sampling layer
    def sampling(args):
        """Sampling z from isotropic Gaussian"""
        z_mean, z_log_var = args

        eps = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var)*eps

    # sampling z
    z = Lambda(sampling, name="z_sample")([z_mean, z_log_var])

        
    # Initial states for decoder is from z
    state_decoder_h = Dense(n_decoder_units, activation='tanh', name="state_decoder_h")(z)
    state_decoder_c = Dense(n_decoder_units, activation='tanh', name="state_decoder_c")(z)

    # Input to decoder lstm is concatenation of z and inputs
    z_repeated = RepeatVector(timesteps, name="z_repeated")(z)
    decoder_inputs = Input(shape=(timesteps, n_features), name="input_layer_decoder")
    decoder_train_input = Concatenate(axis=2, name="decoder_train_input")([decoder_inputs, z_repeated])

    # training decoder
    decoder_lstm0 = LSTM(n_decoder_units,
                         unit_forget_bias=True,
                         dropout=dropout,
                         return_sequences=True,
                         name="decoder_lstm0")
    
    decoder_lstm1 = LSTM(n_decoder_units,
                         unit_forget_bias=True,
                         return_sequences=True,
                         return_state=True,
                         name="decoder_lstm1")
    
    # intermediate outputs
    decoder_im_outputs = decoder_lstm0(decoder_train_input, initial_state=[state_decoder_h, state_decoder_c])
    decoder_outputs, _, _ = decoder_lstm1(decoder_im_outputs)

    decoder_dense = TimeDistributed(Dense(n_features, activation='softmax'), name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    # define inference encoder
    encoder_model = Model(encoder_inputs, [state_decoder_h, state_decoder_c, z])
    
    
    
    # define inference decoder
    decinf_input = Input(shape=(1, n_features+latent_dim), name="decinf_inputs")

    # defime input states
    dec_input_state_c = Input(shape=(1, n_decoder_units), name="d_input_state_c")
    dec_input_state_h = Input(shape=(1, n_decoder_units), name="d_input_state_h")
    dec_input_states = [dec_input_state_h, dec_input_state_c]

    # intermediate lstm outputs
    decinf_im_outputs = decoder_lstm0(decinf_input,
                                      initial_state=dec_input_states)
    
    # output is a vector of sequences, needs reshaping
    decinf_im_outputs = Reshape((1,n_decoder_units))(decinf_im_outputs)
    
    # lstm outputs
    decinf_outputs, state_h, state_c = decoder_lstm1(decinf_im_outputs)
    decoder_states = [state_h, state_c]

    # During inference the decoder output one element at the time
    decoder_inference_dense = Dense(n_features, activation='softmax', name="decoder_inference_dense")
    decinf_outputs = decoder_inference_dense(decinf_outputs)
    decoder_model = Model([decinf_input, dec_input_state_c, dec_input_state_h], [decinf_outputs] + decoder_states)


    def vae_loss(encoder_inputs, decoder_outputs):
        xent_loss = K.categorical_crossentropy(encoder_inputs, decoder_outputs)
        kl_loss = beta * kullback_leibler_divergence(encoder_inputs, decoder_outputs)
        loss = xent_loss + kl_loss
        return loss
    
    optimizer = Adam(lr=learning_rate, amsgrad=True, clipnorm=1.0)
    
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=optimizer, loss=vae_loss, metrics=['acc'])     

    return model, encoder_model, decoder_model

        
    
