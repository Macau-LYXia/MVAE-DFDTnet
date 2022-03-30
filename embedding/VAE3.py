# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:54:38 2022

@author: DELL
"""
from keras import regularizers
from keras.layers import Input, Dense, Lambda,Dropout,concatenate
from keras.models import Model
from keras import backend as K
from keras.losses import  binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os 

np.random.seed(116)

#original_dim = 732

intermediate_dim = 500
intermediate_dim1 = 200
#intermediate_dim2 = 100
latent_dim = 100
#latent_dim = 100

decoded_dp = []
decoded_dp1 = []
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon


def FeatureExtraction_d(original_dim,data):
    
    inputs = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
    h = Dropout(0.2)(h)
    h = Dense(intermediate_dim1, activation='relu',activity_regularizer=regularizers.l1(10e-5))(h)
    h = Dropout(0.2)(h)
  #  h = Dense(intermediate_dim2, activation='relu',activity_regularizer=regularizers.l1(10e-5))(h)
  #  h = Dropout(0.2)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    # 解码层，也就是生成器部分
    #x = Dense(intermediate_dim2, activation='relu',activity_regularizer=regularizers.l1(10e-5))(latent_inputs)
    #x = Dropout(0.2)(x)
    x = Dense(intermediate_dim1, activation='relu',activity_regularizer=regularizers.l1(10e-5))(latent_inputs)
    x = Dropout(0.2)(x)
    x = Dense(intermediate_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(x)
    x = Dropout(0.2)(x)

    outputs = Dense(732, activation='sigmoid')(x)
    decoder1 = Model(latent_inputs, outputs, name='decoder1')
    decoder2 = Model(latent_inputs, outputs, name='decoder2')
    decoder3 = Model(latent_inputs, outputs, name='decoder3')
    decoder4 = Model(latent_inputs, outputs, name='decoder4')
    decoder5 = Model(latent_inputs, outputs, name='decoder5')
    decoder6 = Model(latent_inputs, outputs, name='decoder6')
    decoder7 = Model(latent_inputs, outputs, name='decoder7')
    decoder8 = Model(latent_inputs, outputs, name='decoder8')      
    decoder9 = Model(latent_inputs, outputs, name='decoder9')
    outputs1 = decoder1(encoder(inputs)[2])
    outputs2 = decoder2(encoder(inputs)[2])
    outputs3 = decoder3(encoder(inputs)[2])
    outputs4 = decoder4(encoder(inputs)[2])
    outputs5 = decoder5(encoder(inputs)[2])
    outputs6 = decoder6(encoder(inputs)[2])
    outputs7 = decoder7(encoder(inputs)[2])
    outputs8 = decoder8(encoder(inputs)[2])
    outputs9 = decoder9(encoder(inputs)[2])
    outputs =  concatenate([outputs1,outputs2,
                              outputs3,outputs4,
                              outputs5,outputs6,
                              outputs7,outputs8,
                              outputs9],axis =1) 
   
    # 建立模型
    vae = Model(inputs,outputs, name='vae_mlp')
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    
    history = vae.fit(data,data,
              epochs=1,
              batch_size=100,
              shuffle=True,
              validation_data=(data, data))
    
    val_loss = history.history['val_loss']
    encoder = Model(inputs, z_mean, name='encoder')
    feature = encoder.predict(data, batch_size=100)
    decoded_img1 = decoder1.predict(feature)
    decoded_img2 = decoder2.predict(feature)
    decoded_img3 = decoder3.predict(feature)
    decoded_img4 = decoder4.predict(feature)
    decoded_img5 = decoder5.predict(feature)
    decoded_img6 = decoder6.predict(feature)
    decoded_img7 = decoder7.predict(feature)
    decoded_img8 = decoder8.predict(feature)
    decoded_img9 = decoder9.predict(feature)
    decoded_dp.append(decoded_img1)
    decoded_dp.append(decoded_img2)
    decoded_dp.append(decoded_img3)
    decoded_dp.append(decoded_img4)
    decoded_dp.append(decoded_img5)
    decoded_dp.append(decoded_img6)
    decoded_dp.append(decoded_img7)
    decoded_dp.append(decoded_img8)
    decoded_dp.append(decoded_img9)
    return  feature, decoded_dp

def FeatureExtraction_p(original_dim,data):
    inputs = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
    h = Dropout(0.2)(h)
    h = Dense(intermediate_dim1, activation='relu',activity_regularizer=regularizers.l1(10e-5))(h)
    h = Dropout(0.2)(h)
  #  h = Dense(intermediate_dim2, activation='relu',activity_regularizer=regularizers.l1(10e-5))(h)
  #  h = Dropout(0.2)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    # 解码层，也就是生成器部分
    #x = Dense(intermediate_dim2, activation='relu',activity_regularizer=regularizers.l1(10e-5))(latent_inputs)
    #x = Dropout(0.2)(x)
    x = Dense(intermediate_dim1, activation='relu',activity_regularizer=regularizers.l1(10e-5))(latent_inputs)
    x = Dropout(0.2)(x)
    x = Dense(intermediate_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1915, activation='sigmoid')(x)
    decoder1 = Model(latent_inputs, outputs, name='decoder1')
    decoder2 = Model(latent_inputs, outputs, name='decoder2')
    decoder3 = Model(latent_inputs, outputs, name='decoder3')
    decoder4 = Model(latent_inputs, outputs, name='decoder4')
    decoder5 = Model(latent_inputs, outputs, name='decoder5')
    decoder6 = Model(latent_inputs, outputs, name='decoder6')
    outputs1 = decoder1(encoder(inputs)[2])
    outputs2 = decoder2(encoder(inputs)[2])
    outputs3 = decoder3(encoder(inputs)[2])
    outputs4 = decoder4(encoder(inputs)[2])
    outputs5 = decoder5(encoder(inputs)[2])
    outputs6 = decoder6(encoder(inputs)[2])
    outputs =  concatenate([outputs1,outputs2,
                              outputs3,outputs4,
                              outputs5,outputs6],axis =1)
    # 建立模型
    vae = Model(inputs,outputs, name='vae_mlp')
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    
    history = vae.fit(data,data,
              epochs=1,
              batch_size=100,
              shuffle=True,
              validation_data=(data, data))
    
    val_loss = history.history['val_loss']
    encoder = Model(inputs, z_mean, name='encoder')
    feature = encoder.predict(data, batch_size=100)
    

    decoded_img11 = decoder1.predict(feature)
    decoded_img22 = decoder2.predict(feature)
    decoded_img33 = decoder3.predict(feature)
    decoded_img44 = decoder4.predict(feature)
    decoded_img55 = decoder5.predict(feature)
    decoded_img66 = decoder6.predict(feature)
     
    decoded_dp1.append(decoded_img11)
    decoded_dp1.append(decoded_img22)
    decoded_dp1.append(decoded_img33)
    decoded_dp1.append(decoded_img44)
    decoded_dp1.append(decoded_img55)
    decoded_dp1.append(decoded_img66)
    return  feature, decoded_dp1