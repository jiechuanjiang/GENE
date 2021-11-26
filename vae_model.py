import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import mse
from keras.optimizers import Adam

def sampling(args):
    mu, log_sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.exp(0.5 * log_sigma) * epsilon

def vae_loss(log_sigma, mu):

    def my_loss(y_true, y_pred):

        recon = mse(y_true, y_pred)
        kl = 0.5 *K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
        
        return recon + 0.001*kl
        
    return my_loss

def kl_loss(log_sigma, mu):

    def my_kl_loss(y_true, y_pred):
        return 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
        
    return my_kl_loss

def recon_loss(y_true, y_pred):

    return mse(y_true, y_pred)

def build_vae(state_space, latent_dim):

    s = Input(shape=(state_space, ))
    h = Dense(64, activation='relu')(s)
    h = Dense(64, activation='relu')(h)

    mu = Dense(latent_dim)(h)
    log_sigma = Dense(latent_dim)(h)

    z = Lambda(sampling)([mu, log_sigma])
    p = Lambda(lambda x: K.exp(K.sum(-0.5*(x**2  + np.log(2*np.pi)), axis=1)))(mu)

    encoder = Model(s, [mu, log_sigma, z, p])

    latent_inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu')(latent_inputs)
    h = Dense(64, activation='relu')(h)
    outputs = Dense(state_space,activation='tanh')(h)

    decoder = Model(latent_inputs, outputs)

    vae = Model(s, decoder(encoder(s)[2]))
    vae.compile(optimizer=Adam(0.0003),loss=vae_loss(log_sigma, mu), metrics = [kl_loss(log_sigma, mu), recon_loss])

    return encoder, decoder, vae



