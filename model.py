import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Input, Dense, Concatenate, Add, Reshape
from keras.models import Model

def build_actor(num_features,n_actions):

	I1 = Input(shape = (num_features,))
	h1 = Dense(256,activation = 'relu')(I1) 
	h2 = Dense(256,activation = 'relu')(h1)
	V = Dense(n_actions,activation = 'tanh')(h2)
	model = Model(I1, V)

	return model

def build_critic(state_space,n_actions):

	Inputs = []
	Inputs.append(Input(shape = (n_actions,)))
	Inputs.append(Input(shape = (state_space,)))

	I = Concatenate(axis=1)(Inputs)
	h = Dense(32,activation = 'relu')(I)
	h = Dense(32,activation = 'relu')(h)
	q_total = Dense(1)(h)
	model = Model(Inputs, q_total)

	return model

def build_Q_tot(state_space,actor,critic):

	Inputs = Input(shape=[state_space])
	q_value = critic([actor(Inputs),Inputs])
	model = Model(Inputs, q_value)

	return model



class Agent(object):
	def __init__(self,sess,state_space,n_actions):
		super(Agent, self).__init__()
		self.sess = sess
		self.n_actions = n_actions
		self.state_space = state_space
		K.set_session(sess)

		self.actor = build_actor(self.state_space,self.n_actions)
		self.critic = build_critic(self.state_space,self.n_actions)
		self.Q_tot = build_Q_tot(self.state_space,self.actor,self.critic)
		
		self.actor_tar = build_actor(self.state_space,self.n_actions)
		self.critic_tar = build_critic(self.state_space,self.n_actions)
		self.Q_tot_tar = build_Q_tot(self.state_space,self.actor_tar,self.critic_tar)

		self.S = tf.placeholder(tf.float32,[None, self.state_space])
		self.A = tf.placeholder(tf.float32,[None, self.n_actions])
		self.label = tf.placeholder(tf.float32,[None, 1])
			
		self.opt_actor = tf.train.AdamOptimizer(0.001).minimize(-tf.reduce_mean(self.Q_tot(self.S)),var_list = self.actor.trainable_weights)
		self.opt_critic = tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.label - self.critic([self.A,self.S]))**2),var_list = self.critic.trainable_weights)
		
		self.opt_actor = tf.group(self.opt_actor)
		self.opt_critic = tf.group(self.opt_critic)

		self.soft_replace = tf.group([tf.assign(tar, 0.995*tar + (1 - 0.995)*main) for tar, main in zip(self.Q_tot_tar.trainable_weights, self.Q_tot.trainable_weights)])

		self.sess.run(tf.global_variables_initializer())

	def train_actor(self, X):

		dict_t = {}
		dict_t[self.S] = X
		return self.sess.run(self.opt_actor, feed_dict=dict_t)

	def train_critic(self, S, A, label):

		dict_t = {}
		dict_t[self.A] = A
		dict_t[self.S] = S
		dict_t[self.label] = label
		return self.sess.run(self.opt_critic, feed_dict=dict_t)

	def update(self):
		self.sess.run(self.soft_replace)


	def init_update(self):
		self.Q_tot_tar.set_weights(self.Q_tot.get_weights())
		

