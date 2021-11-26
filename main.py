import os, sys, time
import numpy as np
import tensorflow as tf
from model import Agent
from buffer import ReplayBuffer
from config import *
from maze import Maze
from vae_buffer import VAEBuffer
from vae_model import *
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

alpha = float(sys.argv[1])
env = Maze()
test_env = Maze()
n_ant = 1
state_space = 2
n_actions = 2

agent = Agent(sess,state_space,n_actions)
buff = ReplayBuffer(capacity,state_space,n_actions)
agent.init_update()

vae_0 = []
encoder_0 = []
vae_1 = []
encoder_1 = []
b_0 = []
b_1 = []
m = build_vae(state_space, 1)
encoder_0 = m[0]
vae_0 = m[2]
m = build_vae(state_space, 1)
encoder_1 = m[0]
vae_1 = m[2]
b_0 = VAEBuffer(vae_buffer_size,state_space)
b_1 = VAEBuffer(vae_buffer_size,state_space)


def test_agent():
	sum_reward = 0
	for m in range(10):
		o, d, ep_l = test_env.reset(), False, 0
		while not(d or (ep_l == max_ep_len)):
			a = agent.actor.predict(np.array([o]))[0]
			o, r, d = test_env.step(a)
			sum_reward += r
			ep_l += 1
	return sum_reward/10

f = open(sys.argv[1]+'-'+sys.argv[2]+'.txt', 'w')
obs = env.reset()
obs_t = []
s_g = np.ones((200,2))*0.1
sum_r = 0
t_ep  = 0
explore_flag = 1
while setps<max_steps:

	a = agent.actor.predict(np.array([obs]))[0]
	obs_t.append(obs)
	if explore_flag == 1:
		a = 2*np.random.rand(n_actions) - 1
	else:
		a = np.clip(a + 0.1*np.random.randn(n_actions),-1,1)
	next_obs, reward, terminated = env.step(a)
	setps += 1
	ep_len += 1
	sum_r += reward
	buff.add(obs, a, reward, next_obs, terminated)
	obs = next_obs

	if (terminated)|(ep_len == max_ep_len):

		if terminated == True:
			for o in obs_t:
				b_1.add(o)
		else:
			for o in obs_t:
				b_0.add(o)
		obs_t = []

		obs = env.reset()
		terminated = False
		ep_len = 0
		t_ep += 1
		explore_flag = 1 if np.random.rand()<0.4 else 0

		if np.random.rand() < ratio:
			generated_s = s_g[np.random.randint(200)]
			env.set_state(generated_s[0],generated_s[1])
			obs = env.get_state()

	if setps%4000==0:
		log_r = test_agent()
		f.write(str(sum_r/t_ep)+'	'+str(log_r)+'\n')
		f.flush()
		sum_r = 0
		t_ep = 0

		if b_0.buffer_len > train_len:

			s_0 = b_0.getBatch(M)
			s_1 = b_1.getBatch(M) if b_1.buffer_len > int(train_len/5) else b_0.getBatch(M)
			s_c = np.vstack([s_0,s_1])
			f_0 = encoder_0.predict(s_c,batch_size=M*2)[3]
			f_1 = encoder_1.predict(s_c,batch_size=M*2)[3]
			f_c = np.abs(f_0-f_1) if b_1.buffer_len > train_len else f_0
			p_rank = 1/(np.argsort(np.argsort(f_c)) + 1)**alpha
			index = np.random.choice(M*2,int(0.1*M*2), replace = False, p = p_rank/sum(p_rank))
			s_g = s_c[index]

	if (setps < 1000)|(setps%50!=0):
		continue

	for e in range(5):
		if b_1.buffer_len < 1000:
			break
		X, A, R, next_X, D = buff.getBatch(batch_size)

		Q_target = agent.Q_tot_tar.predict(next_X,batch_size = batch_size)
		Q_target = R + Q_target*gamma*(1 - D)

		agent.train_critic(X, A, Q_target)
		agent.train_actor(X)
		agent.update()

	if b_0.buffer_len > train_len:

		s_0 = b_0.getBatch(128)
		vae_0.fit(s_0, s_0, epochs=1, batch_size=128, verbose=0)

		if b_1.buffer_len > int(train_len/5):
			s_1 = b_1.getBatch(128)
			vae_1.fit(s_1, s_1, epochs=1, batch_size=128, verbose=0)

	


