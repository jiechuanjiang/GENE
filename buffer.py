import numpy as np
class ReplayBuffer(object):

	def __init__(self, buffer_size, state_space, n_action):
		self.buffer_size = buffer_size
		self.pointer = 0
		self.len = 0
		self.states = np.random.rand(self.buffer_size,state_space)
		self.actions = np.random.rand(self.buffer_size,n_action)
		self.rewards = np.random.rand(self.buffer_size,1)
		self.next_states = np.random.rand(self.buffer_size,state_space)
		self.dones = np.random.rand(self.buffer_size,1)
		

	def getBatch(self, batch_size):

		index = np.random.choice(self.len, batch_size, replace=False)
		return self.states[index], self.actions[index], self.rewards[index], self.next_states[index], self.dones[index]

	def add(self, state, action, reward, next_state, done):

		self.actions[self.pointer] = action
		self.rewards[self.pointer] = reward
		self.dones[self.pointer] = done
		self.states[self.pointer] = state
		self.next_states[self.pointer] = next_state
		self.dones[self.pointer] = done
		self.pointer = (self.pointer + 1)%self.buffer_size
		self.len = min(self.len + 1, self.buffer_size)