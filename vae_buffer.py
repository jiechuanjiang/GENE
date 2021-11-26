import numpy as np
class VAEBuffer(object):

	def __init__(self, buffer_size, state_space):
		self.buffer_size = buffer_size
		self.buffer_len = 0
		self.pointer = 0
		self.states = np.zeros((self.buffer_size,state_space))

	def getBatch(self, batch_size):

		index = np.random.choice(self.buffer_len, batch_size, replace=False)
		return self.states[index]

	def add(self, state):

		self.states[self.pointer] = state
		self.pointer = (self.pointer + 1)%self.buffer_size
		self.buffer_len = min(self.buffer_len+1,self.buffer_size)
