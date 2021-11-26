import numpy as np
import copy


class Maze(object):
	def __init__(self):
		super(Maze, self).__init__()
		self.n_action = 2
		self.len_state = 2
		self.x = 0.1
		self.y = 0.1


	def reset(self):

		self.x = 0.1
		self.y = 0.1

		return self.get_state()

	def set_state(self,x,y):

		self.x = np.clip(x,0,1)
		self.y = np.clip(y,0,1)

		return self.get_state()

	def get_state(self):

		return np.array([self.x,self.y])


	def step(self,actions):

		reward = 0
		done = False
		x1 = np.clip(self.x + 0.2*np.clip(actions[0],-1,1),0,1)
		y1 = np.clip(self.y + 0.2*np.clip(actions[1],-1,1),0,1)

		if (x1>=0.2)&(x1<=0.4)&(y1>=0)&(y1<=0.8):
			pass
		elif (x1>=0.6)&(x1<=0.8)&(y1>=0.2)&(y1<=1.0):
			pass
		else:
			self.x = x1
			self.y = y1

		if (self.x-0.9)**2+(self.y-0.9)**2 < 0.02:
			reward = 10
			done = True

		return self.get_state(), reward, done
