import math
import gym
from frozen_lake import *
import numpy as np
import time
import random
from utils import *

def learn_Q_QLearning(env, num_episodes=10000, gamma = 0.99, lr = 0.1, e = 0.2, max_step=6):
	"""Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy(no decay)
	Feel free to reuse your assignment1's code
	Parameters
	----------
	env: gym.core.Environment
		Environment to compute Q function for. Must have nS, nA, and P as attributes.
	num_episodes: int 
		Number of episodes of training.
	gamma: float
		Discount factor. Number in range [0, 1)
	learning_rate: float
		Learning rate. Number in range [0, 1)
	e: float
		Epsilon value used in the epsilon-greedy method. 
	max_step: Int
		max number of steps in each episode

	Returns
	-------
	np.array
	  An array of shape [env.nS x env.nA] representing state-action values
	"""

	Q = np.zeros((env.nS, env.nA))
	########################################################
	#                     YOUR CODE HERE                   #
	########################################################
	for k in range(num_episodes):
		done = False
		state = env.reset()
		m = 0
		while m < max_step and not done:
		#	m = m + 1
			if random.random() < e or Q[state].max == 0:
				#              action = random.randint(0,env.nA - 1)   #pick an action randomly
				action = np.random.randint(env.nA)
			else:
				action = np.argmax(Q[state])
			next_state, reward, done, _ = env.step(action)
			Q[state][action] = (1 - lr) * Q[state][action] + lr * (reward + gamma * np.max(Q[next_state]))
			state = next_state
	########################################################
	#                     END YOUR CODE                    #
	########################################################
	return Q



def main():
	env = FrozenLakeEnv(is_slippery=False)
	Q = learn_Q_QLearning(env, num_episodes = 10000, gamma = 0.99, lr = 0.1, e = 0.4)
	render_single_Q(env, Q)


if __name__ == '__main__':
	main()
