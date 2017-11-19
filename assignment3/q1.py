import math
import gym
from frozen_lake import *
import numpy as np
import time
import math
from utils import *


def rmax(env, gamma, m, R_max, epsilon, num_episodes, max_step = 6):
	"""Learn state-action values using the Rmax algorithm

	Args:
	----------
	env: gym.core.Environment
		Environment to compute Q function for. Must have nS, nA, and P as
		attributes.
	m: int
		Threshold of visitance
	R_max: float 
		The estimated max reward that could be obtained in the game
	epsilon: 
		accuracy paramter
	num_episodes: int 
		Number of episodes of training.
	max_step: Int
		max number of steps in each episode

	Returns
	-------
	np.array
	  An array of shape [env.nS x env.nA] representing state-action values
	"""

	Q = np.ones((env.nS, env.nA)) * R_max / (1 - gamma)
	R = np.zeros((env.nS, env.nA))
	nSA = np.zeros((env.nS, env.nA))
	nSASP = np.zeros((env.nS, env.nA, env.nS))
	########################################################
	#                   YOUR CODE HERE                     #
	########################################################
	for k in range(num_episodes):
		done = False
		s = env.reset()
		for _ in range(max_step):
			if done:
				break
			if np.max(Q[s]) == R_max / (1 - gamma):
				a = np.random.randint(env.nA)
			else:
				a = np.argmax(Q[s])
			next_s, reward, done, _ = env.step(a)
			if nSA[s][a] < m:
				nSA[s][a] = nSA[s][a] + 1
				R[s][a] = R[s][a] + reward
				nSASP[s][a][next_s] = nSASP[s][a][next_s] + 1
				if nSA[s][a] == m :
					up = int(np.ceil(np.log(1 / (epsilon * (1 - gamma))) / (1 - gamma)))
					for i in range(up):
						for s in range(env.nS):
							for a in range(env.nA):
								if nSA[s][a] >= m :
									v = R[s][a] / nSA[s][a]
									for s_ in range(env.nS):
										v = v + gamma * ((nSASP[s][a][s_]) / np.sum(nSASP[s][a])) * np.max(Q[s_])
										Q[s][a] = v

			s = next_s


	########################################################
	#                    END YOUR CODE                     #
	########################################################
	return Q


def main():
	env = FrozenLakeEnv(is_slippery=False)
	print env.__doc__
	Q = rmax(env, gamma = 0.99, m=10, R_max = 1, epsilon = 0.1, num_episodes = 1000)
	render_single_Q(env, Q)


if __name__ == '__main__':
	print "haha"
	main()