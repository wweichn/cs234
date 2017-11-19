### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def compute_reward(P,V,state,action):
	reward = 0
	for i in range(len(P[state][action])):
		reward += P[state][action][i][0] * V[P[state][action][i][1]]      #p(s'|s,a) * V(s')
	return reward



def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""

	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	var = 5
	im_reward = np.zeros([nS,nA])
	for i in range(nS):
		for j in range(nA):
			im_reward[i][j] = P[i][j][0][2]
#    reward = compute_value(P, V, nS, action, state)  (probability, nextstate, reward, terminal)
#    im_reward = compute_im_reward(nS,nA,terminal_states)
	############################
	# YOUR IMPLEMENTATION HERE #
	for k in range(max_iteration):
		if var > tol:
			old_V = V.copy()
			for i in range(nS):
				opt_reward = -10
				for j in range(nA):      #r(s,a)
					reward = im_reward[i][j] + gamma * compute_reward(P,old_V,i,j)   #r(s,a) + cumulative reward
					if reward > opt_reward:
						opt_reward = reward
						V[i] = reward
						policy[i] = j
			var = np.linalg.norm(old_V - V)    #
	return V,policy



def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	im_reward = np.zeros([nS, nA])
	for i in range(nS):
		for j in range(nA):
			im_reward[i][j] = P[i][j][0][2]
	V_pi = np.zeros(len(policy))
	var = 5
	for k in range(max_iteration):
		if var > tol:
			V_pi_old = V_pi.copy()
			for state in range(len(policy)):
				action = policy[state]
				V_pi[state] = im_reward[state][action] + gamma * compute_reward(P,V_pi_old,state,action)
			var = np.linalg.norm(V_pi-V_pi_old)


	############################
	return V_pi

def policy_improvement(P, nS, nA, value_from_policy,policy,gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	im_reward = np.zeros([nS, nA])
	for i in range(nS):
		for j in range(nA):
			im_reward[i][j] = P[i][j][0][2]
	new_policy = np.zeros(nS)
	Q = np.zeros([nS,nA])
	for i in range(nS):
		for j in range(nA):
			temp =  im_reward[i][j] + gamma * compute_reward(P,value_from_policy,i,j)
			if temp > Q[i][j]:
				Q[i][j] = temp
				new_policy[i] = j    #change action taken

			#Policy improvement:
	############################
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
    # random policy till terminal state
	# caculcate policy value
	var = 5
	for k in range(max_iteration):
		V_new = policy_evaluation(P, nS, nA, policy)
		var = np.linalg.norm(V_new - V)
		if var > tol:
			policy = policy_improvement(P,nS,nA,V, policy)
		V = V_new.copy()




	############################
	return V, policy



def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print "Here is an example of state, action, reward, and next state"
	example(env)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	print V_vi,p_vi
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	print "!!!"
	print V_vi,p_vi

