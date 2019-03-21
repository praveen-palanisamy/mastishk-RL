#!/usr/bin/env python
import csv 
import gym 
import gym_advection
import random 
import torch 
from torch.autograd import Variable 
import numpy as np 
from utils.decay_schedule import LinearDecaySchedule 
from function_approximator.perceptron import SLP
import progressbar
import sys

env = gym.make("Advection-AdG-v0")
MAX_NUM_EPISODES = int(sys.argv[1])
MAX_STEPS_PER_EPISODE = 1500
class Shallow_Q_Learner(object):
	def __init__(self, state_shape, action_shape, neurons=10, learning_rate=0.005, gamma=0.98):
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.gamma = gamma # Agent's discount factor
		self.learning_rate = learning_rate # Agent's Q-learning rate
		# self.Q is the Action-Value function. This agent represents Q using a Neural Network.
		self.neurons=int(neurons)
		self.Q = SLP(state_shape, action_shape, self.neurons, device=torch.device("cpu"))
		self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
		# self.policy is the policy followed by the agent. This agents follows
		# an epsilon-greedy policy w.r.t it's Q estimate.
		self.policy = self.epsilon_greedy_Q
		self.epsilon_max = 1.0
		self.epsilon_min = 0.05
		self.epsilon_decay=LinearDecaySchedule(initial_value=self.epsilon_max, final_value=self.epsilon_min, max_steps= 0.5 * MAX_NUM_EPISODES	* MAX_STEPS_PER_EPISODE)
		self.step_num = 0
		
	def get_action(self, observation):
		return self.policy(observation)
		
	def epsilon_greedy_Q(self, observation):
		# Decay Epsilion/exploratin as per schedule
		
		if random.random() < self.epsilon_decay(self.step_num):
			action = random.choice([i for i in range(self.action_shape)])			
		else:
			action = np.argmax(self.Q(observation).data.numpy())
		
		#action = np.argmax(self.Q(observation).data.numpy())
		return action
		
	def learn(self, s, a, r, s_next):
		td_target = r + self.gamma * torch.max(self.Q(s_next))
		td_error = torch.nn.functional.mse_loss(self.Q(s)[a], td_target)		
		# Update Q estimate
		#self.Q(s)[a] = self.Q(s)[a] + self.learning_rate * td_error
		self.Q_optimizer.zero_grad()
		td_error.backward()
		self.Q_optimizer.step()		
		
if __name__ == "__main__":
	observation_shape = env.observation_space.shape
	action_shape = env.action_space.n
	agent = Shallow_Q_Learner(observation_shape, action_shape, sys.argv[2])
	first_episode = True
	episode_rewards = list()
	
	agent_file=open("trained_models/advection_AdG_v0_"+str(MAX_NUM_EPISODES)+"_"+str(agent.neurons)+".csv", 'w')
	agent_writer=csv.writer(agent_file, delimiter=',')
	agent_writer.writerow(['episode', 'reward', 'normalized time', 'steps'])
	bar = progressbar.ProgressBar(maxval=MAX_NUM_EPISODES, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	#episodes=dict()
	for episode in range(MAX_NUM_EPISODES):
		#episodes[episode]=list()
		obs = env.reset()
		cum_reward = 0.0 # Cumulative reward
		step=0
		episode_file=open("trained_models/analysis/"+str(episode)+".csv", 'w')
		episode_writer=csv.writer(episode_file, delimiter=',')
		episode_writer.writerow(['dt', 'error'])
		for step in range(MAX_STEPS_PER_EPISODE):
			# env.render()
			action = agent.get_action(obs)
			next_obs, reward, done, info = env.step(action)
			episode_writer.writerow(info)
			#episodes[episode].append(info)
			if next_obs[0]>0.11:
				break
			agent.learn(obs, action, reward, next_obs)
			cum_reward += reward
			obs = next_obs
			if done is True:
				if first_episode: # Initialize max_reward at the end of first episode
					max_reward = cum_reward
					first_episode = False
				episode_rewards.append(cum_reward)
				if cum_reward > max_reward:
					max_reward = cum_reward
				#print("\nEpisode#{} ended in {} steps. reward ={} ;	mean_reward={} best_reward={}".format(episode, step+1, cum_reward,np.mean(episode_rewards), max_reward))
				break
		print("Episode={0}, reward={1}, normalized time={2}, steps={3}".format(episode, cum_reward, env.tc/env.tmax, step))
		agent_writer.writerow([episode, cum_reward, env.tc/env.tmax, step])
		bar.update(episode+1)
		#episodes[episode]=list(map(list, zip(*episodes[episode])))
	#print(episodes)
	bar.finish()
	env.close()			
				
				
