import gym
import random
import numpy as np
from IPython.display import clear_output


env = gym.make("Taxi-v2").env

def random_learning():
	env.s = 328  # set environment to illustration's state
	env.render()
	epochs = 0
	penalties, reward = 0, 0

	frames = [] # for animation

	done = False

	while not done:
		action = env.action_space.sample()
		
		state, reward, done, info = env.step(action)

		if reward == -10: #Wrong dropoff/pickup
		    penalties += 1
		
		# Put each rendered frame into dict for animation
		frames.append({
		    'frame': env.render(mode='ansi'),
		    'state': state,
		    'action': action,
		    'reward': reward
		    }
		)

		epochs += 1
		
	env.render()    
		
		
	print("Timesteps taken: {}".format(epochs))
	print("Penalties incurred: {}".format(penalties))
	
q_table = np.zeros([env.observation_space.n, env.action_space.n])
def q_learning():
	
	alpha = 0.1
	gamma = 0.6
	epsilon = 0.1

	# For plotting metrics
	all_epochs = []
	all_penalties = []

	for i in range(1, 10001):
		state = env.reset()

		epochs, penalties, reward, = 0, 0, 0
		done = False
		
		while not done:
		    if random.uniform(0, 1) < epsilon:
		        action = env.action_space.sample() # Explore action space
		    else:
		        action = np.argmax(q_table[state]) # Exploit learned values

		    next_state, reward, done, info = env.step(action) 
		    
		    old_value = q_table[state, action]
		    next_max = np.max(q_table[next_state])
		    
		    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
		    q_table[state, action] = new_value

		    if reward == -10:
		        penalties += 1

		    state = next_state
		    epochs += 1
		    
		if i % 1000 == 0:
		    clear_output(wait=True)
		    #print("Episode: {0}".format(i))

	print("Training finished.\n")
	
	env.s=328
	env.render()
	epochs, penalties, rewards=0,0,0
	done=False
	
	while not done:
		action = np.argmax(q_table[env.s])
		state, reward, done, info = env.step(action)

		if reward == -10:
			penalties += 1

		epochs += 1
	env.render()    
		
		
	print("Timesteps taken: {}".format(epochs))
	print("Penalties incurred: {}".format(penalties))	
	
if __name__=="__main__":
	random_learning()
	q_learning()	



