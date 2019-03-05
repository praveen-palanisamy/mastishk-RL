"""
This problem is designed by Adwaith Gupta, 2019
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class AdvectionEnv(gym.Env):
    """
    Description:
        Advection equation defines propogation of a disturbace at a certain propogation speed. The equation is given as:
            
            du/dt + a du/dx=0
            
            where,
            u----> velocity
            t----> time
            a----> propagation (information) speed
            x----> space
            d represents a partial derivative, for example du/dt means partial derivative of velocity w.r.t time
        
        The propagation is defined as a Gaussian wave in this case.
            
        There are many numerical schemes to solve the advection equation for the value of u. But all suffer with numerical errors. The task of the agent is to maintain the numerical solution (Lax-Wendroff) as close as possible (error<0.11) to the exact known solution for the longest time possible (normalized max time) by adjusting the time step (del_t) in the numerical scheme.

    Source:
        This problem is designed by Adwaith Gupta, 2019

    Observation: 
        Type: Box(2)
        Num	Observation                 Min         Max
        0   Error                       0.0         0.11 
        1   Current time step (del_t)   0.0         1.00 
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Do not change del_t
        1	Reduce del_t by 1e-8
        
    Reward:
        +1 if the error decreases (or stays the same) at any point
        -1 if the error increases at any point
        +100 if the normalized max time > 0.95 is attained

    Starting State:
        All observations are assigned a uniform random value as follows:
            Space step (del_x) E [800, 1200]
            Propogation speed (a) E [0.8, 1.2]

    Episode Termination:
        Error > 0.11
        Episode length is greater than 200
        Considered solved when the normalized max time > 0.95
    """
    
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "0.0.1"
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        low=np.array([0.0,0.0])
        high=np.array([0.11, 1.0])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64) 
        # Modify the action space, and dimension according to your custom environment's needs
        self.action_space = gym.spaces.Discrete(2)
        
        self.state=None

    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Next observation from the environment at the current action
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        old_error, del_t = state
        if action==0:
            pass
        if action==1:
            del_t-=1e-8
        new_error, norm_time=self.lax_wendroff(del_t)	   
        self.state=np.array([error, del_t])	 
 
        done = bool(new_error<0.11 and norm_time>0.95)
       
        
        reward=None
        if not done:
            reward=1.0 if new_error<=old_error else -1.0
        else:
            reward=100.0
        	
        return self.state, reward, done, {}

    
    def lax_wendroff(self, del_t):
        pass 

    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        # Implement your reset method here
        # return observation
        self.state = [float("inf"), random.uniform(0.0, 1.0)]
        return np.array(self.state)

    def render(self, mode='human', close=False):
        """

        :param mode:
        :return:
        """
        return
       
env=AdvectionEnv()
print(env.observation_space.low)
print(env.action_space)

