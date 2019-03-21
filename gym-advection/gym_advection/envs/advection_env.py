"""
This problem is designed by Adwaith Gupta, 2019
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

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
        1   Current time step (del_t)   -inf         inf
        
    Actions:
        Type: Discrete(3)
        Num	Action
        0   Do not change del_t
        1	Increase del_t by 0.00005
        2   Reduce del_t by 0.00005
        
    Reward:
        +1 if the error decreases (or stays the same) at any point
        -1 if the error increases at any point
        +100 if the normalized max time > 0.96 is attained

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
        high=np.array([0.11, 0.10])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64) 
        # Modify the action space, and dimension according to your custom environment's needs
        self.action_space = gym.spaces.Discrete(3)
        
        self.state=None
        
        self.N = 1000 # dicretization in x
        self.tmax = 10
        self.xmin = 0
        self.xmax = 10
        self.v = 1 # velocity
        self.xc = 0.25        

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
        old_error, self.dt = state
        if action==1:
        	self.dt+=0.00005
        if action==2:
            self.dt-=0.00005
        new_error=self.lax_wendroff()	
        #print(old_error, new_error)   
        self.state=np.array([new_error, self.dt])	 
        success = bool(new_error<0.11 and (self.tc/self.tmax)>0.96)
        reward=None
        if not success:
            reward=1.0 if new_error<=old_error else -1.0
        else:
            reward=100.0
       
        return self.state, reward, success, [self.dt/0.009, new_error/0.11]

    
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
        #self.dt=random.uniform(0.0088, 0.0092)
        self.dt=0.009
        self.initializeDomain()
        self.initializeU()
        #self.initializeParams()
        self.state = [float("inf"), self.dt]
        self.tc=0.0
        return np.array(self.state)

    def render(self, mode='human', close=False):
        """

        :param mode:
        :return:
        """
        return

    def lax_wendroff(self):
        
        for j in range(self.N+2):
            self.initializeParams()
            self.unp1[j] = self.u[j] + (self.v**2*self.dt**2/(2*self.dx**2))*(self.u[j+1]-2*self.u[j]+self.u[j-1]) - self.alpha*(self.u[j+1]-self.u[j-1])
                
        self.u = self.unp1.copy()
            
        # Periodic boundary conditions
        self.u[0] = self.u[self.N+1]
        self.u[self.N+2] = self.u[1]
            
        uexact = np.exp(-200*(self.x-self.xc-self.v*self.tc)**2)
        error=max(abs(uexact-self.u))
        self.tc+=self.dt
        return error
        
    def initializeDomain(self):
        self.dx = (self.xmax - self.xmin)/self.N
        self.x = np.arange(self.xmin-self.dx, self.xmax+(2*self.dx), self.dx)   
        
    def initializeU(self):
        u0 = np.exp(-200*(self.x-self.xc)**2)
        self.u = u0.copy()
        self.unp1 = u0.copy()        
        
    def initializeParams(self):
        self.alpha = self.v*self.dt/(2*self.dx)              

'''   
env=AdvectionEnv()
#print(env.observation_space.low)
print(env.action_space)
print(env.reset())
for i in range(10):
    print(env.step(0))
'''
