"""
This code solves the advection equation
    U_t + vU_x = 0

over the spatial domain of 0 <= x <= 1 that is discretized 
into 103 nodes, with dx=0.01, using the Lax-Wendroff scheme 
in Eq. (18.20) for an initial profile of a Gaussian curve, 
defined by 
    U(x,t) = exp(-200*(x-xc-v*t).^2)

where xc=0.25 is the center of the curve at t=0.

The periodic boundary conditions are applied either end of the domain.
The velocity is v=1. The solution is iterated until t=1.5 seconds.
"""

import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import progressbar
import random


class LaxWendroff:
    
    def __init__(self, N, dt, decay_trigger_time=0):
        self.N = N # number of nodes
        self.tmax = 10
        self.xmin = 0
        self.xmax = 10
        self.dt = dt # timestep
        self.v = 1 # velocity
        self.xc = 0.25
        self.decay_trigger_time=decay_trigger_time
        self.initializeDomain()
        self.initializeU()
        self.initializeParams()
        
        
    def initializeDomain(self):
        self.dx = (self.xmax - self.xmin)/self.N
        self.x = np.arange(self.xmin-self.dx, self.xmax+(2*self.dx), self.dx)
        
        
    def initializeU(self):
        u0 = np.exp(-200*(self.x-self.xc)**2)
        self.u = u0.copy()
        self.unp1 = u0.copy()
        
        
    def initializeParams(self):
        self.nsteps = round(self.tmax/self.dt)
        self.alpha = self.v*self.dt/(2*self.dx)
        
        
    def solve_and_plot(self):
        tc = 0
        
        #for i in range(self.nsteps):
        while tc<self.tmax:    
            plt.clf()
            error=0
            # The Lax-Wendroff scheme, Eq. (18.20)
            for j in range(self.N+2):
                self.unp1[j] = self.u[j] + (self.v**2*self.dt**2/(2*self.dx**2))*(self.u[j+1]-2*self.u[j]+self.u[j-1]) \
                - self.alpha*(self.u[j+1]-self.u[j-1])
                
            self.u = self.unp1.copy()
            
            # Periodic boundary conditions
            self.u[0] = self.u[self.N+1]
            self.u[self.N+2] = self.u[1]
            
            uexact = np.exp(-200*(self.x-self.xc-self.v*tc)**2)
            #error=self.MSE(uexact, self.u)
            error=self.max_error(uexact, self.u)
            #print(tc, error, self.dt)
            """            
            plt.plot(self.x, uexact, 'r', label="Exact solution")
            plt.plot(self.x, self.u, 'bo-', label="Lax-Wendroff")
            plt.axis((self.xmin-0.12, self.xmax+0.12, -0.2, 1.4))
            plt.grid(True)
            plt.xlabel("Distance (x)")
            plt.ylabel("u")
            plt.legend(loc=1, fontsize=12)
            plt.suptitle("Time = %1.3f" % (tc+self.dt))
            plt.pause(0.01)
            """
            tc += self.dt
            #if tc>self.decay_trigger_time:
            	#self.dt-=1e-8
            if random.uniform(0,1)>0.5:
            	self.dt-=1e-8	
            if error>0.11:
                break 
        return (self.dt, self.decay_trigger_time, tc)   	
            
    def MSE(self, pred, obs):
        diff=sum(pred-obs)
        return math.sqrt(diff*diff)	  
        
    def max_error(Self, pred, obs):
    	diff=abs(pred-obs)
    	return max(diff)              


def main():
    trials=100
    bar = progressbar.ProgressBar(maxval=trials, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    with open('final_times_random_decay.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['time step','decay trigger time', 'final time'])
        for trial in range(trials):
            sim = LaxWendroff(1000, random.uniform(0.0088, 0.0092), random.uniform(0,10))
            csv_out.writerow(sim.solve_and_plot())
            bar.update(trial+1)
            #print("Finished trial number ", trial+1)
    bar.finish()        
    #plt.show()
    
    
if __name__ == "__main__":
    main()

#N = 100  
#tmax = 2.5 # maximum value of t



