import gym 

#from src.pomdp_simulator import Simulator

# Note: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html 

from pathlib import Path 
import os, inspect, sys

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))

#print(cmd_folder)
path = Path(cmd_folder)
path_src = str(path.parent)+'\src'
#print(path.parent)


if path.parent not in sys.path: 
    sys.path.insert(0,path.parent)
    

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
    
    
if path_src not in sys.path: 
    sys.path.insert(0,path_src)
    
#from simulator_main import reset as sim_reset
    
    
for i in sys.path: 
    print(i)

from pomdp_simulator import *




class CustomEnv(gym.Env): 
    metadata = {'render.modes': ['human']}


    def __init__(self): 
        print('Environment initialised!') 
        #super(CustomEnv,self).__init__()

        self.sim = Simulator('../examples/Tiger.pomdpx')
        self.state = self.sim.initial_state 
        self.observable_state = self.sim.get_observable_state(self.state)
        
        print(self.sim)
        print(self.sim.state)
        print(self.sim.print_model_information(verbose=True))
    def step(self, action): 
        print('Step successful!') 
        new_state, obs_dict, step_reward, observable_state = self.sim.step(action,self.state)
        
        
    def reset(self): 
        print('Environment reset') 
        state = self.sim.initial_state 
        observable_state = self.sim.get_observable_state(state) 
        
    def close(self): 
        print('close') 

