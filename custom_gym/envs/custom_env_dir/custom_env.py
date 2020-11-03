import gym 
from gym import spaces
import numpy as np 
import random 

#from src.pomdp_simulator import Simulator

# Note: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html 

# Code for specifying custom gym environment with POMDPX information. 

from pathlib import Path 
import os, inspect, sys


def build_directory_path(): 
        
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
        

build_directory_path()

from pomdp_simulator import *
from simulator_main import numpy_conversion
import numpy as np


class CustomEnv(gym.Env): 
    metadata = {'render.modes': ['human']}


    def __init__(self): 
        #super(CustomEnv, self).__init()
        print('Environment initialised!') 
        #super(CustomEnv,self).__init__()
        
        history = False 
        self.step_number = 0 

        self.sim = Simulator('../examples/rockSample-7_8.pomdpx')
        self.state = self.sim.initial_state 
        self.observable_state = self.sim.get_observable_state(self.state)
        self.observation = {} 
        for i in self.sim.observation_key_list: 
            self.observation[i] = random.choice(self.sim.observation_names[i]) # initial random observation 
        
        
        self.numpy_space = numpy_conversion(self.sim, 
                                            self.observable_state,
                                            self.observation) # requires an observation. initial observation is random?
        
        
        self.observation_space = spaces.Box(low=0,high=1,shape=self.numpy_space.shape) # putting low = 0 and high = 1 as one-hot encoding 
        
        #print(self.gym_observation_space)

        
        # note: this borrows from how this is handled in simulator_main.control_method. Not good design principle. 
        self.action_keys = list(self.sim.actions.keys())[0]
        self.action_list = self.sim.actions[self.action_keys]
        action_n = len(self.action_list)
        self.action_space = spaces.Discrete(action_n) # note: control_method uses numpy_zeros instead 
        
        #print("-----",self.action_space)
        
        
        # Define action and observation space 
        # They must be gym.spaces objects 
        # e.g. 
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS) 
        # self.observation_space = spaces.Box(low=0,high=255, shape=(HEIGHT, WIDTH, N_Channels), dtype=np.uint8)
        
        # two choices here: history and non-history. 
        
        if history == False: 
            pass 
        
        
        
        #print(self.sim)
        #print(self.sim.state)
        #print(self.sim.print_model_information(verbose=True))
        
        #print(self.observable_state)
        
    def step(self, action="random"): # temporarily taking a random for testing purposes 
        if action == "random": 
            print('testing only - random action')
            action = random.choice(self.action_list) 
            print(action)
        else: 
            action = self.action_list[action]
            print('chose action',action)
        new_state, step_observation, step_reward, observable_state = self.sim.step(action,self.state)
        observable_state = self.sim.get_observable_state(new_state) # ugly design practice, but in place because handled this way in other simulator 
        print('Step successful!') 
        #print(action)
        #print(self.action_keys)
        #print(self.action_list[0])
        print(new_state)
        #print(obs_dict) 
        print(step_reward) 
        print(observable_state) 
        
        # needs to return 'observation, reward, done, info' for the openAI API 
        # observation needs to be an openAI space. i.e. BOX 
        
        gym_observation = numpy_conversion(self.sim, 
                                           observable_state, step_observation) # needs to convert (new_state & observable_state) from numpy to gym space 
        gym_reward = step_reward 
        if self.step_number > 50:
            gym_done = True 
        else: 
            gym_done = False 
            self.step_number=self.step_number + 1 
            
        gym_info = {} # dictionary used for debugging purposes only 
        
        return gym_observation, gym_reward, gym_done, gym_info 

        
    def reset(self): 
        print('Environment reset') 
        self.state = self.sim.initial_state 
        observable_state = self.sim.get_observable_state(self.state) 
        self.observation = {}
        for i in self.sim.observation_key_list: 
            self.observation[i] = random.choice(self.sim.observation_names[i])
            
        gym_observation = numpy_conversion(self.sim, observable_state, self.observation) 
        
        return gym_observation
    
    def render(self, mode='human'): 
        if mode == 'human':
            pass 
        
    def close(self): 
        print('close') 





