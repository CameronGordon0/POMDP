#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file runs a simulator for the parsed pomdpx files. 

Currently manual input for the Tiger problem. 

To do: 
    - Change to automated input 
    - Enable running on rockSample & switching between problems 
    - Correct parser bugs for AUV and Tag pomdpx files 
    
    
    - New design choice is to work with dictionaries, for more readable data structures 
    - May wish to later convert data structures in the parser to Pandas for increased readability 

"""


from parser_main import Parser 
import numpy as np 
#from qlearning import QLearning
#from dqn import DQN

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.optimizers import Adam



class Simulator(): 
    def __init__(self, model_filename):
        """
        In the init I need to pull out the things I need for the rest of the problem 
        
        Main one that will be useful is 

        """
        
        
        parser = Parser(model_filename)
        self.state, self.pairs = parser.states
        print(self.state)
        self.state_name = self.state.keys()
        #print(self.state_name)
        #self.action_name = parser.actions['action_agent']
        self.transition = parser.state_transition 
        self.state_variables = parser.state_variable_dict 
        print(self.state_variables)
        #print(self.transition)
        #print(parser.actions)
        self.actions = parser.actions
        self.action_name = self.actions.keys()
        self.reward = parser.reward_table
        #print(self.reward)
        self.initial_belief = parser.initial_belief
        #print(self.initial_belief)
        
        self.initial_state = {} 
        for key in self.initial_belief.keys(): 
            distribution = self.initial_belief[key]
            choices = self.state[key][0] # note: may need to pull out observability here 
            choice = np.random.choice(choices,p=distribution)
            self.initial_state[key] = choice
            
        #print(self.initial_state)
        
        self.observation = parser.obs_table
        #self.observation_name = parser.observations['obs_sensor']
        #print(self.observation)
        self.observation_names = parser.observations
        
        self.total_reward = 0 
        #self.initial_state = self.state_name.index(np.random.choice(self.state_name,p=self.initial_belief))
        
        print('get_index_test',self.get_index('listen'))
                                                   
    def print_model_information(self): 
        print('MODEL INFORMATION \n.................................')
        print('State', self.state_name)
        print('Action', self.actions)
        print('Observation', self.observation_names)
        #print('Observation R',self.observation)
        print('Initial belief', self.initial_belief)
        print('Initial state', self.initial_state)
        
    def get_index(self, item): 
        num = None 
        for key in self.actions: 
            for i in self.actions[key]:
                try:
                    if i == item: 
                        num = self.actions[key].index(item)
                    #print(index)
                        break
                except: 
                    pass 
        for key in self.state: 
            for i in self.state[key]: 
                try:
                    if item in i: 
                        num = i.index(item)
                        break 
                except: 
                    pass
        return num
    
    def get_observation_tuple(self, action, new_state, key): 
        observation_table = self.observation[key] 
        action_index = self.get_index(action) 
        observation_tuple = [] 
        observation_tuple.append(action_index)
        for i in new_state: 
            state_index = self.get_index(new_state[i])
            observation_tuple.append(state_index)
            
        #print('obs_tuple',observation_tuple)
        observation_tuple = tuple(observation_tuple) 
        return observation_tuple
    
    def get_new_observation(self,action,new_state): 
        obs_dict = {} 
        for key in self.observation: 
            observation_table = self.observation[key] 
            observation_tuple = self.get_observation_tuple(action, new_state, key)
            distr = observation_table[observation_tuple]
            obs_dict[key] = np.random.choice(self.observation_names[key],p=distr)
        return obs_dict 
    
    def get_action_tuple(self): 
        pass 
    
    def get_state_tuple(self, action, state, key): 
        
        cond_vars = self.state_variables[key]
        update_list = [] 
        for i in cond_vars: 
            if i in self.action_name: 
                action_index = self.get_index(action)
                update_list.append(action_index)
            if i in self.state_name:
                state_index = self.get_index(state[i])
                update_list.append(state_index) 
                
        update_index = tuple(update_list) 
        
        return update_index
    
    def get_reward_tuple(self, action, state): 
        reward_index = []
        reward_index.append(self.get_index(action))
        for key in state.keys(): 
            state_var = state[key]
            state_var_index = self.get_index(state_var)
            reward_index.append(state_var_index)
        reward_index = tuple(reward_index) 
        return reward_index
    
    def get_new_state(self, action, state): 
        
        #print('get_new_state')
        #print('action',type(action),action)
        #print('state', type(state), state)
        
        
        
        new_state = {} 
        
        for key in state.keys():
            """
            Calculate the new state 
            """
            update_index = self.get_state_tuple(action, state, key)
            distr = self.transition[key][update_index]
            
            #print(distr)
            
            choice = np.random.choice(self.state[key][0],p=distr)
            new_state[key]=choice 
        
        return new_state 


    def step(self, action, state): 
        """ 
        Takes the action (str), state (dict) 
        
        Returns the next state (dict), observation (dict), and reward (float)
        
        
        """
        
        #print('STEP_FUNCTION')

        new_state = self.get_new_state(action, state)
        obs_dict = self.get_new_observation(action, new_state)
            
        reward_index = self.get_reward_tuple(action, state)
        step_reward = self.reward[reward_index]
        
        return new_state, obs_dict, step_reward


