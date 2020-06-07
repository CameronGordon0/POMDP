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


    def step(self, action, state): 
        """ 
        Appears to work currently 
        
        
        """
        
        #print('STEP_FUNCTION')
        
        #print(self.transition)
        
        action_index = self.get_index(action)
        #print(action, action_index)
        #print(state_index)
        #print(state)
        #print(self.transition)
        new_state = {} 
        
        #current_state_index = []
        #next_state_tuple = []
        
        #print(self.action_name)
        
        obs_dict = {}
        #reward_index = None
        
        for key in state.keys():
            """
            Calculate the new state 
            """
            
            #print('key',key) #i.e. what's updating 
            var_to_update = state[key]
            #print('var_to_update',var_to_update)
            cond_vars = self.state_variables[key]
            #print('cond_vars',cond_vars)
            
            update_list = [] 
            for i in cond_vars: 
                if i in self.action_name: 
                    #print(i,'////')
                    action_index = self.get_index(action)
                    update_list.append(action_index)
                if i in self.state_name:
                    #print('cccc',i)
                    state_index = self.get_index(state[i])
                    #print(state_index,state[i])
                    update_list.append(state_index) 
                    
            update_index = tuple(update_list)
            #print('update_index',update_index)
            #reward_index = update_index
            
            distr = self.transition[key][update_index]
            choice = np.random.choice(self.state[key][0],p=distr)
            #choice = np.random.choice(self.state[key])
            new_state[key]=choice 
        #print('old state', state)
        #print('new_state',new_state)
        
        #print(self.reward)
        #print(self.observation)
        for key in self.observation: 
            observation_table = self.observation[key] 
            action_index = self.get_index(action) 
            observation_tuple = [] 
            observation_tuple.append(action_index)
            for i in new_state: 
                state_index = self.get_index(new_state[i])
                observation_tuple.append(state_index)
                
            #print('obs_tuple',observation_tuple)
            observation_tuple = tuple(observation_tuple)
                
            #print(self.observation_names)
            distr = observation_table[observation_tuple]
            #print('----',distr)
            #print(self.observation_names)
            obs_dict[key] = np.random.choice(self.observation_names[key],p=distr)
            #print(obs_dict)
            
            
            
        reward_index = []
        reward_index.append(self.get_index(action))
        for key in state.keys(): 
            state_var = state[key]
            #print(state_var)
            state_var_index = self.get_index(state_var)
            reward_index.append(state_var_index)
        reward_index = tuple(reward_index)
        #print(self.reward.shape)
        #print('+++++',reward_index)
        step_reward = self.reward[reward_index]
        #print('[',step_reward)
        #step_reward = 0
        
        
        
        #print(state, action)
        """
        
        #for key in state.keys(): 
            
            This will lead to issues - but 
            actually want to pass a tuple in for the second transition part 
            issue is to do with conditional variables 
            should probably try and get a list of the conditional vars (e.g. rover_0, rock_0)
            to get a changeable size for the tuple that will need to be passed to the transition 
            
            may want to build new functions into the parser to specifically enable this 
            another option is to convert everything to pandas? 
            
            
            May need to get rid of the pairs issue????? 
            
            
            
            
            print('key',key)
            print(self.transition[key].shape)
            #print('conditional_vars',self.state_variables[key])
            
            cond_vars = self.state_variables[key]
            print('cond vars',cond_vars)
            
            for i in cond_vars:
                state_index = self.get_index(i)
                print('si',state_index)
            
            
            
            current_state_index.append(state_index)
            print(current_state_index)
            print(state_index)
            

            distribution = self.transition[key][action_index,state_index]
            print(distribution)
            choice = np.random.choice(self.state[key][0],p=distribution)
            new_state[key] = choice
            choice_index = self.get_index(choice)
            next_state_tuple.append(choice_index)
        
        current_state_index = tuple(current_state_index)
        next_state_tuple = tuple(next_state_tuple)
        

        
        #print(':::',self.observation)
        #print(self.observation_names)
        
        
        
        """
        """
        for key in self.observation.keys(): 
            
        
            observation_dist = self.observation[key][action_index,next_state_tuple][0]
            #print(observation_dist)
        
            step_observation = np.random.choice(self.observation_names[key],p=observation_dist)
            obs_dict[key] = step_observation 
            #print(step_observation)
            
        #print(step_observation)
    
        #step_observation = self.observation_name.index(step_observation)
        
        print(new_state,obs_dict,step_reward)
        
        """
        
        return new_state, obs_dict, step_reward
    
    
    

            
            
        
        
        
        
    
    
