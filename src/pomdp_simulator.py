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
    
    - Cleaning up variables and dependencies 

"""


from parser_main import Parser 
import numpy as np 

class Simulator(): 
    def __init__(self, model_filename):
        """
        Variable types in init
        self.state <class 'dict'> state vars, e.g. {'robot_0': (['s00', 's01'],'true'),'rock0_0': (['bad', 'good'], 'false')}
        self.pairs <class 'dict'> pairs of keys (e.g. robot_1 = robot_0)
        self.state_key_list <class 'dict_list'> e.g. ['robot_0'] # to-vars removed (e.g. 'robot_1')
        self.transition <class 'dict'> Numpy array for function T(s'|s,a) 
        self.state_variables <class 'dict'> conditional transition vars e.g. {'robot_0': ['action_robot', 'robot_0']} 
        self.actions <class 'dict'> self.actions {'action_robot': ['amn', 'ame', 'ams', 'amw'} 
        self.action_key_list <class 'list'> ['action_robot'] # note: needs to be corrected in main 
        self.reward <class 'numpy.ndarray'> Numpy array for function R(a,s))
        self.initial_belief <class 'dict'> e.g. {'rock0_0': array([0.5, 0.5]), 'rock1_0': array([0.5, 0.5])}
        self.initial_state <class 'dict'> e.g. {'robot_0': 's03', 'rock0_0': 'bad'}
        self.observation <class 'dict'> # note this is the O(o|s',a) function. Key is the observation name 
        self.observation_names <class 'dict'> # observation names. e.g. {'obs_sensor': ['ogood', 'obad']}
        self.observation_key_list <class 'list'> list of observation keys 
        """
        
        parser = Parser(model_filename)
        self.state, self.pairs = parser.states
        self.state_key_list = list(self.state.keys())
        
        templist = [] 
        for i in self.state_key_list: 
            i = self.pairs[i] 
            if i not in templist: 
                templist.append(i)
        self.state_key_list = templist
        
        tempdict = {} 
        
        for i in self.state_key_list: 
            tempdict[i] = self.state[i] 
            
        self.state = tempdict 
        
        self.transition = parser.state_transition 
        self.state_variables = parser.state_variable_dict 
        self.actions = parser.actions
        self.action_key_list = list(self.actions.keys())
        
        
        self.reward = parser.reward_table
        self.initial_belief = parser.initial_belief
        
        self.initial_state = {} 
        for key in self.initial_belief.keys(): 
            distribution = self.initial_belief[key]
            choices = self.state[key][0] # note: may need to pull out observability here 
            choice = np.random.choice(choices,p=distribution)
            self.initial_state[key] = choice
        
        self.observation = parser.obs_table
        self.observation_names = parser.observations
        self.observation_key_list = list(parser.observations.keys())
        
                                                  
    def print_model_information(self, verbose = True, show_types=False, show_vars=False): 
        if verbose: 
            print('MODEL INFORMATION \n.................................')
            print('State', self.state_key_list)
            print('Action', self.actions)
            print('Observation', self.observation_names)
            #print('Observation R',self.observation)
            print('Initial belief', self.initial_belief)
            print('Initial state', self.initial_state)
        
        if show_types: 
            print("\nVariable types in init")
            print("self.state", type(self.state))
            print("self.pairs", type(self.pairs)) 
            print("self.state_key_list", type(self.state_key_list)) 
            print("self.transition", type(self.transition))
            print("self.state_variables", type(self.state_variables)) 
            print("self.actions", type(self.actions))
            print("self.action_key_list", type(self.action_key_list))
            print("self.reward", type(self.reward)) 
            print("self.initial_belief", type(self.initial_belief))
            print("self.initial_state",type(self.initial_state))
            print("self.observation", type(self.observation)) 
            print("self.observation_names", type(self.observation_names)) 
            
        if show_vars: 
            print("\nVariables in init")
            print("self.state", self.state)
            print("self.pairs", self.pairs) 
            print("self.state_key_list", self.state_key_list)

        
        
    def get_index(self, item): 
        """
        Parameters: 
            item (str) - Name of action or state to find the index for. 
            
        Description: 
        Searches through state and action dictionaries and lists to find the index of 'item'. 
        
        Returns: num (int), the index of the item in the relevant [dictionary and list].
        """
        num = None 
        for key in self.actions: 
            for i in self.actions[key]:
                try:
                    if i == item: 
                        num = self.actions[key].index(item)
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
    
    def get_observation_tuple(self, action, new_state): 
        """
        Parameters: 
            action (str) - the action taken 
            new_state (dict) - the new state that resulted from the action
            
        Description: 
            Gets the tuple of O(s',a); the observation from an action that 
                               led to the new state
            
        Returns: a tuple of the observation
        """
        action_index = self.get_index(action) 
        observation_tuple = [] 
        observation_tuple.append(action_index)
        for i in new_state: 
            state_index = self.get_index(new_state[i])
            observation_tuple.append(state_index)
            
        observation_tuple = tuple(observation_tuple) 
        return observation_tuple
    
    def get_new_observation(self,action,new_state): 
        """
        Parameters: 
            action (str) - the action taken 
            new_state (dict) - the new state that resulted from the action 
            
        Description: 
            Get the observation O(s',a) 
            
        Returns: a dictionary of the new observation 
        """
        obs_dict = {} 
        for key in self.observation: 
            observation_table = self.observation[key] 
            observation_tuple = self.get_observation_tuple(action, new_state)
            distr = observation_table[observation_tuple]
            obs_dict[key] = np.random.choice(self.observation_names[key],p=distr)
        return obs_dict 
    
    
    def get_state_tuple(self, action, state, key): 
        """
        Parameters: 
            action (str) - the action taken 
            state (dict) - the state to be updated 
            key (str) - the part of the state to be updated (e.g. rock-0, robot-0)
            
        Gets the tuple to determine the new state in numpy. 
        
        Returns: a tuple to update the state 
        
        """
        
        cond_vars = self.state_variables[key]
        update_list = [] 
        for i in cond_vars: 
            if i in self.action_key_list: 
                action_index = self.get_index(action)
                update_list.append(action_index)
            if i in self.state_key_list:
                state_index = self.get_index(state[i])
                update_list.append(state_index) 
                
        update_index = tuple(update_list) 
        
        return update_index
    
    def get_reward_tuple(self, action, state): 
        """
        Parameters
        ----------
        action (str) : the action taken 
            
        state (dict) : the resulting state

        Returns: a tuple for extracting the reward R(s',a) from numpy.  

        """
        
        reward_index = []
        reward_index.append(self.get_index(action))
        for key in state.keys(): 
            state_var = state[key]
            state_var_index = self.get_index(state_var)
            reward_index.append(state_var_index)
        reward_index = tuple(reward_index) 
        return reward_index
    
    def get_new_state(self, action, state): 
        """
        Parameters: 
            action (str) - the action taken 
            state (dict) - the current state 
            
        Description: 
            Gets the new state. 
            
        Returns: 
            A dictionary of the new state. 
        """
        
        new_state = {} 
        
        for key in state.keys():
            """
            Calculate the new state 
            """
            update_index = self.get_state_tuple(action, state, key)
            distr = self.transition[key][update_index]
                        
            choice = np.random.choice(self.state[key][0],p=distr)
            new_state[key]=choice 
        
        return new_state 


    def step(self, action, state): 
        """ 
        Takes the action (str), state (dict) 
        
        Returns the next state (dict), observation (dict), reward (float), 
        and observable part of the state (dict) 
        
        """
        
        new_state = self.get_new_state(action, state)
        
        # also need to pull out the observable parts of the state 
        observable_state = self.get_observable_state(state) # this should be new state, but further down inthe pipeline it's been manually converted ... 

        obs_dict = self.get_new_observation(action, new_state)
        reward_index = self.get_reward_tuple(action, state)
        step_reward = self.reward[reward_index]
                
        return new_state, obs_dict, step_reward, observable_state



    def get_observable_state(self, state): 
        """
        Parameters: 
            state (dict) - the current state 
            
        Returns: 
            A dictionary of the observable part of the state. 
        """
        observable_state = {} 
        for key in state: 
            if self.state[key][1] == 'true': 
                observable_state[key] = state[key] 
        return observable_state 
    
    def action_to_vector(self): 
        pass 
    
    def vector_to_action(self): 
        pass 
    
    
    #def reset_simulator(self):
     #   pass  
     
     
    

