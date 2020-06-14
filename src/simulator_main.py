#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:28:45 2020

@author: camerongordon

Difficulties for the simulator modules (issue with the different representations;
                                        currently have three for most of the variables (dict, str, int)) 

Should note that the dict representation was chosen deliberately to handle the hander problems. 

Should try to keep consistent entries.

If necessary, should perform the conversion outside the entry module (to prevent breaking other components) 

Need to do thinking around the qlearning module - the construction of the Q-Matrix
Should it be belief x action?? 

May be a tricky issue to decide how to handle this - eventually needs to pass to numpy matrices (tabular) & as vectors for the keras implementations. 

Creating a standard way of handling may be useful. 

Experimentation - may want an option to run QLearning etc on the fully observable state. Would be a good cross-check (esp for rockSample) 

May need to consider how the 'observable' parts of the state are handled as well.

E.g. in rockSample the agent knows where it is & where the rocks are?  

"""

import numpy as np 
from DQN_Class import DQN
from pomdp_simulator import Simulator
import random 


# rockSample-3_1
# Tiger 
# rockSample-7_8




def numpy_conversion(simulator, 
                     observed_current_state,
                     observation): 
    """
    Takes two dictionaries: 
        observed_current_state (fully-observed part of state) 
        observation 
        
    Returns a numpy array responding to the indexing 

    """
    
    observation_space = get_observation_space(simulator)
    
    length = 0 
    
    for i in observed_current_state: 
        
        obj = observed_current_state[i]
        
        index = simulator.state[i][0].index(obj)
        length += len(simulator.state[i][0]) 
        
        observation_space[index] = 1
        
        
    for i in observation: 
        # need to get the index out of the list 
        obj = observation[i] 
        
        index = simulator.observation_names[i].index(obj)
        
        #print('test',length+index)
        observation_space[length+index]=1
    
    return observation_space 

def get_observation_space(simulator): 
    """
    Best approach here may just be a flat structure 
    
    one hot encoding of fully observed state(s) + observation(s) 
    
    just run through a loop of keys for both 
    adding through the length of the variables 
    
    create a 1x vector one hot encoding 

    """
    
    state_key_list = simulator.state_key_list 
    observation_key_list = simulator.observation_key_list 
    
    #for key in simulator.state
    length = 0 
    for key in state_key_list: 
        if simulator.state[key][1]=='true': 
            #print(key)
            length+= len(simulator.state[key][0])
            #length = len(initial_belief[key]) # need to change
    for key in observation_key_list: 
        length+= len(simulator.observation_names[key])
    
    observation_space = np.zeros((length,))
    #print(len(observation_space))
    return observation_space 


def reset(simulator): 
    state = simulator.initial_state 
    observable_state = simulator.get_observable_state(state)
    return state, observable_state


def control_method(simulator, 
                   control="Random", 
                   training_period = 100, 
                   verbose=False, 
                   history = False,
                   maxsteps = 100):

    observation_space = get_observation_space(simulator)

    # define some objects for handling actions 
    action_keys = list(simulator.actions.keys())[0] 
    action_list = simulator.actions[action_keys]
    action_n = len(action_list)
    action_space = np.zeros(action_n) 
    
    dqn = DQN(action_list, observation_space)
    
    
    for it in range(training_period):
        #print(i)
        total_reward = 0 
        state, observable_state = reset(simulator)
        observation = {}
        for i in simulator.observation_key_list: 
            observation[i] = random.choice(simulator.observation_names[i])
        
        
        if verbose: 
            print('iteration', it)
        
        for j in range(maxsteps): 
            
            if verbose:
                print('step',j+1)
    
            if control == "Random": 
                action_taken = random.choice(action_list)
            if control == "Human": 
                action_index = int(input('What action to take:\n'+str(action_list)))
                action_taken = simulator.actions[action_keys][action_index]
            if control == "DQN": 
    
                numpy_observation = numpy_conversion(simulator,observable_state,observation) 
                
                action_index= dqn.act(numpy_observation)
                action_taken = simulator.actions[action_keys][action_index]
                
            next_state, step_observation, step_reward, observable_state = simulator.step(action_taken,state)
            
            # need to do some conversion to this representation?? 
            if verbose: 
                print('Action taken',action_taken)
                print('State ', next_state,'\n Observation ',step_observation,'\n', step_reward,'\n')
            
            if control == "DQN": 
                # train 
                
                cur_state = numpy_conversion(simulator,observable_state,observation)
                obs_new_state = simulator.get_observable_state(next_state)
                new_state = numpy_conversion(simulator,obs_new_state,step_observation)
                done = False
                if j >= maxsteps-1:
                    done = True
                dqn.remember(cur_state, action_index, step_reward, new_state, done)
            
                dqn.replay()
                dqn.target_train()
            state = next_state
            total_reward += step_reward 
            observable_state = simulator.get_observable_state(state) 
            observation = step_observation
    
        print('iteration',it,control,total_reward)
            
def main(file = '../examples/Tag.pomdpx', 
         control = 'DQN', 
         training_period = 150,
         testing_period = 1): 
    simulator = Simulator(file)
    simulator.print_model_information()
    
    control_method(simulator,control,training_period)
    
    
if __name__ == '__main__': 
    main()
    
