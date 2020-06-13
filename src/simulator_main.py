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

#from parser_main import Parser 
import numpy as np 
#from qlearning import QLearning
from DQN_Class import DQN
from pomdp_simulator import Simulator
import random 

#from collections import deque
#from keras.models import Sequential
#from keras.layers import Dense, LSTM 
#from keras.optimizers import Adam

# rockSample-3_1
# Tiger 
# rockSample-7_8


def get_stateobservation_matrix(): 
    """
    
    takes a dictionary representation of the state & observation 
    
    return a matrix representation of the current observable part of the state and the observation 


    definitely apply this as a simulator function - no need to place it here 

    """ 
    
    
    
    pass 

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
    #print('obs_space',observation_space)
    
    length = 0 
    
    for i in observed_current_state: 
        # need to get the index out of the list 
        
        obj = observed_current_state[i]
        
        #print(obj)
        
        #print(simulator.state[i][0])
        
        index = simulator.state[i][0].index(obj)
        length += len(simulator.state[i][0]) 
        
        observation_space[index] = 1
        
        
    for i in observation: 
        # need to get the index out of the list 
        obj = observation[i] 
        
        #print(obj)
        
        #print(simulator.observation_names[i])
        
        index = simulator.observation_names[i].index(obj)
        
        #print('test',length+index)
        observation_space[length+index]=1
    
    #print(observation_space)
    #print(len(observation_space))
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
    
    #print('...',simulator.observation_names)
    
    #for key in simulator.state
    length = 0 
    #observable_state ={} 
    for key in state_key_list: 
        #print(key)
        #print(simulator.state[key])
        if simulator.state[key][1]=='true': 
            #print(key)
            length+= len(simulator.state[key][0])
            #length = len(initial_belief[key]) # need to change
    for key in observation_key_list: 
        length+= len(simulator.observation_names[key])
    
    observation_space = np.zeros((length,))
    print(len(observation_space))
    return observation_space 


def reset(simulator): 
    state = simulator.initial_state 
    observable_state = simulator.get_observable_state(state)
    return state, observable_state


def control_method(simulator, 
                   control="Random", 
                   training_period = 10, 
                   verbose=True, 
                   history = False,
                   maxsteps = 100):
    
    # get the initial state & observation information & establish the observation_space 
    
    

        
    #print('\\\\\\\ ',observation)
    
    observation_space = get_observation_space(simulator)
    
    #initial_belief = simulator.initial_belief 

    total_reward = 0 
    
    # define some objects for handling actions 
    action_keys = list(simulator.actions.keys())[0] 
    action_list = simulator.actions[action_keys]
    action_n = len(action_list)
    action_space = np.zeros(action_n) 
    
    dqn = DQN(action_list, observation_space)
    
    
    
    

    
    
        
    
    
    for i in range(training_period):
        print(i)
        state, observable_state = reset(simulator)
        observation = {}
        for i in simulator.observation_key_list: 
            observation[i] = random.choice(simulator.observation_names[i])
        
        
        
        if verbose: 
            print('a', i)
        
        for j in range(maxsteps): 
            
            if verbose:
                print('step',j+1)
    
            if control == "Random": 
                action_taken = random.choice(action_list)
            if control == "Human": 
                action_index = int(input('What action to take:\n'+str(action_list)))
                action_taken = simulator.actions[action_keys][action_index]
            if control == "DQN": 
                print('DQN')
                print('..........................')
    
    
                
                numpy_observation = numpy_conversion(simulator,observable_state,observation) 
                
                action_index= dqn.act(numpy_observation)
                print('action_index_dqn',action_index)
                action_taken = simulator.actions[action_keys][action_index]
                print('look here Cameron', action_taken)
                
            
                #print('Action taken',action_taken)
            next_state, step_observation, step_reward, observable_state = simulator.step(action_taken,state)
            
            # need to do some conversion to this representation?? 
            if verbose: 
                print('Action taken',action_taken)
                print('State ', next_state,'\n Observation ',step_observation,'\n', step_reward,'\n')
            
            if control == "DQN": 
                pass # train 
            state = next_state
            total_reward += step_reward 
            observable_state = simulator.get_observable_state(state) 
            observation = step_observation
    
        print(control,total_reward)
            
    
def main(file = '../examples/rockSample-7_8.pomdpx', 
         control = 'DQN', 
         training_period = 50,
         testing_period = 1): 
    simulator = Simulator(file)
    simulator.print_model_information()
    
    control_method(simulator,control,training_period)
    
    
if __name__ == '__main__': 
    main()
    
