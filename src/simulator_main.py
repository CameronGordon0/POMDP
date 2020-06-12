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


def control_method(simulator, control="Random", training_period = 10,verbose=True):
    
    state = simulator.initial_state 
    total_reward = 0 
    action_name = list(simulator.actions.keys())[0] 
    action_list = simulator.actions[action_name]
    action_n = len(action_list)
    action_space = np.zeros(action_n)
    
    initial_belief = simulator.initial_belief 
    
    #for key in simulator.state
    length = 0 
    observable_state ={} 
    for key in initial_belief: 
        #print(key)
        #print(simulator.state[key])
        if simulator.state[key][1]=='true': 
            #print(key)
            observable_state[key] = len(initial_belief[key])
            length = len(initial_belief[key]) # need to change
    
    observation_space = np.zeros((len(observable_state), length+len(simulator.observation)))
            
        
    
    
    for i in range(training_period):
        if verbose:
            print('step',i+1)

        if control == "Random": 
            action_taken = random.choice(action_list)
        if control == "Human": 
            action_index = int(input('What action to take:\n'+str(action_list)))
            action_taken = simulator.actions[action_name][action_index]
        if control == "DQN": 
            print('DQN')
            print('..........................')
            print('action_name', action_name) 
            print(action_list)
            print(action_n)
            print(action_space)
            print('state',state)
            
            print(initial_belief)
            for i in initial_belief: 
                print(i,len(initial_belief[i]))
            #print(len(initial_belief))
            
            print(simulator.observation_names)
            #print(observable_state)

                    
            print(observable_state)
            print(len(simulator.observation))
            print(len(observable_state))
            print([observable_state.values()][0])
            
            """
            idea one: pass the observation is len(fully obs) + len(observation)
            """
            
            print(observation_space,len(observation_space[0]))
            
            
            # need to work through the logic of the DQN within here. 
            # ideally, want to create this object outside whatever loop is occurring in the training period 
            # i.e. want it to continue training 
            
            # really need to think about the information that's passing through to it 
            # maybe should whiteboard it?? 
            
            # automatic convewrsion from dictionary to np input would be a good idea 
            
            # possible way to run a history would be to set up the numpy array with each row as a history?? 
            # or to keep a short-term buffer of history? 
            
            # how would this feed through to the function? 
            
            
            dqn = DQN(action_space, observation_space)
            
        
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
    
    print(control,total_reward)
            
    
def main(file = '../examples/rockSample-7_8.pomdpx', 
         control = 'Random', 
         training_period = 10,
         testing_period = 1): 
    simulator = Simulator(file)
    simulator.print_model_information()
    #initial_state = simulator.initial_state
    #state_key_list = simulator.state_key_list
    #total_reward = 0
    
    #obs = initial_state  # initial state = observation 
    
    control_method(simulator,control,training_period)
    
    
if __name__ == '__main__': 
    main()
    
