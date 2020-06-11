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

from parser_main import Parser 
import numpy as np 
from qlearning import QLearning
#from dqn import DQN
from pomdp_simulator import Simulator
import random 

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.optimizers import Adam

# rockSample-3_1
# Tiger 
# rockSample-7_8


def control_method(simulator, control="Random", training_period = 10):
    
    state = simulator.initial_state 
    total_reward = 0 
    
    for i in range(training_period):
        print('step',i+1)
        action_name = list(simulator.actions.keys())[0] 
        if control == "Random": 
            action_taken = random.choice(simulator.actions[action_name])
        if control == "Human": 
            action_index = int(input('What action to take:\n'+str(simulator.actions[action_name])))
            action_taken = simulator.actions[action_name][action_index]
        if control == "DQN": 
            pass 
        next_state, step_observation, step_reward, observable_state = simulator.step(action_taken,state)
        print('State ', state,'\n Observation ',step_observation,'\n', step_reward,'\n')
        
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
    state = simulator.initial_state
    state_name = simulator.state_name
    total_reward = 0
    
    obs = state  # initial state = observation 
    
    control_method(simulator,control,training_period)
    
    
if __name__ == '__main__': 
    main()
    
