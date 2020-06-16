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
import matplotlib.pyplot as plt 


# rockSample-3_1
# Tiger 
# rockSample-7_8



def history_queue(new_observation=None, old_history=None):
    # take in a single observation & a current history 
    # return the new history containing the new observation 
    # acts as a queue for the numpy arrays (FIFO) 
    
    # note that the new_observation needs to have the same dimensions as the old history
    
    # see example: 
        # old_history = np.zeros((5,1,3))
        # old_history[-1] = np.ones((1,3))
        # new_observation = 2*np.ones((1,1,3))
        # new_hist = history_queue(new_observation,old_history)
    
    
    new_history = old_history[1:] # copies the old history less the oldest observation 
        
    new_history = np.append(new_history,new_observation,axis=0) 
    
    return new_history  



def numpy_conversion(simulator, 
                     observed_current_state,
                     observation,
                     history = False, # can probably leave most of the logic for the history out of this one ?
                     # trick may be to reshape the vector in the main loop rather than messing around with something here 
                     history_len=1): 
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
        # suggested change observation_space[history_len-1][index]
        
        
    for i in observation: 
        # need to get the index out of the list 
        obj = observation[i] 
        
        index = simulator.observation_names[i].index(obj)
        
        #print('test',length+index)
        observation_space[length+index]=1
        # suggested change observation_space[history_len-1][length + index]
    
    if history == True: 
        observation_space = np.reshape(observation_space,(1,int(observation_space.shape[0])))
    
    return observation_space 

def get_observation_space(simulator, 
                          history = False, 
                          history_len = 1): 
    """
    Best approach here may just be a flat structure 
    
    one hot encoding of fully observed state(s) + observation(s) 
    
    just run through a loop of keys for both 
    adding through the length of the variables 
    
    create a 1x vector one hot encoding 
    
    
    can probably extend this to a hx history, where the history becomes a queue of the recent frames 
    (with zeros for the first few frames) 
    
    will require changes to how the inputs are handled for the other components in main / DQN_Class 

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
    
    if history: 
        observation_space = np.zeros((history_len,length,))
    else: 
        observation_space = np.zeros((length,)) 
        
    # option here is to use a history check to allow running for historyless & specified history 
    # may allow more easier checks in development rather than everything needing to be implemented at once (and breaking things) 
    
    #observation_space = np.zeros((history_len,length,)) suggested change 
    #print(len(observation_space))
    return observation_space 


def reset(simulator,
          history=False,
          history_len=1): 
    state = simulator.initial_state 
    observable_state = simulator.get_observable_state(state)
    return state, observable_state


def control_method(simulator, 
                   control="Random", 
                   training_period = 100, 
                   verbose=False, 
                   history = False,
                   history_len = 1, 
                   maxsteps = 25):
    
    observation_space = get_observation_space(simulator) #needs to contain history as well 

    # define some objects for handling actions 
    action_keys = list(simulator.actions.keys())[0] 
    action_list = simulator.actions[action_keys]
    action_n = len(action_list)
    action_space = np.zeros(action_n) 
    
    
    
    dqn = DQN(action_list, observation_space) 
    # need to check how this is handled in the DQN (history) 
    
    
    results_y = [] 
    results_x = np.arange(0,training_period)
    
    for it in range(training_period):
        #print(i)
        total_reward = 0 
        state, observable_state = reset(simulator) 
        observation = {}
        for i in simulator.observation_key_list: 
            observation[i] = random.choice(simulator.observation_names[i])
            
        iteration_history = get_observation_space(simulator) # define within the observable_space function, as this is passed to the DQN file  
        
        
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
                
                # history stack needs to be specified here 
                numpy_observation = numpy_conversion(simulator,observable_state,observation) 
                # need to make sure the numpy_observation is the same dim (along the axis) as the history matrix 
                # want to do this within the numpy_observation 
                
                #history_queue = history_queue(new_observation=numpy_observation,old_history=)
                
                action_index= dqn.act(numpy_observation)
                # need to make modifications in the dqn file to handle the extra index 
                action_taken = simulator.actions[action_keys][action_index]
                
            next_state, step_observation, step_reward, observable_state = simulator.step(action_taken,state)
            
            # need to do some conversion to this representation?? 
            if verbose: 
                print('Action taken',action_taken)
                print('State ', next_state,'\n Observation ',step_observation,'\n', step_reward,'\n')
            
            if control == "DQN": 
                # train 
                
                # history stack needs to be specified here. Add new observation, remove the old one 
                
                cur_state = numpy_conversion(simulator,observable_state,observation)
                obs_new_state = simulator.get_observable_state(next_state)
                new_state = numpy_conversion(simulator,obs_new_state,step_observation)
                # because this refers to new state not next state (data_type difference) 
                
                if history == True: 
                    pass # need to handle the history 
                
                done = False
                if j >= maxsteps-1:
                    done = True
                    
                dqn.remember(cur_state, action_index, step_reward, new_state, done) 
                # need to see how this is handled by the dqn file 
                
                # may need to do some thinking on how the stack is remembered 
            
                dqn.replay()
                dqn.target_train()
            state = next_state
            total_reward += step_reward 
            observable_state = simulator.get_observable_state(state) 
            # ideally want to leave this as same as current implementation 
            # think this is fine, as the call really only depends on next_state (which is a dictionary) 
            # although need to keep a running history at some point in the algorithm 
            # may want to set this up initially and then keep it separate from the immediate state (which is passed to the simulator) 
            observation = step_observation 
            # need to handle history stack here 
    
        print('iteration',it,control,total_reward)
        results_y.append(total_reward)
    results_y = np.asarray(results_y)
    
    details = {}
    details['model'] = control 
    # details could be a useful dictionary for storing the key parameters for plotting
    # (e.g. the model type, the key parameters)
    # should also include the problem in the details (pull from parser)
    
    
    return results_x, results_y, control

def plot_results(x,y,details): 
    fig = plt.figure()
    plt.plot(x,y)
    plt.title(details)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.show()

    #plot(x,y)
    
            
def main(file = '../examples/Tiger.pomdpx', 
         control = 'DQN', 
         training_period = 30,
         testing_period = 1,
         verbose = False,
         history = False,
         history_len = 0): 
    simulator = Simulator(file)
    simulator.print_model_information()
    
    x,y,details = control_method(simulator,control,training_period)
    
    plot_results(x,y,details)
    
    
def unit_test_1(): 
    # testing history componets in control_method 
    file = '../examples/rockSample-3_1.pomdpx'
    #file = '../examples/rockSample-7_8.pomdpx'
    #file = '../examples/Tiger.pomdpx'
    simulator = Simulator(file)
    obs_space = get_observation_space(simulator,history=True,history_len=10)
    print('obs_test',obs_space)
    action_keys = list(simulator.actions.keys())[0] 
    action_list = simulator.actions[action_keys]
    
    dqn = DQN(action_list, obs_space,history=True) 
    print('DQN',dqn)
    
    state, observable_state = reset(simulator,history=True,history_len=10)
    print('state',state)
    print('observable state', observable_state)
    
    observation = {}
    for i in simulator.observation_key_list: 
        observation[i] = random.choice(simulator.observation_names[i])
        
    iteration_history = get_observation_space(simulator,history=True,history_len=10)
    
    print('observation',observation)
    print('iteration_history', iteration_history) 
    
    numpy_observation = numpy_conversion(simulator,observable_state,observation, history=True)
    # print('numpy_observation',numpy_observation)
    # print(numpy_observation.shape)
    # x, y = int(numpy_observation.shape[0]),int(numpy_observation[1])
    # numpy_observation = np.reshape(numpy_observation,(1,x))
    print(numpy_observation.shape)
    print(iteration_history.shape)
    
    
    iteration_history = history_queue(numpy_observation, iteration_history)
    
    print('iteration_history',iteration_history) 
    
    action_index = dqn.act(iteration_history)
    
    print(action_index)
    
    iteration_history = np.reshape(iteration_history,
                                   (-1,int(iteration_history.shape[0]),int(iteration_history.shape[1])))
    
    print(iteration_history.shape)
    print(dqn.model.predict(iteration_history)[0])
    
    
    test = np.argmax(dqn.model.predict(iteration_history)[0])
    
    print('test',test)
    
    #dqn.
    
    
    
    
    
    
if __name__ == '__main__': 
    #main()
    unit_test_1()
    
