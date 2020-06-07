#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:28:45 2020

@author: camerongordon
"""

from parser_main import Parser 
import numpy as np 
from qlearning import QLearning
from dqn import DQN
from pomdp_simulator import Simulator
import random 

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.optimizers import Adam

# rockSample-3_1
# Tiger 
# rockSample-7_8
    
def main(file = '../examples/Tiger.pomdpx', 
         control = 'Random', 
         training_period = 10,
         testing_period = 1): 
    simulator = Simulator(file)
    simulator.print_model_information()
    state = simulator.initial_state
    state_name = simulator.state_name
    total_reward = 0
    
    obs = state  # initial state = observation 
    
    
    #control = 'Human' 
    if control == 'Human':
        for i in range(training_period):
            print('step ', i+1)
            action_name = list(simulator.actions.keys())[0]
            action_index = int(input('What action to take:\n'+str(simulator.actions[action_name])))
            action_name = simulator.actions[action_name][action_index]
            print('Took ', action_name)
        
            next_state, step_observation, step_reward = simulator.step(action_name,state)
            print(step_observation,'\n', step_reward,'\n')
            state = next_state
            total_reward += step_reward 
        print('Human Agent ', total_reward)
    
    if control == 'Random': 
        for i in range(training_period): 
            print('step ', i+1) 
            action_name = list(simulator.actions.keys())[0] 
            action_taken = random.choice(simulator.actions[action_name]) 
            print('Took', action_taken) 
            
            next_state, step_observation, step_reward = simulator.step(action_taken,state)
            print('State ', state,'\n Observation ',step_observation,'\n', step_reward,'\n')
            state = next_state
            total_reward += step_reward 
        print('Random Agent :', total_reward) 
        
    if control == 'QLearning': 
        q_agent = QLearning(file)
        for i in range(training_period): 
            done = False
            episode_reward = 0 
            max_steps = 20 
            n_step = 0 
            while not done: 
                n_step +=1 
                if n_step > max_steps: 
                    done = True 
                action = q_agent.get_action(obs) 
                print('Main', 'get_action',action)
                if type(action)==int: 
                    action = list(simulator.actions.values())[0][action]
                    # ugly way of handling it 
                print('Took ', action)
                next_state, step_observation, step_reward = simulator.step(action,state)
                print('State ', state,'\n Observation ',step_observation,'\n', step_reward,'\n')
                state = next_state
                episode_reward += step_reward 
                
                
                # may be easiest to handle the conversion to int values here 
                # other option may be to look at the data structure in the QLearning module 
                
                
                experience = obs, action, step_observation, step_reward, done 
                
                # issue with obs having been set as state 
                
                # may be a good idea to plan out the QLEARNING module. Run on beliefs? 
                
                print('EXPERIENCE')
                for x in experience: 
                    print(x,type(x))
                q_agent.train(experience)
                obs = step_observation
                
            print(episode_reward)
            
            
    
    if control == 'DQN': 
        pass 
    
    if control == 'DRQN': 
        pass 
    
    if control == 'ADRQN': 
        pass 
            
            

    """
    # training period 
    if control == 'Human': 
        pass 
    elif control == 'QLearning': 
        
        for i in range(training_period): 
            done = False 
            episode_reward = 0 
            max_steps = 20 # arbitrary max steps 
            n_step = 0
            while not done: 
                n_step += 1 
                if n_step > max_steps: 
                    done = True 
                action = q_agent.get_action(obs)
                print('action',action,'state',state,'obs',obs)
                next_state, next_obs, step_reward = simulator.step(action,state)
                
                experience = obs, action, next_obs, step_reward, done 
                
                episode_reward += step_reward 
                
                q_agent.train(experience) 
                state = next_state 
                obs = next_obs 
                
    
    
    

    # testing period 
    for i in range(testing_period):
        if control == 'Human': 
            action = int(input('What action to take: \n 0) Listen, 1) Left, 2) Right\n'))
        elif control == 'QLearning': 
            action = q_agent.get_action(obs)
        
    
        next_state, step_observation, step_reward = simulator.step(action,state)
        state = next_state
        print('Reward', step_reward)
        print('Observed', step_observation)
        total_reward+=step_reward
    print('Total Reward',total_reward)
    """
    
if __name__ == '__main__': 
    main()
    
