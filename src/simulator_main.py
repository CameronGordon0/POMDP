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

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.optimizers import Adam

# rockSample-3_1
# Tiger 
# rockSample-7_8
    
def main(file = '../examples/rockSample-7_8.pomdpx', 
         control = 'QLearning', 
         training_period = 10,
         testing_period = 1): 
    simulator = Simulator(file)
    simulator.print_model_information()
    state = simulator.initial_state
    state_name = simulator.state_name
    total_reward = 0
    #q_agent = QLearning(file)
    obs = state  # initial state = observation 
    
    
    control = 'Human' 
    if control == 'Human':
        for i in range(training_period):
            print('step ', i+1)
            action_name = list(simulator.actions.keys())[0]
            action_index = int(input('What action to take:\n'+str(simulator.actions[action_name])))
            action_name = simulator.actions[action_name][action_index]
            print('Took ', action_name)
        
            next_state, step_observation, step_reward = simulator.step(action_name,state)
            print(next_state,'\n', step_observation,'\n', step_reward,'\n')
            state = next_state
            total_reward += step_reward 
    print(total_reward)

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
    
