#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file runs a simulator for the parsed pomdpx files. 

Currently manual input for the Tiger problem. 

To do: 
    - Change to automated input 
    - Enable running on rockSample & switching between problems 
    - Correct parser bugs for AUV and Tag pomdpx files 

"""


from parser_main import Parser 
import numpy as np 
from qlearning import QLearning
from dqn import DQN

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.optimizers import Adam



class Simulator(): 
    def __init__(self, model_filename):
        parser = Parser(model_filename)
        self.state = parser.states
        print(self.state)
        self.state_name = self.state['state_0'][0]
        self.action_name = parser.actions['action_agent']
        self.transition = parser.state_transition[0] 
        self.reward = parser.reward_table
        self.initial_belief = parser.initial_belief
        self.observation = parser.obs_table
        self.observation_name = parser.observations['obs_sensor']
        
        self.total_reward = 0 
        self.initial_state = self.state_name.index(np.random.choice(self.state_name,p=self.initial_belief))
                                                   
    def print_model_information(self): 
        print('State', self.state_name)
        print('Action', self.action_name)
        print('Observation', self.observation_name)
        print('Observation R',self.observation)
        
        


    def step(self, action,state): 
        step_reward = self.reward[action][state]
        next_state_distribution = self.transition[action,state]
        next_state = np.random.choice(self.state_name,p=next_state_distribution)
        next_state = self.state_name.index(next_state)
    
        step_observation = np.random.choice(self.observation_name,p=self.observation[action,next_state])
        
        step_observation = self.observation_name.index(step_observation)
        
        return next_state, step_observation, step_reward
    
    
    

            
            
        
        
        
        
    
    
    
    
    
def main(file = '../examples/rockSample-3_1.pomdpx', 
         control = 'QLearning', 
         training_period = 100): 
    simulator = Simulator(file)
    simulator.print_model_information()
    state = simulator.initial_state
    state_name = simulator.state_name
    total_reward = 0
    q_agent = QLearning(file)
    obs = state  # initial state = observation 

    
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
    for i in range(100):
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
    
if __name__ == '__main__': 
    main()