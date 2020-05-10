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

model_filename = '../examples/Tiger.pomdpx'

parser = Parser(model_filename)


state = parser.states
state_name = state['state_0'][0]
action_name = parser.actions['action_agent']
transition = parser.state_transition[0] 
reward = parser.reward_table
initial_belief = parser.initial_belief
observation = parser.obs_table
observation_name = parser.observations['obs_sensor']

def step(action,state,name): 
    step_reward = reward[action][state]
    next_state_distribution = transition[action,state]
    next_state = np.random.choice(name,p=next_state_distribution)

    step_observation = np.random.choice(observation_name,p=observation[action,state_name.index(next_state)])
    return next_state, step_observation, step_reward

state = state_name.index(np.random.choice(state_name,p=[0.5,0.5]))
total_reward = 0 
for i in range(10):
    action = int(input('What action to take: \n 0) Listen, 1) Left, 2) Right\n'))

    next_state, step_observation, step_reward = step(action,state,state_name)
    state = state_name.index(next_state)
    print('Reward', step_reward)
    print('Observed', step_observation)
    total_reward+=step_reward
print('Total Reward',total_reward)