#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:26:43 2020

@author: camerongordon 

Note: drawing from https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c 
As example DQN model. Note modifications required to draw this from the Gym format. 
Alternate option considered: creating a custom gym environment using the simulator. 

Design principle: 
    - all information passed to the agent should be in vector / matrix form 
    - no inheritance from simulator class - keep the environments separate  
    - main difficulty may be deciding how to pass the observation / action spaces to the agent 
    
Initial model: 
    - No prioritised experience replay, only simple experience replay 

"""


from collections import deque 
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam 
import random 
import numpy as np 

class DQN: 
    
    
    def __init__(self,action_vector,state_matrix): 
        # define the action & the state shape 
        
        # this will probably involve concatinating the fully-observed parts of the state 
        # & the other observations in a numpy array either here or in the main loop 
        
        self.memory = deque(maxlen=2000) 
        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.04 
        self.tau = 0.05 
        
        self.state_matrix = state_matrix 
        self.action_vector = action_vector
        
        
        
        self.model = self.create_model()
        
        self.target_model = self.create_model() 
        
    def create_model(self):
        model   = Sequential()
        #state_shape  = self.env.observation_space.shape
        state_shape = self.state_matrix.shape # need to define by the simulator 
        #action_shape = self.action_vector.shape 
        
        
        #print('???????????',state_shape[0])
        
        model.add(Dense(100, input_dim=state_shape[0], 
            activation="relu"))
        model.add(Dense(70, activation="relu"))
        model.add(Dense(30, activation="relu"))
        model.add(Dense(len(self.action_vector)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model 
    
    def remember(self, state, action, reward, new_state, done):
        # note: to convert this to prioritised experience replay, 
        # need to store the TD-Error in this tuple 
        self.memory.append([state, action, reward, new_state, done]) 
        
    def replay(self):
        # note: need to modify this for PER (extract the TD-Error & sort the memory)
        
        batch_size = 32
        if len(self.memory) < batch_size: 
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = np.reshape(state,(1,state.shape[0])) # modified 
            new_state = np.reshape(new_state,(1,new_state.shape[0]))# modified 
            
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(
                    self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
    
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights) 
        
        
    def act(self, state):
        # Note: need to modify this one for the pomdpx environment details 
        
        # state here is going to be the numpy obs-state that generated in main 
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon: #  
            #print('something happened')
            #print(self.epsilon)
            return self.action_vector.index(random.choice(self.action_vector)) ## modified 
        #print('something else happened')
        
        #print('state before choosing',state)
        #print(state.shape)
        state = np.reshape(state,(1,state.shape[0]))
        #print(state)
        #print('---',state.shape)
        """
        Note: not entirely certain about this reshaping method, but enables the function to run 
        """
        
        return np.argmax(self.model.predict(state)[0]) ## 
    
    
    ## note further details required to be called within the main loop of the simulator 
    
