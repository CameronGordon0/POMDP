#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:45:26 2020

@author: camerongordon
"""

from pomdp_simulator import Simulator
import numpy as np 


class DQN(Simulator): 
    def __init__(self, 
                 file,
                 discount_rate = 0.9, 
                 learning_rate = 0.01, 
                 memory_size=1000): 
        super().__init__(file) 
        
        self.eps = 1 # set epsilon exploration parameter 
        
        self.discount_rate = discount_rate 
        self.learning_rate = learning_rate 
        self.memory_size = memory_size 
        self.memory = deque(maxlen = memory_size) 
        self.state_memory = [] 
        
        self.build_model() 
        
        def build_model(self): 
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(len(self.observation_name),), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(len(self.action_name), activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
            return self.model 
    
        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done)) 
            
        def act(self, state):
            if np.random.rand() < self.eps: # simply the decision 
                return random.randrange(len(self.action_name))
            q_values = self.model.predict(state) # interesting - generates a predicted q_value 
            return np.argmax(q_values[0]) # returns the max out of the prediction 
        
        
        def experience_replay(self):
            """
            Contains the key experience replay buffer - 
            What would be required to make this a prioritised replay buffer? 
            
            Note: apparently LSTM does not play nicely with the randomised experience replay 
            https://ai.stackexchange.com/questions/7721/how-does-lstm-in-deep-reinforcement-learning-differ-from-experience-replay 
            """
            
            if len(self.memory) < BATCH_SIZE:
                return
            batch = random.sample(self.memory, BATCH_SIZE)
            for state, action, reward, state_next, terminal in batch:
                q_update = reward
                # essentially an interation of q-learning 
                if not terminal:
                    print('predict',self.model.predict(state_next))
                    q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                q_values = self.model.predict(state)
                q_values[0][action] = q_update # take the values for the immediate state 
                # need to modify the model.fit function to incorporate LSTM, will require some more updates 
                self.model.fit(state, q_values, verbose=0) # linear fit of state & values (i.e. model) 
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        
        def add_run_memory(self, run_memory): 
            self.state_memory.append(run_memory) # add trajectory for a given run 