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


Note: 
    - on Tiger the model appears to converge to the 'no history' action (i.e. 'listen' always) 
    - on rockSample-3_1 converges to a score of 10 
    - on rockSample-7_8 converges to a score of 10
    -- think this may be due to it homing in on the terminal state & avoiding searching for other options 



"""


from collections import deque 
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Conv2D, MaxPooling2D  
from keras.optimizers import Adam 
import random 
import numpy as np 


USE_GPU = False 

"""
Note: https://towardsdatascience.com/deep-learning-using-gpu-on-your-macbook-c9becba7c43 

Gives details of using GPU & Keras using PlaidML library. Supposed to give speed-up. 
"""

class DQN: 
    
    
    def __init__(self,
                 action_vector,
                 state_matrix,
                 history = False,
                 history_len = 0,
                 DRQN = False,
                 Dropout = False,
                 Conv = False): 
        # define the action & the state shape 
        
        # this will probably involve concatinating the fully-observed parts of the state 
        # & the other observations in a numpy array either here or in the main loop 
        
        
        # need to include the history length 
        # may have issues with how this is defined for the first few entries??
        
        self.memory = deque(maxlen=2000) 
        self.gamma = 0.99 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.02
        self.tau = 0.05 
        
        self.state_matrix = state_matrix 
        self.action_vector = action_vector
        
        self.history = history
        self.DRQN = DRQN
        self.Dropout=Dropout
        self.Conv = Conv
        
        self.model = self.create_model()
        
        self.target_model = self.create_model() 
        
        
    def create_model(self,
                     L1=30,
                     L2=30,
                     L3=30):
        # defined L1,L2,L3 as the neurons in a layer 
        model   = Sequential()
        #state_shape  = self.env.observation_space.shape
        state_shape = self.state_matrix.shape # need to define by the simulator 
        print('state',state_shape)
        #action_shape = self.action_vector.shape 
        print('DQN state shape',state_shape)
        
        if self.Conv: 
            state_shape = (state_shape[0],state_shape[1],1)
            model.add(Conv2D(filters=64,kernel_size = 2,input_shape=state_shape,activation="relu"))
            #model.add(MaxPooling2D(pool_size=2))
            #model.add(Conv2D(filters=32,kernel_size = 2,activation="relu"))
            #model.add(MaxPooling2D(pool_size=2))
            #model.add(Conv2D(filters=16,kernel_size = 2,activation="relu"))
            #model.add(MaxPooling2D(pool_size=2))
        else:
            model.add(Dense(L1, input_shape=state_shape, 
            activation="relu"))
        model.add(Dense(L2, activation="relu"))
        if self.Dropout: 
            model.add(Dropout(0.4))
        model.add(Dense(L3, activation="relu"))
        
        if self.history == True and not self.DRQN: # this one's interesting - it doesn't like Flatten & LSTM together for dimension reasons 
            model.add(Flatten())
            
        if self.DRQN: 
            model.add(LSTM(100))
        
        model.add(Dense(len(self.action_vector)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model 
    
    def remember(self, 
                 state, 
                 action, 
                 reward, 
                 new_state, 
                 done):
        # note: to convert this to prioritised experience replay, 
        # need to store the TD-Error in this tuple 
        self.memory.append([state, action, reward, new_state, done]) 
        
    def replay(self):
        # note: need to modify this for PER (extract the TD-Error & sort the memory)
        
        batch_size = 32 # batch size reduced to force earlier use of memory 
        if len(self.memory) < batch_size: 
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            
            # modified 

            if self.history: 
                if self.Conv: 
                    #print('here')
                    state = np.reshape(state,(1,int(state.shape[0]),int(state.shape[1]),1))
                else:
                    state = np.reshape(state,(-1,int(state.shape[0]),state.shape[1]))
            else:
                state = np.reshape(state,(1,state.shape[0]))
            
            #new_state = np.reshape(new_state,(1,new_state.shape[0]))# modified 
            
            if self.history: 
                if self.Conv: 
                    new_state = np.reshape(new_state,(1,int(new_state.shape[0]),int(new_state.shape[1]),1))
                    #print('ere')
                else:
                    new_state = np.reshape(new_state,(-1,int(new_state.shape[0]),new_state.shape[1]))
            else:
                new_state = np.reshape(new_state,(1,new_state.shape[0]))
            
            
            target = self.target_model.predict(new_state) # should the target be this state or the next state?? 
            if done:
                target[0][action] = reward
            else:
                #print('check',new_state.shape)
                Q_future = max(
                    self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
            # note: original model here had epochs set as 1 
            # do not see significant performance improvement (indeed may become overfit)
    
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights) 
        
        
    def act(self, state):
        # Note: need to modify this one for the pomdpx environment details 
        
        # state here is going to be the numpy obs-state that generated in main 
        
        #self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        # I think this is a key, key issue. 
        # definitely anneals way too fast
        #print(self.epsilon)
        
        
        
        if np.random.random() < self.epsilon: #  
            #print('something happened')
            #print(self.epsilon)
            return self.action_vector.index(random.choice(self.action_vector)) ## modified 
        #print('something else happened')
        
        #print('state before choosing',state)
        #print(state.shape)
        
        if self.history: 
            if self.Conv: 
                state = np.reshape(state,(1,int(state.shape[0]),int(state.shape[1]),1))
                #print('look here',state.shape)
            else:
                state = np.reshape(state,(-1,int(state.shape[0]),int(state.shape[1])))
        else:
            state = np.reshape(state,(1,state.shape[0]))
        
        #print(state)
        #print('---',state.shape)
        """
        Note: not entirely certain about this reshaping method, but enables the function to run 
        """
        #print('act check',state)
        

        
        return np.argmax(self.model.predict(state)[0]) ## 
    
    
    ## note further details required to be called within the main loop of the simulator 
    
