# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:58:22 2020

@author: MrCameronGordon

28/9/2020 Attempting to refactor the DQN class. 
Main goals are to improve readability, check the implementations of priority experience replay. 

Additional goal: implement DDQN 


"""

from collections import deque 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, LSTM, Dropout, Input  
from keras.optimizers import Adam 
import keras.losses
import random 
import numpy as np 


class DQN: 
    def __init__(self):
        
        pass
    
    def create_model(self, 
                     model_type = "basic",
                     L1 = 50, 
                     L2 = 50, 
                     L3 = 50):
        
        state_shape = self.state_matrix.shape 

        if model_type == "basic": 
            # basic model 
            # 3 layers, relu activations, mean_squared_error  
        
            model = Sequential() 
            model.add(Dense(L1, input_shape=state_shape, activation = "relu"))
            model.add(Dense(L2, activation = "relu"))
            model.add(Dense(L3, activation = "relu"))
            model.add(Flatten())
            

            
        if model_type == "LSTM": 
            # DRQN - input layer, LSTM 
            model = Sequential() 
            model.add(Dense(L1, input_shape=state_shape, activation = "relu"))
            model.add(LSTM(100)) 
            
        if model_type == "ResNet": 
            # use skip connections, based on UNET (up and down) 
            
            inputs = Input(shape=state_shape)
            
            layer1 = Dense(L1, activation='relu')(inputs)
            layer2 = Dense(50, activation = 'relu')()
            
            
            
        model.add(Dense(len(self.action_vector))) # output layer 

        model.compile(loss= "mean_squared_error",
                      optimizer = Adam(lr = self.learning_rate)) 
        return model 
        
        

        
        pass 
    def calculate_TD_Error(self): 
        pass 
    
    def calculate_priority(self): 
        pass 
    
    def calculate_sample_probability(self): 
        pass 
    
    def remember(self): 
        pass 
    
    def replay(self): 
        pass 
    
    def target_train(self): 
        pass 
    
    def act(self): 
        pass 
    
    def state_numpy_conversion(self): 
        pass 