#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:25:27 2020

@author: camerongordon 

Note: DRQN is the same architecture as DQN
Except for the second to last layer (LSTM) 

Ideally, want to see the following expansions: 
    - Prioritised Experience Replay 
    - Double Q-Network 
"""


from collections import deque 
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam 
import random 
import numpy as np 


class DRQN:  
    
    def __init__(self): 
        pass 
    
    def build_model(self): 
        pass 
    
    def remember(self): 
        pass 
    def replay(self): 
        pass 
    def target_train(self): 
        pass 
    
    def act(self): 
        pass 
    