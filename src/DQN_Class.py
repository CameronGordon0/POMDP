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

DQN file - contains modules for custom loss functions (flooding), prioritised experience buffer, and other specifications. 

"""


from collections import deque 
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Conv2D, MaxPooling2D, Input, Conv2DTranspose, concatenate  
from tensorflow.keras.models import Model

from keras.optimizers import Adam 
import keras.losses
import random 
import numpy as np 

import keras.backend as K 

#import queue 


USE_GPU = False 



"""
Note: https://towardsdatascience.com/deep-learning-using-gpu-on-your-macbook-c9becba7c43 

Gives details of using GPU & Keras using PlaidML library. Supposed to give speed-up. 
"""

class DQN: 
    
    
    def __init__(self,
                 action_vector,
                 state_matrix,
                 history = True,
                 history_len = 0,
                 DRQN = False,
                 Dropout = False,
                 Conv = False, 
                 PriorityExperienceReplay=True,
                 Deep=False,
                 useflooding=True, 
                 learning_rate = 0.01,
                 batch_size = 32,
                 network_width=50,
                 flooding_value=0, 
                 LSTM_only = False,
                 LSTM_len = 0): 

        
        self.memory = deque(maxlen=2000) 

        self.network_width = network_width
        
        self.useflooding = useflooding
        self.flooding_value = 0
        self.LSTM_only = LSTM_only 
        self.LSTM_len = LSTM_len
        
        
        self.gamma = 0.99 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = learning_rate
        self.tau = 0.05 # rate of averaging (for training networks) 
        self.batch_size = batch_size
        
        self.state_matrix = state_matrix 
        self.action_vector = action_vector
        
        self.history = history
        self.DRQN = DRQN
        self.Dropout=Dropout
        self.Conv = Conv
        self.PriorityExperienceReplay = PriorityExperienceReplay 
        self.Deep = Deep 
        
        self.model = self.create_model(L1 = self.network_width,
                                              L2 = self.network_width, 
                                              L3 = self.network_width) 
        
        self.target_model = self.create_model(L1 = self.network_width,
                                              L2 = self.network_width, 
                                              L3 = self.network_width) 
        
        self.model.summary()
        
        #self.model = self.create_unet()
        #self.target_model = self.create_unet()
        
        if self.PriorityExperienceReplay: 
            # set a couple of the parameters required for importance sampling & prioritised experience replay 
            self.alpha = 0.6 
            self.beta = 0.5 # use this for calculating the importance sampling weight 
            self.min_priority = 0.01 
            #self.sample_probabilities = np.zeros(2000) # initialise a sample probability vector 
            
        self.training_delay = None 
        self.current_iteration = 0 
        
        
    def create_model(self,
                     L1=50,
                     L2=50,
                     L3=50):
        # defined L1,L2,L3 as the neurons in a layer 
        # each to try new architectures (e.g. autoencoding)
        
        model   = Sequential()
        #state_shape  = self.env.observation_space.shape
        state_shape = self.state_matrix.shape # need to define by the simulator 
        print('state',state_shape)
        #action_shape = self.action_vector.shape 
        print('DQN state shape',state_shape)
        
        if not self.LSTM_only:

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
            if self.Deep:
                model.add(Dense(L2, activation="relu"))
                model.add(Dense(L2, activation="relu"))
                model.add(Dense(L2, activation="relu"))
                model.add(Dense(L2, activation="relu"))
                model.add(Dense(L2, activation="relu"))
                
            if self.Dropout: 
                model.add(Dropout(0.1))
            model.add(Dense(L3, activation="relu"))
            
            if self.history == True and not self.DRQN: # this one's interesting - it doesn't like Flatten & LSTM together for dimension reasons 
                model.add(Flatten())
                
            if self.DRQN: 
                model.add(LSTM(50))
                
        if self.LSTM_only:
            model.add(LSTM(self.LSTM_len,input_shape = state_shape))
            print('added lstm')
            
        model.add(Dense(len(self.action_vector)))
        
        if (self.useflooding):
            model.compile(loss=self.custom_loss_function,
                          optimizer=Adam(lr=self.learning_rate))#,  metrics=['accuracy']) 
        else: 
            model.compile(loss=self.custom_loss_function,
                          optimizer=Adam(lr=self.learning_rate))#,
                          #metrics=['accuracy']) 
        return model 
    
    def create_unet(self): # an attempted network structure 
        state_shape = self.state_matrix.shape
        inputs = Input(shape=state_shape) 
        
        L1 = Dense(5)(inputs)
        L2 = Dense(5)(L1)
        L3 = Dense(5)(L2)
        
        L4 = Dense(5)(L3) 
        cat1 = concatenate([L4, L2])
        L5 = Dense(5)(cat1)
        #cat2 = concatenate([L1,L5])
        L6 = Flatten()(L5)
        
        output = Dense(len(self.action_vector))(L6)

        model = Model(inputs, output)
        
        model.compile(optimizer='adam', loss='mse')
        return model 
    
    def calculate_TD_Error(self,state,action,reward,new_state): 
        state = self.state_numpy_conversion(state)
        new_state = self.state_numpy_conversion(new_state) 
        
        Q_current = self.model.predict(state)[0][action]
        Q_future = max(self.target_model.predict(new_state)[0])
        
        TD_Error = reward + self.gamma*Q_future - Q_current
        return TD_Error 
    
    def calculate_priority(self,td_error):
        
        original_calc = True  
        
        if original_calc: 
            return np.power(abs(td_error)+self.min_priority,self.alpha)
        else:
            if td_error > 0: 
                return np.power(abs(td_error)*abs(td_error)*abs(td_error)+self.min_priority,self.alpha)
                
            else: 
                return np.power(abs(td_error)/2+self.min_priority,self.alpha) 
    
    def calculate_sample_probability(self): 

        
        mem_len = len(list(self.memory))
        sample_probabilities = np.zeros(mem_len)
        importance_weights = np.zeros(mem_len) 
        
        total = 0 
        index = 0
        for mem in self.memory: 
            priority_a = self.calculate_priority(mem[5])
            total += priority_a 
            sample_probabilities[index] = priority_a 
            index += 1 
            
        for i in range(mem_len): 
            sample_probabilities[i] = sample_probabilities[i]/total 
            importance_weights[i] = np.power((1/mem_len)*(1/sample_probabilities[i]),self.beta) 
            
            
        return sample_probabilities 
    
    def remember(self, 
                 state, 
                 action, 
                 reward, 
                 new_state, 
                 done):

        td_error = self.calculate_TD_Error(state,action,reward,new_state)
        
        if self.PriorityExperienceReplay: 
            # note this conversion is extremely inefficient 
            self.memory.append([state, action, reward, new_state, done, abs(td_error)])
            sort_list = sorted(self.memory, key=lambda mem: mem[5]) 
            
            self.memory = deque(sort_list,maxlen=5000)
            
        else: 
            self.memory.append([state, action, reward, new_state, done])      
        
        
    def replay(self):
        
        batch_size = self.batch_size # batch size reduced to force earlier use of memory 
        if len(self.memory) < batch_size: 
            return
        
        # select replay samples
        if self.PriorityExperienceReplay: 
            mem_len = len(self.memory)
            
            sample_probabilities = self.calculate_sample_probability()
            sample_indexes = np.random.choice(mem_len,batch_size,replace=True,p=sample_probabilities)
            
            samples = [self.memory[i] for i in sample_indexes]

        else:
            samples = random.sample(self.memory, batch_size)
        for sample in samples:
            if self.PriorityExperienceReplay:
                state, action, reward, new_state, done, td_error = sample
            else:
                state, action, reward, new_state, done = sample
            

            state = self.state_numpy_conversion(state)

            new_state = self.state_numpy_conversion(new_state)
            
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
            #target_weights[i] = weights[i]
            target_weights[i] = self.tau*weights[i] + (1-self.tau)*target_weights[i]
            # note: modifying according to - https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        self.target_model.set_weights(target_weights) 
        
        
    def act(self, state): 
        """
        Applies an epsilon greedy action. i.e. randomises if < epsilon,
        takes the greedy best action otherwise
        """
        

        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon: #  
            return self.action_vector.index(random.choice(self.action_vector)) ## modified 

        state = self.state_numpy_conversion(state)

        return np.argmax(self.model.predict(state)[0]) 
        
    def state_numpy_conversion(self,state): 
        # covnerts the numpy_shape to the appropriate dimensions 
        
        if self.history: 
            if self.Conv: 
                state = np.reshape(state,(1,int(state.shape[0]),int(state.shape[1]),1))
                #print('look here',state.shape)
            else:
                state = np.reshape(state,(-1,int(state.shape[0]),int(state.shape[1])))
        else:
            state = np.reshape(state,(1,state.shape[0]))
        
        return state 
    
    def custom_loss_function(self,y_true, y_pred): 
        """
        Implements a custom MSE loss function as described in 
        https://towardsdatascience.com/how-to-create-a-custom-loss-function-keras-3a89156ec69b
        
        This is in order to use 'flooding'. (i.e. regularising the objective J*=|J-b|+b)
        
        We set b as 0.05 here (value is a design decision)
        """
        
        #flooding_value = 0
        loss = K.abs(K.square(y_pred - y_true)-self.flooding_value)+self.flooding_value
        print(loss)
        
        return loss
