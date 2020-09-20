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
    
    - note: the above also occurs with different parameters for annealing, LSTM, Conv2D etc 
    - may require action being included in the input (ADQN, ADRQN), or longer run time (i.e. more exploration) 
    - also the case for changes to history length & training epochs (curr set at 1), however changes to run time 
    - also changes to convergence speed (rather than the end score) 
    - option to show difference may be to plot a running average of the scores? (may be more useful for the early annealing) 
    - difficulty appears to be linking the obs/(multiple action) connection without behaviour being extinguished 
    - worth testing the macro-action set up as the final layer (would require a few modifications to the loop) 
    -- may need to create a new main loop to run a macro-action set up in (as each action would run for multiple time steps) 
    
    - still have not implemented a Priority Experience Replay 
    
    
    - annealling for Priority Experience Replay seems to be extremely important (just sampling the top TD errors heavily biases to optimism) 
    - implementation on Priority Experience Replay draws on https://adventuresinmachinelearning.com/prioritised-experience-replay/ 
    - still need to anneal for importance sampling 
    
    



"""


from collections import deque 
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Conv2D, MaxPooling2D  
from keras.optimizers import Adam 
import keras.losses
import random 
import numpy as np 

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
                 Deep=False): 
        # define the action & the state shape 
        
        # this will probably involve concatinating the fully-observed parts of the state 
        # & the other observations in a numpy array either here or in the main loop 
        
        
        # need to include the history length 
        # may have issues with how this is defined for the first few entries??
        
        self.memory = deque(maxlen=2000) 
        #self.priority_array =
        #self.pq = queue.PriorityQueue(maxsize=2000)
        #print(self.pq)
        
        
        self.gamma = 0.99 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.tau = 0.05 
        
        self.state_matrix = state_matrix 
        self.action_vector = action_vector
        
        self.history = history
        self.DRQN = DRQN
        self.Dropout=Dropout
        self.Conv = Conv
        self.PriorityExperienceReplay = PriorityExperienceReplay 
        self.Deep = Deep 
        
        self.model = self.create_model()
        
        self.target_model = self.create_model() 
        
        
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
        #model.compile(loss=keras.losses.Huber(),
        #              optimizer=Adam(lr=self.learning_rate)) # trying out a different loss function - think the risk aversion comes from here. Tried mean_squared_logarithmic_error 
        return model 
    
    def calculate_TD_Error(self,state,action,reward,new_state): 
        state = self.state_numpy_conversion(state)
        new_state = self.state_numpy_conversion(new_state) 
        
        Q_current = self.model.predict(state)[0][action]
        Q_future = max(self.target_model.predict(new_state)[0])
        TD_Error = reward + self.gamma*Q_future - Q_current
        #print(TD_Error)
        return TD_Error 
    
    def calculate_priority(self,td_error):
        
        # attempting to skew it positive 
        # original was return np.power(abs(td_error)+self.min_priority,self.alpha) for both 
        
        original_calc = True  
        
        if original_calc: 
            return np.power(abs(td_error)+self.min_priority,self.alpha)
        else:
            if td_error > 0: 
                return np.power(abs(td_error)*abs(td_error)*abs(td_error)+self.min_priority,self.alpha)
                
            else: 
                return np.power(abs(td_error)/2+self.min_priority,self.alpha) 
    
    def calculate_sample_probability(self): 
        # returns a numpy array of the probability of sampling 
        # calculates at each run 
        #print(self.memory)
        
        mem_len = len(list(self.memory))
        #print(mem_len,'+++')
        sample_probabilities = np.zeros(mem_len)
        importance_weights = np.zeros(mem_len) # note: unsure how to standardise this 
        
        #priority_a = 
        total = 0 
        index = 0
        for mem in self.memory: 
            priority_a = self.calculate_priority(mem[5])
            total += priority_a 
            sample_probabilities[index] = priority_a 
            index += 1 
            
        #print(sample_probabilities.shape)
        
        
        
        for i in range(mem_len): 
            sample_probabilities[i] = sample_probabilities[i]/total 
            importance_weights[i] = np.power((1/mem_len)*(1/sample_probabilities[i]),self.beta) 
            
            # also need to normalize the importance_weights & decide how to sample according to the weights 
            
        #print(np.sum(sample_probabilities))
        return sample_probabilities 
    
        
    
    
            
        
    
    def remember(self, 
                 state, 
                 action, 
                 reward, 
                 new_state, 
                 done):
        # note: to convert this to prioritised experience replay, 
        # need to store the TD-Error in this tuple 
        
        #print('remember',type(new_state),new_state.shape)
        
        # step 1) Calculate the TD-Error 
        td_error = self.calculate_TD_Error(state,action,reward,new_state)
        
        # step 2) Store the tuple 
        if self.PriorityExperienceReplay: 
            # note this conversion is extremely inefficient 
            
            self.memory.append([state, action, reward, new_state, done, abs(td_error)])
            #print(self.memory)
            sort_list = sorted(self.memory, key=lambda mem: mem[5]) 
            
            
            #print(sort_list)
            self.memory = deque(sort_list,maxlen=5000)
            # step 3) sort by the TD-Error 
            
            # step 4) Annealling (prevent oversampling of top TD entries)
            
            #print(self.pq)
            
        else: 
            #print('wrong pass')
            self.memory.append([state, action, reward, new_state, done]) 
        
        
        
        # step 4) Annealling? 
        
        
        
    def replay(self):
        # note: need to modify this for PER (extract the TD-Error & sort the memory)
        
        batch_size = 32 # batch size reduced to force earlier use of memory 
        if len(self.memory) < batch_size: 
            return
        
        # select replay samples
        if self.PriorityExperienceReplay: 
            mem_len = len(self.memory)
            
            sample_probabilities = self.calculate_sample_probability()
            #temp_arr = np.asarray(self.memory) 
            sample_indexes = np.random.choice(mem_len,batch_size,replace=True,p=sample_probabilities)
            # use true or false here?? 
            
            samples = [self.memory[i] for i in sample_indexes]
            # note conversion to and from deque is super inefficient and becomes much worse as deque grows 
            #samples = list(self.memory)[-batch_size:] # correct sorting to get the top TD_Errors 
            # now need to anneal this 
            #print(len(samples))
            #print(samples)
        else:
            samples = random.sample(self.memory, batch_size)
        for sample in samples:
            if self.PriorityExperienceReplay:
                state, action, reward, new_state, done, td_error = sample
                #print(td_error)
            else:
                state, action, reward, new_state, done = sample
            
            # modified 
            
            """
            if self.history: 
                if self.Conv: 
                    #print('here')
                    state = np.reshape(state,(1,int(state.shape[0]),int(state.shape[1]),1))
                else:
                    state = np.reshape(state,(-1,int(state.shape[0]),state.shape[1]))
            else:
                state = np.reshape(state,(1,state.shape[0]))
            """
            
            #new_state = np.reshape(new_state,(1,new_state.shape[0]))# modified 
            state = self.state_numpy_conversion(state)
            """
            if self.history: 
                if self.Conv: 
                    new_state = np.reshape(new_state,(1,int(new_state.shape[0]),int(new_state.shape[1]),1))
                    #print('ere')
                else:
                    new_state = np.reshape(new_state,(-1,int(new_state.shape[0]),new_state.shape[1]))
            else:
                new_state = np.reshape(new_state,(1,new_state.shape[0]))
            """
            new_state = self.state_numpy_conversion(new_state)
            
            target = self.target_model.predict(state) # should the target be this state or the next state?? 
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
            # maybe changes to introduce PER? (to force more sampling of interesting spaces)
    
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
        """
        if self.history: 
            if self.Conv: 
                state = np.reshape(state,(1,int(state.shape[0]),int(state.shape[1]),1))
                #print('look here',state.shape)
            else:
                state = np.reshape(state,(-1,int(state.shape[0]),int(state.shape[1])))
        else:
            state = np.reshape(state,(1,state.shape[0]))
        """
        state = self.state_numpy_conversion(state)
        #print(state)
        #print('---',state.shape)
        """
        Note: not entirely certain about this reshaping method, but enables the function to run 
        """
        #print('act check',state)
        #print(self.model.predict(state)[0])

        
        return np.argmax(self.model.predict(state)[0]) ## 
    
    
    ## note further details required to be called within the main loop of the simulator 
    
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
