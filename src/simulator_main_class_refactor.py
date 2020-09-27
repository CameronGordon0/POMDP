# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:27:22 2020

@author: MrCameronGordon

19/9/2020 - Attempting to refactor the simulator main into a class structure. 

Main aim is to improve readability, reduce cross dependency, and improve testing. 

Another aim is potential unit testing. 

pomdp_simulator is relatively cleanly structured. It may be worth shifting some methods here. 
"""

import numpy as np 
from pomdp_simulator import Simulator
import random 
import matplotlib.pyplot as plt 
from DQN_Class import DQN 


model_name = {'../examples/Tiger.pomdpx':'Tiger',
              '../examples/rockSample-3_1.pomdpx':'Rock Sample (3,1)',
              '../examples/rockSample-7_8.pomdpx':'Rock Sample (7,8)',
              '../examples/rockSample-10_10.pomdpx':'Rock Sample (10,10)',
              '../examples/rockSample-11_11.pomdpx':'Rock Sample (11,11)',
              '../examples/Tag.pomdpx':'Tag',
              '../examples/auvNavigation.pomdpx': 'AUV',
              '../examples/functional_imitation.pomdpx':'functional_imitation'} 

terminal_states = {'../examples/Tiger.pomdpx':None,
              '../examples/rockSample-3_1.pomdpx':'s2',
              '../examples/rockSample-7_8.pomdpx':'st',
              '../examples/rockSample-10_10.pomdpx':'st',
              '../examples/rockSample-11_11.pomdpx':'st',
              '../examples/Tag.pomdpx':'tagged',
              '../examples/auvNavigation.pomdpx': 'ST',
              '../examples/functional_imitation.pomdpx':'functional_imitation'} 

class simulatorMain(): 
    
    def __init__(self, file='../examples/rockSample-7_8.pomdpx',
                 training_period=100,
                 verbose=False,
                 history=True,
                 history_len=50,
                 maxsteps=50,
                 include_actions=True,
                 recurrent=False,
                 priority_replay=True,
                 training_delay=0): 
        
        self.simulator = Simulator(file) 
        self.file=file
        self.verbose = verbose 
        self.training_period = training_period 
        
    
        self.history = history
        self.history_len = history_len 
        
        self.maxsteps = maxsteps 
        
        self.max_seen = -1000
    
        self.include_actions = include_actions 
        self.priority_replay = priority_replay 
        self.recurrent = recurrent
        
        self.fixed_initial = False 
        self.include_reward = False # note: this is an idea which we'll test later (including the reward to the observation)
        
        self.training_delay = training_delay 
        
        
        self.training_details = [] # diagnostics 
        self.results_y = [] # note append is O(1) 
        self.results_x = np.arange(0,training_period)
        
        self.state_key_list = self.simulator.state_key_list 
        self.observation_key_list = self.simulator.observation_key_list 
        
        self.action_keys = list(self.simulator.actions.keys())[0] 
        self.action_list = self.simulator.actions[self.action_keys]
        #print(self.action_list)
        self.action_n = len(self.action_list) 
        self.action_space = np.zeros(self.action_n)
        
        
        self.observation_space_length = self.get_observation_space() 

        self.reset() 

        
        

        
        self.dqn = DQN(self.action_list,
                       self.history_space,
                       history=self.history,
                       DRQN=self.recurrent,
                       PriorityExperienceReplay = self.priority_replay) 
        
        

    
    def run(self):
        """
        we could otherwise call this 'run' 
        contains these functionalities: 
        1) initialises the simulator 
        2) runs control_method
        3) plots_results
        
        """
        
        self.control_method()
        self.write_to_csv() 
        self.plot_results() 
         
    
    # these are data conversion utilities 
    def history_queue(self, new_observation=None, old_history=None): 
        """
        Parameters: 
            new_observation (numpy array) 
            old_history (numpy array) 
            
        Description: 
            Takes in a single observation and the current history, 
            to return a new history containing the new observation.
            Acts as a queue for the numpy arrays (FIFO) 
        
        # note that the new_observation needs to have the same dimensions as the old history
        
        # see example: 
            # old_history = np.zeros((5,1,3))
            # old_history[-1] = np.ones((1,3))
            # new_observation = 2*np.ones((1,1,3))
            # new_hist = history_queue(new_observation,old_history)
            
        Return: 
            new_history (numpy array) 
        
        """
        self.old_history = self.history_space[:] 
        #print(old_history.shape,new_observation.shape)
        self.history_space = self.history_space[1:] # copies the old history less the oldest observation 
        self.history_space = np.append(self.history_space,self.numpy_observation,axis=0) 
        
        
        #print('old',self.old_history)
        #print('new',self.history_space)
        #return new_history  

    
    def numpy_conversion(self): 
        """
        Converts the history to a numpy array (for passing to DQN) 
        
        Note: currently has a lot of dependencies (simulator, observed_current_state, history (bool), history_len, include_actions (bool), previous_action_index=None (unused)) 
        
        Got a few question marks here - so want to test it properly 
        """ 
        # insert check that zero_space only contains zeros here (i.e. not being changed in place) 
        zero_space = np.zeros((1,self.observation_space_length))
        #print(zero_space)
        
        length = 0 
        for i in self.current_observable_state: 
            obj = self.current_observable_state[i]
            index = self.simulator.state[i][0].index(obj) # pulls the index of the observed state variables 
            length += len(self.simulator.state[i][0]) 
            
            zero_space[0][index] = 1 
            
        for i in self.current_observation: 
            obj = self.current_observation[i]
            index = self.simulator.observation_names[i].index(obj) # pulls the index of the observation 
            zero_space[0][length+index] = 1 
            
        if self.include_actions:
            for key in self.simulator.observation_key_list: 
                length += len(self.simulator.observation_names[key]) 
            zero_space[0][length+self.previous_action_index] = 1 
            
        #zero_space.reshape(zero_space,(1,self.observation_space_length))
        #print(zero_space.shape)
            
        return zero_space
    
    def update(self): 
        self.numpy_observation = self.numpy_conversion()
        self.history_queue()
        
        # easiest way of handling the necessary memory update (s0, s1) 
        # may be to store a self.previous_history() and a self.new_history() 
        # may require changes, as currently all in-place conversions 
        
        
        #print(self.history_space)
        
        
        
        
        
         
    
    def get_observation_space(self): 
        """        
        Initialises a one-hot encoded history buffer. 
        This is stored as a numpy array of length x history_len 
        Where length is 'fully observed state + observation + [actions + reward]' 
        Where include_actions, include_reward, history are options 
        
        Returns: observation_space_length (int)
        """
        
        length = 0 
        for key in self.state_key_list: 
            if self.simulator.state[key][1] == 'true': 
                length+= len(self.simulator.state[key][0]) # pulls out the state = s1, s2, ..., st etc information 
        for key in self.observation_key_list: 
            length+=len(self.simulator.observation_names[key]) 
            
        if self.include_actions: 
            length+= self.action_n 
            
        if self.include_reward: 
            length += 1 
            
        return length 
            
        
        
    
    # these are simulating utilities 
    
    def reset(self): 
        """
        Resets the current_state and the current_observable_state according to the simulator. 
        
        Note: design choice to store the current_state and current_observable_state as class attributes. 
        
        """
        self.current_state = self.simulator.set_initial_state() 
        self.current_observable_state = self.simulator.get_observable_state(self.current_state) 
        self.history_space = np.zeros((self.history_len,self.observation_space_length))
        self.current_observation = {}
        for i in self.simulator.observation_key_list: 
            self.current_observation[i] = random.choice(self.simulator.observation_names[i]) # initial random observation 
            
        self.previous_action_index = np.random.choice(self.action_n)

        self.update() # updates the history queue 

            
        
        
        
    
    def control_method(self): 
        """
        Dependencies: simulator, control (Random, Human, DQN), training_period (int), 
        verbose (bool), history (bool), history_len (int), maxsteps (int), 
        include_actions (bool), fixed_initial (bool), recurrent (bool), 
        training_delay (int), priority_replay (bool) 
        
        Functionalities: 
            Runs the main logic for the training and simulation 
            Runs the diagnostic data and write_to_csv 
            
            
        Note: currently very unwieldy, with multiple paths based on boolean values. 
        Should shift as many of the dependencies into a class attribute as possible. 
        Should split into sub-methods. Suggested: run_step, run_episode, run_diagnostics. 
        There's a lot of code here that's just dealing with what should be attributes, 
        and data conversion. 
        Should make this for DQN only (if need to have other methods, can do this separately) 
        
        """
        
        
        self.dqn.epsilon_decay = np.exp((np.log(0.01))/(0.5*self.training_period)) 
        self.dqn.training_delay = self.training_delay # may draft without training delay 
        
        for iteration in range(self.training_period): 
            self.dqn.epsilon *= self.dqn.epsilon_decay 
            
            total_reward = 0 
            #episode_details = [] 
            
            self.reset() # reset to randomised current_state and current observable state 
            
            
            done = False 
            
            #max_seen = -1000 
            
            if self.verbose: 
                print('iteration', iteration)
                
            for step in range(self.maxsteps): # assume we're going to use DQN for all here 
                if done == True: 
                    break 
                if self.verbose: 
                    print('step', step + 1) 
                
                it_hist = self.dqn.state_numpy_conversion(self.history_space)
                
                q_vals = self.dqn.model.predict(it_hist)[0] # need to re look at this 
                #q_vals = None
                action_index = self.dqn.act(self.history_space) 
                #print(action_index)
                
                action_taken = self.simulator.actions[self.action_keys][action_index] 
                self.previous_action_index = action_index
                
                next_state, step_observation, step_reward, observable_state = self.simulator.step(action_taken, self.current_state)
                #print(next_state,self.current_state)
                #print(self.current_observation,step_observation)
                #print(action_taken)
                
                
                total_reward += step_reward 
                

                
                self.training_details.append([iteration, step, self.dqn.epsilon, action_taken, step_observation, self.current_observable_state, self.current_state,step_reward, total_reward, q_vals])
                
                self.current_state = next_state 
                
                self.current_observable_state = self.simulator.get_observable_state(self.current_state)
                
                self.current_observation = step_observation # need to check 
                
                if self.verbose: 
                    print('Action taken', action_taken) 
                    print('State ', next_state,'\n Observation ',step_observation,'\n', step_reward,'\n') 
                    
                #self.create_memories() # need to check the details here 
                

                
                
                self.update() 
                done = self.check_if_terminal() 
                
                self.create_memories(step_reward,done)
                
                # need to create this method 
                
            self.dqn.replay() 
            self.dqn.target_train() # need to check both of these methods in the DQN_Class during refactor 
            
            if total_reward > self.max_seen: 
                self.max_seen = total_reward 
            print('iteration', iteration, total_reward, 'epsilon', round(self.dqn.epsilon,2),'best seen', self.max_seen) 
            
            self.results_y.append(total_reward) 
            
            #self.training_details.append(episode_)
                
                
                
            
            
            
            
            
            
        
        
        
        
        
        pass 
    
    def create_memories(self, step_reward,done): 
        """ 
        Dependencies: simulator, observable_state, observation, history, 
        include_actions, previous_action_index,action_index,next_state, 
        step_observation, iteration_history, maxsteps, state, dqn, step_reward 
        
        Note: excessive dependencies, need to strip these out 
        Only purpose is to act as a wrapper for dqn 
        """
        # may be easiest to just convert them directly in here 
        
        
        
        
        
        
        self.dqn.remember(self.old_history,self.previous_action_index,step_reward,self.history_space,done) #### need to do properly. Actually think about this. 
        pass # note this is just a conversion and wrapper to call DQN methods 
    
    
    # these are diagnostic / results utilities 
    
    def plot_results(self): 
        """
        Plots the training results.
        
        Parameters: 
            x: the epochs 
            y: the score per epoch 
        """ 
        fig = plt.figure()
        plt.plot(self.results_x,self.results_y) 
        plt.title(model_name[self.file])
        plt.xlabel('Epoch') 
        plt.ylabel('Score')
        plt.show()
        
    
    def write_to_csv(self): 
        """
        Writes the following diagnostics to a csv file: 
            ['Episode','Step','Action', 'Observation', 'Fully Observed State', 'State','Step Reward','Total Reward','Q-values']
            
        Need to have a process for writing the file name. Best idea is simply model_name. Avoids creating too many files. 
        
        """
        import csv 
        
        with open('../Diagnostics/'+model_name[self.file]+'.csv','w+',newline='') as myFile: 
            wr = csv.writer(myFile, quoting=csv.QUOTE_ALL) # note: quoting may change the readout 
            wr.writerow(['Episode','Step','Epsilon','Action', 'Observation', 'Fully Observed State', 'State','Step Reward','Total Reward','Q-values '+str(self.action_list)])
            for row in range(len(self.training_details)): 
                wr.writerow(self.training_details[row]) 
                
                
    def check_if_terminal(self): 
        """
        Checks if the current state is terminal. 
        
        As POMDPX doesn't have a consistent definition for terminal states, need to use a dictionary here. 
        """ 
        done = False
        for i in self.current_state:
            if self.current_state[i] == terminal_states[self.file]:
                done = True 
        return done 
        
         
    
if __name__ == '__main__': 
    sim = simulatorMain()
    print('got here')
    sim.run() 
