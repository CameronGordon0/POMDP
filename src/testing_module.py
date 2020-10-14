#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:17:02 2020

@author: camerongordon


This file automates the testing for multiple parameters 


Note the main function from simulator_main 

def main(file = '../examples/rockSample-3_1.pomdpx', 
         control = 'DQN', 
         training_period = 30,
         testing_period = 1,
         verbose = False,
         history = False,
         history_len = 4,
         maxsteps=50,
         include_actions=True,
         recurrent=False): 
    simulator = Simulator(file)
    simulator.print_model_information()
    
    x,y,details = control_method(simulator,control,training_period,
                                 verbose=verbose,history=history,history_len=history_len,
                                 maxsteps=maxsteps,include_actions=include_actions,
                                 recurrent=recurrent)
    
    plot_results(x,y,model_name[file])
"""


#from simulator_main import *  

from datetime import date


from simulator_main_class_refactor import * 

from datetime import datetime 

from pomdp_simulator import Simulator 

import matplotlib.pyplot as plt 

import numpy as np 

from numpy import convolve 


model_name = {'../examples/Tiger.pomdpx':'Tiger',
              '../examples/rockSample-3_1.pomdpx':'Rock Sample (3,1)',
              '../examples/rockSample-7_8.pomdpx':'Rock Sample (7,8)',
              '../examples/rockSample-10_10.pomdpx':'Rock Sample (10,10)',
              '../examples/rockSample-11_11.pomdpx':'Rock Sample (11,11)',
              '../examples/Tag.pomdpx':'Tag',
              '../examples/auvNavigation.pomdpx': 'AUV',
              '../examples/functional_imitation.pomdpx':'functional_imitation'} 


despot_scores = {'../examples/Tiger.pomdpx':[13.45,8.71], #un undiscounted, discounted
              '../examples/rockSample-3_1.pomdpx':[15.4,13.2],
              '../examples/rockSample-7_8.pomdpx':[40.6,20.06],
              '../examples/Tag.pomdpx':[-9.27,-7.05],
              '../examples/auvNavigation.pomdpx': 'AUV'} 



def get_file(val): 
    for file, name in model_name.items(): 
         if val == name: 
             return file 
  
    return "file doesn't exist"

testing_models = list(model_name.values())
bool_list = [True, False] 

def get_model_type_and_format(control='DQN',
                   recurrency=False,
                   include_actions=False,
                   prioritised_experience_replay=False,
                   history=False,
                   include_reward = False): 
    """
    returns tuple of the name [0]
    and pyplot colour [1]
    and linestyle [2] associated with a model 
    
    """
    
    if control == 'DQN': 
        if not recurrency: 
            if not include_actions: 
                if not prioritised_experience_replay: 
                    if not history: 
                        return 'DQN', 'tab:orange', 'solid','s'
                    else:
                        return 'DQN', 'tab:blue', 'solid','s'
                else:
                    if not history:
                        return 'DQN (PER)', 'tab:orange' , 'solid','s'
                    else:
                        return 'DQN (PER)', 'tab:blue' , 'solid','s'
            else: 
                if not prioritised_experience_replay: 
                    if not include_reward:
                        return 'ADQN' , 'tab:red' , 'solid','v'
                    else: 
                        return 'RADQN', 'tab:olive', 'solid'
                else:
                    if not include_reward:
                        return 'ADQN (PER)' , 'tab:red' , 'solid','v'
                    else:
                        return 'RADQN (PER)', 'tab:olive', 'solid'
        else: 
            if not include_actions: 
                if not prioritised_experience_replay: 
                    return 'DRQN' , 'tab:green' , 'solid','o'
                else:
                    return 'DRQN (PER)' , 'tab:green' , 'solid','^'
            else: 
                if not prioritised_experience_replay: 
                    if not include_reward:
                        return 'ADRQN' , 'tab:purple' , 'solid','x'
                    else: 
                        return 'RADRQN', 'tab:pink', 'solid'
                else:
                    if not include_reward:
                        return 'ADRQN (PER)' , 'tab:purple' , 'solid','x'
                    else:
                        return 'RADRQN (PER)', 'tab:pink', 'solid'
    if control == 'Random': 
        return 'Random' , 'k' , 'solid','<'
    else: 
        raise NameError('Unspecified Model')
            

for recurrency in bool_list: 
    for include_actions in bool_list: 
        for prioritised_experience_replay in bool_list: 
            #for history in bool_list:
            print(get_model_type_and_format(recurrency=recurrency,
                                     include_actions = include_actions,
                                     prioritised_experience_replay=prioritised_experience_replay)[2])
            
            
def tester(model,file,
           training_period, 
           evaluation_period, 
           maxsteps,
           history_len,
           recurrency,include_actions,prioritised_experience_replay,include_reward, results,expert): 
    start = datetime.now() 
    sim = simulatorMain(file=file,
                        training_period=training_period,
                        evaluation_period=training_period,
                        maxsteps=maxsteps,
                        history_len = history_len,
                        include_actions=include_actions,
                        recurrent=recurrency,
                        priority_replay=prioritised_experience_replay,
                        include_reward=include_reward)
    
    sim.run(expert)
    x = sim.training_results_x
    y = sim.training_results_y
    final_result = round(sim.final_result,2)
                        
    history=True

    end = datetime.now() 
    runtime = end - start 
    details = get_model_type_and_format(recurrency=recurrency,
                         include_actions = include_actions,
                         prioritised_experience_replay=prioritised_experience_replay, 
                         history=history,include_reward=include_reward)

    if include_reward: 
        print(details)
    
    results[model,
            history,
            history_len,
            'DQN',
            recurrency,
            maxsteps,
            include_actions,
            prioritised_experience_replay,
            include_reward] = x, y, runtime, details, final_result
    return results
    


def run_tests(prioritised=False,moving_average=True,av_len=8,save_fig=True,expert=False): 
    
    
    model = 'Tag'
    training_period = 150
    evaluation_period=150
    history = True 
    history_len = 5
    control = 'DQN' 
    #recurrency = False 
    maxsteps = 30  
    #include_actions = False
    #prioritised = True 
    
    
    file = get_file(model) 
    simulator = Simulator(file) 
    
    despotScoreDiscounted = [despot_scores[file][1] for i in range(training_period)]
    despotScoreUndiscounted = [despot_scores[file][0] for i in range(training_period)]
    
    
    
    drqn_seen = False
    dqn_seen = False
    
    results = {} 
    for recurrency in bool_list: 
        for include_actions in bool_list: 
            for prioritised_experience_replay in [prioritised]: 
                results = tester(model,file,
                                 training_period=training_period, 
                                 evaluation_period=evaluation_period, 
                                 maxsteps=maxsteps,
                                 history_len=history_len,
                                 recurrency=recurrency,include_actions=include_actions,
                                 prioritised_experience_replay=prioritised_experience_replay,
                                 results=results,include_reward=False,expert=expert)

        results = tester(model,file,
                        training_period=training_period, 
                        evaluation_period=evaluation_period, 
                        maxsteps=maxsteps,
                        history_len=history_len,
                        recurrency=recurrency,
                        include_actions=True,
                        prioritised_experience_replay=prioritised,
                        include_reward=True,results=results,expert=expert)
                
    for result in list(results.values()):
        x = result[0] 
        y = result[1] 
        
        
        details = result[3]
        
        if moving_average: # 5 it moving aveage 
            moving_av_y = movingaverage(y,av_len)
            plt.plot(x[len(x)-len(moving_av_y):],moving_av_y,label=details[0]+': '+str(result[4]),color=details[1],linestyle=details[2]) 
            #plt.scatter(x[::3],y[::3],color=details[1],label=details[0], marker=details[3],alpha=0.5)
            
        else: 
        
            plt.plot(x,y,label=details[0],color=details[1],linestyle=details[2]) 
    
        print(details[0],'runtime',result[2])
    
    plt.plot(x,despotScoreDiscounted, label='Despot (Discounted): '+str(despotScoreDiscounted[0]), color = 'k')
    plt.plot(x,despotScoreUndiscounted, label='Despot (Undiscounted): '+str(despotScoreUndiscounted[0]), color = 'k', linestyle='dashed')

    
    plt.xlabel('iteration') 
    plt.ylabel('score') 
    plt.title(model)
    plt.legend() 
    if save_fig == True: 
        if moving_average:
            if prioritised: 
            
                plt.savefig('../Results/'+model+'hist'+str(history_len)+'av'+str(av_len)+'train'+str(training_period)+'expert'+str(expert)+str(date.today())+'PER.png', bbox_inches='tight')
            else: 
                plt.savefig('../Results/'+model+'hist'+str(history_len)+'av'+str(av_len)+'train'+str(training_period)+'expert'+str(expert)+str(date.today())+'.png', bbox_inches='tight')
        else:
            if prioritised: 
                plt.savefig('../Results/'+model+'hist'+str(history_len)+'no_av'+'train'+str(training_period)+'expert'+str(expert)+str(date.today())+'PER.png', bbox_inches='tight')
            else: 
                plt.savefig('../Results/'+model+'hist'+str(history_len)+'no_av'+'train'+str(training_period)+'expert'+str(expert)+str(date.today())+'.png', bbox_inches='tight')
    #plt.show()
    plt.close()


                


    
    
    


def write_tests_to_excel(): 
    pass 


def movingaverage(values, window):
    #from https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/ 
    
    # note: when plotting, need to drop a few data points lost during convolution 
    
    # so plot(x[len(x)-len(moving_av_y):],moving_av_y)
    
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
 
#x = [1,2,3,4,5,6,7,8,9,10]
#y = [3,5,2,4,9,1,7,5,9,1]
 
#yMA = movingaverage(y,3)
for expert in [False]:
    for prioritised in [False,True]: 
        run_tests(prioritised=prioritised,expert=expert) 
