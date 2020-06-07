#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the Pomdpx Parser class. 

This project extends a python pomdpx parser https://github.com/larics/python-pomdp to handle special characters ('*','-') and special terms ('identity', 'uniform') consistent with the PomdpX File Format as documented at https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation. 


"""

from parser_utilities import * 



class Parser(): 
    def __init__(self,model_filename):
        root = ET.parse(model_filename).getroot()
        self.description = get_description(root)
        self.discount = get_discount(root)
        self.actions = get_actions(root) 
        self.states = get_states(root)
        self.observations = get_observations(root)
        self.initial_belief = get_initial_belief(root)
        self.reward_table = get_reward_function(root) 
        self.obs_table = get_obs_function(root)
        self.state_transition, self.state_variable_dict = get_state_transition(root)
    


def main(): 
    print('main') 
    #model_filename = '../examples/functional_imitation.pomdpx'
    model_filename = '../examples/Tiger.pomdpx'
    #model_filename = '../examples/rockSample-3_1.pomdpx'
    #model_filename = '../examples/rockSample-7_8.pomdpx'
    
    root = ET.parse(model_filename).getroot()
    print(get_description(root))
    print(get_discount(root))
    print(get_actions(root))
    print(get_observations(root))
    print(get_reward_var(root))
    print(get_states(root))
    print('initial_belief',get_initial_belief(root)) 
    print('transition_table',get_state_transition(root)) 
    print('obs_table',get_obs_function(root)) 
    print('reward_table',get_reward_function(root))



if __name__ == "__main__": 
    main() 
