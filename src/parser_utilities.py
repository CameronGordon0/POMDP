


"""
This file contains the functions for the pomdpx parser class. 

This project extends a python pomdpx parser https://github.com/larics/python-pomdp to handle special characters ('*','-') and special terms ('identity', 'uniform') consistent with the PomdpX File Format as documented at https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation. 

Note that the PomdpX format specified by the Approximate POMDP Planning Toolkit (APPL) allows for both Table (TBL) and Directed Acyclic Graph (DAG) formats for the parameters. Only the TBL format has been implemented for this project.

"""


import numpy as np
import xml.etree.ElementTree as ET
import copy
import random


"""
The below functions get the basic pomdp information (Description, Discount, Actions, Observations, States, and Rewards). 

Returns are dictionaries (Key: variable name, Entry: lists of the applicable variable)

"""

def get_description(root):
    for child in root: 
        if child.tag == 'Description' :
            description = child.text 
    return description 

def get_discount(root):
    for child in root: 
        if child.tag == 'Discount' :
            discount = child.text 
    return discount 

def get_actions(root): 
    A_dict = {} 
    for child in root.findall('Variable'): 
        for var in child: 
            if var.tag == 'ActionVar': 
                vname = var.attrib['vname']
                try: 
                    if var[0].tag == 'ValueEnum': 
                        actions = var[0].text.split(' ')
                    else: 
                        actions = ["a%s" % x for x in range(1, int(var[0].text) + 1)]
                    A_dict[vname]= actions 
                except: 
                    A_dict[vname]= None 
    return A_dict 

def get_observations(root): 
    O_dict = {} 
    for child in root.findall('Variable'): 
        for var in child: 
            if var.tag == 'ObsVar':     
                vname = var.attrib['vname'] 
                try:
                    if var[0].tag == 'ValueEnum': 
                        obs = var[0].text.split(' ')
                    else: 
                        obs = ["o%s" % x for x in range(1, int(var[0].text) + 1)]
                    O_dict[vname] = obs 
                except: 
                    O_dict[vname] = None
    return O_dict

def get_reward_var(root):
    R_dict = {} 
    for child in root.findall('Variable'): 
        for var in child: 
            if var.tag == 'RewardVar':     
                vname = var.attrib['vname'] 
                try:
                    if var[0].tag == 'ValueEnum': 
                        reward = var[0].text.split(' ')
                    else: 
                        reward = ["r%s" % x for x in range(1, int(var[0].text) + 1)]
                    R_dict[vname] = reward
                except:
                     R_dict[vname] = None
                
    return R_dict                

def get_states(root): 
    State_dict = {} 
    name_pairs = {}
    for child in root.findall('Variable'): 
        for var in child: 
            if var.tag == 'StateVar': 
                vnamePrev, vnameCurr = var.attrib['vnamePrev'],var.attrib['vnameCurr']
                name_pairs[vnameCurr] = vnamePrev
                name_pairs[vnamePrev] = vnamePrev
                try: 
                    fullyObs = var.attrib['fullyObs']
                except: 
                    fullyObs = False 
                try:
                    if var[0].tag == 'ValueEnum': 
                        state = var[0].text.split(' ')
                    else: 
                        state = ["s%s" % x for x in range(0, int(var[0].text))]
                    State_dict[vnamePrev] = state, fullyObs
                    State_dict[vnameCurr] = state, fullyObs 
                except:
                     State_dict[vnamePrev] = None, fullyObs
                     State_dict[vnameCurr] = None, fullyObs 
    #print(State_dict)
    return State_dict, name_pairs 







"""
The below functions fill the key data structures for the problem (Initial Belief, Transition Function, Observation Function, and Reward Function). 

Returns are in the form of numpy arrays. 

"""
def get_initial_belief(root): 
    """
    Gets the initial belief for the POMDP. 
    """
    for child in root.findall('InitialStateBelief') :
        
        INITIAL_DICT = {}
        for cond in child.findall('CondProb'): 
            name = 'Null'
            for var in cond.findall('Var'): 
                name = var.text
                varlist = [var.text] # note that the var is restricted to vnameCurr states [i.e. to]
            for parent in cond.findall('Parent'): 
                parentlist = parent.text.split(' ') 

                initial_belief = initialise_matrix(root,varlist,parentlist)
                #print('init',initial_belief)
            
            for param in cond.findall('Parameter'): 
                #print('???',param.text)
                if 'type' in param.attrib:
                    if param.attrib['type'] != 'TBL': 
                        print('Only TBL Parameter Implemented')
                        raise ValueError 
                for entry in param.findall('Entry'): 
                    #print(entry)
                    fill_table(root,varlist,parentlist,entry,initial_belief)
                    
            INITIAL_DICT[name] = initial_belief
            #print(INITIAL_DICT)
                    
    return INITIAL_DICT 


def get_state_transition(root): 
    """
    Gets the State-Transition Matrix T(new state|state,action). 
    Return in the form of a list of numpy arrays of dimensions [action][state][new state], where each entry in the list refers to a different conditional probability in the problem. 

    """

    # get the T(s'|s,a)
    
    for child in root.findall('StateTransitionFunction') :
        
        CONDITION_DICT = {}
        VARIABLE_DICT = {} 
        #CONDITION_TABLE = []
        for cond in child.findall('CondProb'): 
            name = 'NULL'
            for var in cond.findall('Var'): 
                # note that variable should contain only one entry, the 'to''
                name = var.text
                states, pairs = get_states(root)
                name = pairs[name]
                varlist = [var.text] # note that the var is restricted to vnameCurr states [i.e. to]
            for parent in cond.findall('Parent'): 
                # note that the parentlist is the conditional variables (i.e. 'from')
                parentlist = parent.text.split(' ') 
                VARIABLE_DICT[name] = parentlist 
                state_transition_table = initialise_matrix(root,varlist,parentlist)
                #print('ppp',state_transition_table.shape)
            
            for param in cond.findall('Parameter'): 
                if 'type' in param.attrib:
                    if param.attrib['type'] != 'TBL': 
                        print('Only TBL Parameter Implemented')
                        raise ValueError 
                
                for entry in param.findall('Entry'): 
                    fill_table(root,varlist,parentlist,entry,state_transition_table)
            #CONDITION_TABLE.append(state_transition_table)
            CONDITION_DICT[name]=state_transition_table
    #print('???',CONDITION_DICT)
    return CONDITION_DICT, VARIABLE_DICT 

def get_obs_function(root): 
    """
    Gets the Observation Matrix O(observation|new state, action). 
    Returns in the form of a numpy arrays of dimensions [action][new state][observation] where each entry corresponds to a probability. 
    """
    # get the O(o|s',a)
    OBS_DICTIONARY = {}
    for child in root.findall('ObsFunction') :
        for cond in child.findall('CondProb'): 
            name = 'NULL'
            for var in cond.findall('Var'): 
                name = var.text
                varlist = [var.text] # note that the var is restricted to obs [i.e. to]

            for parent in cond.findall('Parent'): 
                parentlist = parent.text.split(' ')                 
                obs_table = initialise_matrix(root,varlist,parentlist)
            for param in cond.findall('Parameter'): 
                if 'type' in param.attrib:
                    if param.attrib['type'] != 'TBL': 
                        print('Only TBL Parameter Implemented')
                        raise ValueError 
                for entry in param.findall('Entry'): 
                    obs_table = fill_table(root,varlist,parentlist,entry,obs_table) 
            OBS_DICTIONARY[name] = obs_table 
    #print(':::',OBS_DICTIONARY)
    return OBS_DICTIONARY


def get_reward_function(root): 
    """
    Gets the Reward Matrix in the form R(action,state). 
    Returns in the form of a numpy array of dimension [action][state]. 
    """
    
    for child in root.findall('RewardFunction'):
        for func in child.findall('Func'): 
            for var in func.findall('Var'): 
                #print(var.text)
                varname = var.text # note that the var is restricted to reward_agent [i.e. to]
            for parent in func.findall('Parent'): 
                parentname = parent.text.split(' ') 
                # note that the parname here is the 'from' ' 
                validlist = get_valid_list(root, parentname)
                dims = get_list_dimensions(validlist)
                reward_table = np.zeros(dims)

            for param in func.findall('Parameter'): 
                if 'type' in param.attrib:
                    if param.attrib['type'] != 'TBL': 
                        print('Only TBL Parameter Implemented')
                        raise ValueError 
                for entry in param.findall('Entry'): 
                    for instance in entry.findall('Instance') :
                        # may need to change this???? 
                        
                        reward_table = get_numpy_reward(root,parentname,entry,instance, reward_table)
    #print('reward_table',reward_table.shape)
    #print(reward_table)
    return reward_table



"""
The below functions are general utilities used for parsing and converting the data structures. 
"""


def initialise_matrix(root,varlist,parentlist):
    """
    Initialises a numpy array for the initial belief, state-transition, observation function, and reward based on the applicable variables. Returns a numpy array of zeros. 

    Parameters
    ----------
    root : Pomdpx xml root
    varlist : List of Variables (Ouptut)
    parentlist : List of Parent Variables (Input)

    Returns
    -------
    Numpy Array
        Returns numpy array of dimensions [(input variables)][output variables]

    """
    
    newlist = parentlist + varlist
    #print(newlist,'000')
    validlist = get_valid_list(root,newlist)
    #print(validlist,'000')
    dims = get_list_dimensions(validlist)
    return np.zeros(dims)

def get_entry_details(entry): 
    """
    Parses an Entry tag to return details for the Instance (e.g. action state0) and the Probability Table associated with the entry. 

    """
    instancelist = entry.findall('Instance')
    probtable = entry.findall('ProbTable')
    instancelist, probtable = instancelist[0].text.split(' '), probtable[0].text.split(' ')
    instancelist = [k for k in instancelist if k !='']
    probtable = [k for k in probtable if k !='']
    return instancelist, probtable 

def fill_table(root,varlist,parentlist,entry,numpy_array):
    """
    Updates a numpy array (e.g. the state-transition matrix) with details for an entry. 
    
    Note that wildcard details (e.g. '*' which applies to all applicable variables or '-' which applies the conditions of a single instance to multiple entries) have been implemented, as have key terms 'uniform' and 'identity' for probability distributions. 
    
    The below method is not elegant. 
    

    Parameters
    ----------
    root : Pomdpx xml root
    varlist : List of Variables (Ouptut)
    parentlist : List of Parent Variables (Input)
    entry : Details of an Entry Tag (Instance and Probability Table)
    numpy_array : Data structure (e.g. state-transition matrix)
    
    Returns
    -------
    numpy_array : Data structure (e.g. state-transition matrix)

    """
    
    # fills details of a single entry 
    newlist = parentlist+varlist
    validlist = get_valid_list(root,newlist)
    instance, prob = get_entry_details(entry)
    
    dimlist = list(get_list_dimensions(validlist))
        
    indices_list= get_indices_for_update(instance,dimlist,validlist)
    
    if len(indices_list) == 1: 
        indices_list = tuple(indices_list[0])
        if type(prob) == list: 
            prob = prob[0]
        numpy_array[indices_list] = prob
    else: 
        for i in indices_list: 
            ind = tuple(i)            
            p = prob[0]
            if p == 'identity': 
                x = np.sqrt(len(indices_list))
                y = list(ind)[-int(x):]
                if all(z == y[0] for z in y):
                    probab = 1
                else: 
                    probab = 0
    
            elif p == 'uniform':
                probab = 1/len(indices_list)
            else:
                probab = prob[indices_list.index(i)]
            if p == '': # non-character not allowed, but appears in pomdpx files 
                probab = 0
            try:
                numpy_array[ind] = probab # generic catch-all for the entry 
            except: 
                numpy_array[ind] = 0
    
    return numpy_array 

def get_indices_for_update(instance_to_parse,dims,validlist):
    """
    Gets the indices [to send to a numpy array] to update based on an instance. 
    
    Handles the special characters '*' (generates a slice) and '-' (uses a recursive function).
    
    Takes the Instance, Dimensions of each of the Instance components, and the list of valid variables. 

    """
    start = 0 
    results = []
    running_output = []
    maxlen = len(instance_to_parse)
    #print(validlist)
    for i in range(0,len(instance_to_parse)):
        #print(i)
        if instance_to_parse[i] == '*':
            instance_to_parse[i] = slice(0,dims[i])
        if instance_to_parse[i] in validlist[i]: 
            instance_to_parse[i] = validlist[i].index(instance_to_parse[i])
            
    if '-' in instance_to_parse:
        get_indices(start,instance_to_parse,running_output,maxlen,dims,results)
    else: 
        results = [instance_to_parse]
    return results

def get_indices(start,test,output,maxlen,dims,results): 
    """
    Handles the indices for the get_indices_for_update function, including the recursive logic required.
    
    Indices are stored in the results list. 

    """

    for i in range(start,len(test)): 
        if type(test[i]) == int: 
            output.append(test[i])
            if len(output) == maxlen: 
                pass
            
        elif test[i] == '*': 
            output.append(slice(0,dims[i]))
            if len(output) == maxlen: 
                pass
        elif test[i] == '-':
            B = range(0,dims[i])
            for j in B: 
                newtest = [k for k in test]
                newtest[i]=j
                if '-' not in newtest:
                    results.append(newtest)
                get_indices(i,newtest,output,maxlen,dims,results)


def get_numpy_reward(root,parent,entry,instance, numpy_array):
    
    """
    Fills the numpy entries for the reward function. Note that can possibly be alternately implemented with the fill_table method. 
    """

    validlist = get_valid_list(root,parent)
    value = get_entry_value(entry)
    instance = instance.text.split(' ')
    index_list = [] 
    
    for i in range(len(instance)): 

        window = validlist[i]
        if instance[i] == '*': # wildcard means it applies to anything that could be here
            index = slice(0,len(window)) # using a slice here is important 
        else: 
            index = window.index(instance[i])

        index_list.append(index)

    index_list = tuple(index_list)
    numpy_array[index_list] = value
    
    return numpy_array

def get_entry_value(entry): 
    for value in entry.findall('ValueTable'): 
        return value.text


def try_all_variables(root,key): 
    """
    Tries a variable within the variable dictionaries (e.g. finding the relevant action)

    """
    action_dict = get_actions(root)
    state_dict, pairs = get_states(root) 
    obs_dict = get_observations(root)
    
    if key in action_dict: 
        valid = action_dict[key]
    elif key in state_dict: 
        key = pairs[key]
        #print(key)
        valid = state_dict[key][0]
    elif key in obs_dict:
        valid = obs_dict[key]
    else: 
        valid = None 
        
    return valid 

def get_valid_list(root,test_list): 
    """
    Gets the valid variable entries within an instance list. 
    """
    
    validlist = []
    for i in test_list: 
        valid = try_all_variables(root,i)
        if valid != None:
            validlist.append(valid)
    return validlist

def get_list_dimensions(A): 
    """
    Gets the dimensions of a list
    """
    lenlist = [] 
    for i in A: 
        if i != None:
            lenlist.append(len(i)) 
    return tuple(lenlist)
