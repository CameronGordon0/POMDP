"""
POMDP Parser for the pomdpx format. 

Based on https://github.com/larics/python-pomdp 
""" 

import numpy as np 
import xml.etree.ElementTree as ET 
import copy 


class POMDP: 
    
    def __init__(self, model_filename): 
        root_model = ET.parse(model_filename).getroot() 
        self.description, self.discount, self.states, self.actions, self.observations,self.reward = get_general_info(root_model) 



def get_general_info(root) :
    for child in root: 
        if child.tag == 'Description': 
            description = child.text 
        elif child.tag == 'Discount': 
            discount = float(child.text) 
    for child in root.findall('Variable'): 
        states = {} 
        actions = {} 
        observations = {} 
        reward = {} 
        for k in child: 
            attributes = k.attrib
            #print(attributes)
            
            if k.tag == 'StateVar': 
                if k[0].tag == 'ValueEnum': 
                    enum = k[0].text.split(' ') 
                states[attributes['vnamePrev']]=attributes,enum
            if k.tag =='ActionVar': 
                if k[0].tag == 'ValueEnum': 
                    enum = k[0].text.split(' ') 
                actions[attributes['vname']]=attributes,enum
            if k.tag =='ObsVar': 
                if k[0].tag == 'ValueEnum': 
                    enum = k[0].text.split(' ') 
                observations[attributes['vname']]=attributes,enum
            if k.tag =='RewardVar': 
                #print(k.tag)
                #if k[0].tag == 'ValueEnum': 
                 #   enum = k[0].text.split(' ') 
                reward[attributes['vname']]=attributes
            #print('next') 
            """
            if k.tag == 'StateVar': 
                vname_prev = k.attrib['vnamePrev'] 
                vname_current = k.attrib['vnameCurr'] 
                fullyObs = k.attrib['fullyObs']
                
                if k[0].tag == 'ValueEnum': 
                    states[vname_prev] = []
                    
                print('next')
        """
    return description, discount, states, actions, observations, reward  



# Testing 

if __name__ == '__main__': 
    pomdp = POMDP('../Testing Files/Tiger.pomdpx') 
    
    print(pomdp.reward)