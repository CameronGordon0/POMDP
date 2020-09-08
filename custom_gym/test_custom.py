#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:12:49 2020

@author: camerongordon

env-reregister error solved by using https://stackoverflow.com/questions/61281918/custom-environments-for-gym-error-cannot-re-register-id 
"""

from pathlib import Path 
import os, inspect, sys

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))

#print(cmd_folder)
path = Path(cmd_folder)
path_src = str(path.parent)+'\src'
#print(path.parent)


if path.parent not in sys.path: 
    sys.path.insert(0,path.parent)
    

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
    
    
if path_src not in sys.path: 
    sys.path.insert(0,path_src)


#for i in sys.path:
#    print(i)





#print(parser_path)

#from parser_path import *


import gym 
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'CustomEnv-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

import envs.custom_env_dir


env = gym.make('CustomEnv-v0')
#print(env.action_space.sample())
obs, reward, done, info = env.step(env.action_space.sample())
print(obs)
#env.step()
#env.reset() 

"""
Note that stable baselines requires that Tensorflow be from versions 1.8.0 to 1.15.0 
To do this use 'pip install tensorflow==1.15.0' 

Also an issue with Python 3.8. use conda install python=3.7.6 
"""

from stable_baselines.common.env_checker import check_env
print("check env")
check_env(env)

#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
from stable_baselines import A2C

model = A2C('CnnPolicy', env).learn(total_timesteps=10)
