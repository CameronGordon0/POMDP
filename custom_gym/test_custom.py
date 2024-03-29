#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:12:49 2020

@author: camerongordon

env-reregister error solved by using https://stackoverflow.com/questions/61281918/custom-environments-for-gym-error-cannot-re-register-id 

This file contains code for creating a custom gym environment and populating it with information specified in POMDPX format. 

Note: issues remain integrating this with OpenAI Baselines. 
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


#from parser_path import *


import gym 
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'CustomEnv-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

import envs.custom_env_dir


env = gym.make('CustomEnv-v0')
obs, reward, done, info = env.step(env.action_space.sample())
print(obs)


"""
Note that stable baselines requires that Tensorflow be from versions 1.8.0 to 1.15.0 
To do this use 'pip install tensorflow==1.15.0' 

Also an issue with Python 3.8. use conda install python=3.7.6 

use pip install git+https://github.com/hill-a/stable-baselines

note: directly import the class from the stable_baselines. file path on github

avoid ddpg as MPI introduces a lot of issues 
"""

from stable_baselines.common.env_checker import check_env
print("check env")
check_env(env)

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq import DQN

model = DQN(MlpPolicy, env).learn(total_timesteps=10)
