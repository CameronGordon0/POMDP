#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:12:49 2020

@author: camerongordon
"""



import gym 
import custom_gym.envs.custom_env_dir


env = gym.make('CustomEnv-v0')
env.step()
env.reset() 