import gym 

from src.pomdp_simulator import Simulator



class CustomEnv(gym.Env): 

	def __init__(self): 
		print('Environment initialised!') 
	def step(self): 
		print('Step successful!') 
	def reset(self): 
		print('Environment reset') 

