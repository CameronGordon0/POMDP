# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:05:55 2020

@author: MrCameronGordon
"""
import re

def parser(): 
    write = []
    with open('rock_sample_despot.txt') as f: 
        lines = f.readlines() 
        columns = [] 
        
        i = 1 
        Round = None 
        Step = None 
        Observation = None
        Action = None
        Reward = None
        Full_line=False
        Observed_state = None
        
        
        for line in lines: 
            line = line.strip() # removes learding/ trailing white space 
            if 'Round' and 'Step' in line: 
                Full_line = False
                line = line.replace('-','')
                line = line.split()
                Round = line[1]
                Step = line[3]
                print(Round,Step)
            if 'Observation' in line: 
                line = line.replace('[','{')
                line = line.replace(']','}')
                line = line.split()
                Observation = line[3]
                print(Observation)
            if 'robot_1' in line: 
                line = line.replace('[','{')
                line = line.replace(',','}')
                #line = re.split(':,',line)
                line = line.split(' ')
                Observed_state = line[0]
                print(Observed_state)
                
                
                #print(line[0])
            if 'tagged' in line: 
                Reward = 10
                print(Reward)
                #for x in line: 
                #    print(x)
            if 'Action' in line: 
                line = line.split(':')
                Action = line[1]
                
                print(Action)
            if 'Reward' in line: 
                line = line.split('=')
                Reward = line[1]
                print(Reward)
                Full_line = True
                
            if 'Simulation terminated' in line: 
                Full_line = True
            if Full_line:
                write.append([Round, Observation,Observed_state, Action, Reward])
                Full_line = False
    
    with open('rock_sample_7_8.csv','a+',newline='') as file: 
        import csv 
        wr = csv.writer(file,quoting=csv.QUOTE_ALL)
        for row in write: 
            wr.writerow(row)
        
                
                
parser()