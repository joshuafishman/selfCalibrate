# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 13:19:36 2016

@author: JoshuaF
"""

import refractiveSelfCalibration #autocalibration library
import yaml
import sys
import os.path  

path       = sys.argv[1] if len(sys.argv)>1 else '' #current input
configPath = ''                                     #full path

while not configPath or not os.path.isfile(configPath):
                           
    if os.path.exists(os.path.join(configPath, path)) or os.path.exists(configPath): #add path to configPath 
        if os.path.exists(os.path.join(configPath, path)): 
            configPath = os.path.join(configPath, path)    
        else:
            print ('\n' + os.path.join(configPath, path) + ' is an invalid path.')     
        
        if not os.path.isfile(configPath):           
            path = raw_input('Input a file or folder name within ' + configPath + ', "cd" to go up a level or "exit" to quit.\n')       
        else:
            print ('\nFound file at:\n' +configPath)
    else:
        path = raw_input('Input the config file or a folder containing it please:\n')
        path = 'config.txt' if path == '' else path #hit enter to use local config.txt
    
    
    if path == 'exit':  #allows quitting in Spyder
        sys.exit()     
    
    if path == 'cd':    #back up a level
        configPath = os.path.split(configPath)[0] 
        path = ''
       
    if len(path)>0 and path[-1] == '\t': #tab-to-complete
        name = path[:-1]
        path = '' 
        
        if '\\' in name or '/' in name:
            print ('Please search for only one level at a time.')
            
        if len (configPath) > 0:
            places = [place for place in os.listdir(configPath) if name in place]
        else:
            print ('Please enter a directory before searching.')
            continue
        
        if len(places) == 1:
            path = str(*places) 
            print path
        elif len(places) == 0:
            print ('No locations found in ' +configPath + ' containing ' +name)      
        else:
            print ('Multiple locations containing "' + name + '"' + ' in ' +configPath + ' :\n' + str(places))
    
    
#config file must have exactly correct variable names
with open(configPath , 'r') as f: 
    config = yaml.load(f)

for name,var in config.iteritems():
    if name not in ['dataPath','exptPath','camIDs']: #these values should be strings
        try:
            config[name] = float(var) #make sure everything that's supposed to be a number is
        except:
            raise ValueError (name + ' has an invalid value.')
              
globals().update(config) #add all the variables in the config file to the global namespace

refractiveSelfCalibration.CalibrationTiff(dX,dY,nX,nY,nCalPlanes,znet, sX,sY,pix_Pitch,so,f,nFrames, n1,n2,n3,tW,zW, tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol, dataPath, exptPath, camIDs)

