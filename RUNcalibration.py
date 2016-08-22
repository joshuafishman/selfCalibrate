# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 13:19:36 2016

@author: JoshuaF
"""

import refractiveSelfCalibration #autocalibration library
import yaml
import sys

if len(sys.argv) > 1:
    configPath = sys.argv[1]
else:    
    configPath = raw_input('Input the config file please: ')
    configPath = 'config.txt' if configPath == '' else configPath #hit enter to just use local 'config.txt' as the config file

#config file must have exactly correct variable names
config = yaml.load(open(configPath , 'r'))

for name,var in config.iteritems():
    if name not in ['dataPath','exptPath','camIDs']: #these values should be strings
        try:
            config[name] = float(var) #make sure everything that's supposed to be a number is
        except:
            raise Exception (name + ' has an invalid value.')
              
globals().update(config) #add all the variables in the config file to the global namespace

refractiveSelfCalibration.CalibrationTiff(dX,dY,nX,nY,nCalPlanes,znet, sX,sY,pix_Pitch,so,f,nFrames, n1,n2,n3,tW,zW, tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol, dataPath, exptPath, camIDs)

