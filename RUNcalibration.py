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

#config file must have exactly correct variable names
config = yaml.load(open(configPath , 'r'))

for name,var in config.iteritems():
    if name not in ['dataPath','exptPath','camIDs']: #these values should be strings
        try:
            var = str(format(float(var),'.16f')) #make sure everything that's supposed to be a number is
        except:
            raise Exception (name + ' has an invalid value.')
        
                
globals().update(config) #add all the variables in the config file to the local namespace

refractiveSelfCalibration.Calibration(dX,dY,nX,nY,nCalPlanes,znet, sX,sY,pix_Pitch,so,f,nFrames, n1,n2,n3,tW,zW, tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol, dataPath, exptPath, camIDs)

