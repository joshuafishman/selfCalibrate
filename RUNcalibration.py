# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 13:19:36 2016

@author: JoshuaF
"""

import refractiveSelfCalibration #autocalibration library
import yaml 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("configPath", help = "Config file path")

configPath = parser.parse_args().configPath   

#config file must have exactly correct variable names
with open(configPath , 'r') as f: 
    config = yaml.load(f)
    
#these are the expected variables in config. Order matters.
if config['image_type'] == 'multi':            
    argnames = ["dataPath","exptPath","camIDs","image_type","dX","dY","nX","nY","nCalPlanes","znet","n1","n2","n3","tW","zW","tol","fg_tol","maxiter","bi_tol","bi_maxiter","z3_tol","rep_err_tol","sX","sY","pix_Pitch","so","f","nFrames"]
else:
    argnames = ["dataPath","exptPath","camIDs","image_type","dX","dY","nX","nY","nCalPlanes","znet","n1","n2","n3","tW","zW","tol","fg_tol","maxiter","bi_tol","bi_maxiter","z3_tol","rep_err_tol","sX","sY","pix_Pitch","so","f"]

values = []
for name in argnames:   
    try:
        if name in ['dataPath','exptPath','camIDs', 'image_type']: #these values should be strings
            values.append(config[name])
        else:
            values.append(float(config[name]))  #make sure everything that's supposed to be a number is
    except KeyError:
        raise KeyError ("Missing " + name + " value in config file.")
    except ValueError:             
        raise ValueError (name + ' has an invalid value.')

refractiveSelfCalibration.Calibration(*values)

