# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 13:19:36 2016

@author: JoshuaF
"""

import refractiveSelfCalibration #autocalibration library
import yaml 
import argparse
import inspect


def parseConfig (conf): 
    """
    parse the config parameters
    :param conf: dictionary of config parameters
    :returns:    list of correctly ordered arguments for Calibration
    """
    
    conf_low = {name.lower(): value for name,value in conf.iteritems()}  #case-independently-named config parameters
    values   = []                                                        #list to be filled with ordered arguments for calibration
    args     = inspect.getargspec(refractiveSelfCalibration.Calibration) #names of arguments for calibration
    numopt   = len(args[3])                                              #number of optional arguments
    
    for n,name in enumerate(args[0]):   
        try:
            if name.lower() in ['datapath','exptpath','camids', 'image_type']: #these values should be strings
                values.append(conf_low[name.lower()])
            else:
                values.append(float(conf_low[name.lower()]))  #make sure everything that's supposed to be a number is
        
        except KeyError:
            if n >= len(args[0])-numopt: #optional argument
                continue #if value was not provided ignore it
            else:                        #non-optional argument
                raise KeyError ("Missing " + name + " value in config.")
                
        except ValueError:             
            raise ValueError (name + ' has an invalid value.')
    
    return values
    
parser = argparse.ArgumentParser()
parser.add_argument("configPath", help = "Config file path")

configPath = parser.parse_args().configPath   

with open(configPath , 'r') as f: 
    config = yaml.load(f)
    
refractiveSelfCalibration.Calibration(*parseConfig(config))

