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
    
    conf_low = {name.lower(): value for name,value in conf.iteritems()} #config file must have exactly correct variable names, ignoring case
    
    args = inspect.getargspec(refractiveSelfCalibration.Calibration) #arguments for calibration
    values = []
    
    try:
        
        for name in args[0][:-len(args[3])]:   #non-optional values
            if name.lower() in ['datapath','exptpath','camids', 'image_type']: #these values should be strings
                values.append(conf_low[name])
            else:
                values.append(float(conf_low[name]))  #make sure everything that's supposed to be a number is
        
        for name in args[0][-len(args[3]):]:   #optional values
            try:
                if name.lower() in ['datapath','exptpath','camids', 'image_type']: #these values should be strings
                    values.append(conf_low[name.lower()])
                else:
                    values.append(float(conf_low[name.lower()]))  #make sure everything that's supposed to be a number is
            except KeyError:
                continue
        
    except KeyError:
        raise KeyError ("Missing " + name + " value in config file.")
    except ValueError:             
        raise ValueError (name + ' has an invalid value.')
    
    return values
    
parser = argparse.ArgumentParser()
parser.add_argument("configPath", help = "Config file path")

configPath = parser.parse_args().configPath   

with open(configPath , 'r') as f: 
    config = yaml.load(f)
    
refractiveSelfCalibration.Calibration(*parseConfig(config))

