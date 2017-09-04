#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:53:57 2017

@author: m300517
"""

from netCDF4 import Dataset

file = "./CloudCoverage_20170602.nc"

nc2 =  Dataset(file)

#ccmask = nc.variables['cloudmask'][:].copy()

#nc.close()
