#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from collections import defaultdict
import pprint
from numpy import genfromtxt

if len(sys.argv) < 2:
    print "USAGE: compare_perch_cnn.py <stats.txt>"
    exit(1)

results_dir = sys.argv[1]

objects  = [
'002_master_chef_can',
'003_cracker_box',
'004_sugar_box',
'005_tomato_soup_can',
'006_mustard_bottle',
'007_tuna_fish_can',
'008_pudding_box',
'009_gelatin_box',
'010_potted_meat_can',
'011_banana',
'019_pitcher_base',
'021_bleach_cleanser',
'024_bowl',
'025_mug',
'035_power_drill',
'036_wood_block',
'037_scissors',
'040_large_marker',
'051_large_clamp',
'052_extra_large_clamp',
'061_foam_brick',
]

for object_name in objects:
    filename = results_dir + '/' + object_name + '_stats.txt'
    data = genfromtxt(filename, delimiter=' ')
    trans_error = data[0::2,:]
    rot_error = data[1::2,:]
    trans_mean = np.median(trans_error, axis=0)
    rot_mean = np.median(rot_error, axis=0)
    print object_name
    print trans_mean
    print (180 * rot_mean) / (np.pi)
    print '\n'
