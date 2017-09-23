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

perch_dir = sys.argv[1]
baseline_dir = sys.argv[2]

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

# TRANS_THRESH = 0.01 # m
# ROT_THRESH = 0.174533 * 100  # rad (10 deg)
TRANS_THRESH = 1000.0 # m
ROT_THRESH = 0.01  # rad (10 deg)

np.set_printoptions(suppress=True,precision=3)

all_scenes = set()
all_valid_scenes = set()

trans = np.zeros(21)
rots = np.zeros(21)
correct = np.zeros(21)

base_trans = np.zeros(21)
base_rots = np.zeros(21)
base_correct = np.zeros(21)

ii = 0

for object_name in objects:
    filename = perch_dir + '/' + object_name + '_stats.txt'
    base_filename = baseline_dir + '/' + object_name + '_stats.txt'
    data = genfromtxt(filename, delimiter=' ')
    base_data = genfromtxt(base_filename, delimiter=' ')

    trans_error = data[1::3]
    rot_error = data[2::3]

    base_trans_error = base_data[1::3]
    base_rot_error = base_data[2::3]

    scenes = data[0::3] 
    all_scenes = all_scenes.union(set(scenes))

    valid_idxs = trans_error != 1000
    # print valid_idxs
    trans_error = trans_error[valid_idxs]
    rot_error = rot_error[valid_idxs]
    scenes = scenes[valid_idxs]

    base_trans_error = base_trans_error[valid_idxs]
    base_rot_error = base_rot_error[valid_idxs]

    is_correct = (trans_error < TRANS_THRESH) & (rot_error < ROT_THRESH)
    base_is_correct = (base_trans_error < TRANS_THRESH) & (base_rot_error < ROT_THRESH)

    all_valid_scenes = all_valid_scenes.union(set(scenes))

    trans_median = np.median(trans_error, axis=0)
    rot_median = np.median(rot_error, axis=0)
    trans_mean = np.mean(trans_error, axis=0)
    rot_mean = np.mean(rot_error, axis=0)

    base_trans_median = np.median(base_trans_error, axis=0)
    base_rot_median = np.median(base_rot_error, axis=0)
    base_trans_mean = np.mean(base_trans_error, axis=0)
    base_rot_mean = np.mean(base_rot_error, axis=0)

    print object_name
    # print 'Mean:'
    # print trans_mean
    # print (180 * rot_mean) / (np.pi)
    # print 'Median'

    trans[ii] = trans_mean
    rots[ii] = (180 * rot_mean) / (np.pi)
    correct[ii] = sum(is_correct) * 100.0 / len(is_correct)

    base_trans[ii] = base_trans_mean
    base_rots[ii] = (180 * base_rot_mean) / (np.pi)
    base_correct[ii] = sum(base_is_correct) * 100.0 / len(base_is_correct)

    ii = ii + 1

    # print trans_median
    # print (180 * rot_median) / (np.pi)
    # print '\n'
a = len(all_scenes)
b = len(all_valid_scenes)
print a,b
print b*100.0/a
print correct
print base_correct
np.savetxt('trans.txt', 100 * trans, '%5.2f')
np.savetxt('rots.txt', rots, '%5.2f')
np.savetxt('perch_accuracy.txt', correct, '%5.2f')
np.savetxt('baseline_accuracy.txt', base_correct, '%5.2f')
