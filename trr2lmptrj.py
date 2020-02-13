#!/usr/bin/env python2
# envirment: cgmap(Anaconda2/4.3.0)
# Run on midway2 

import sys
sys.path.append('/project/gavoth/congwang/cgmap/src/')
sys.path.append('/project/gavoth/congwang/mdtraj/mdtraj/')
import cgmap as cg
import mdtraj as md
import md_check as check

############################### config #####################################

input_traj = 'traj_center.trr'
input_top  = 'traj_center.gro'

output_traj = 'traj_center.lammpstrj'

output_dir = './input/'
input_dir  = './input/'

############################### run ########################################

trj = md.load(input_dir + input_traj, top=input_dir + input_top)

trj.save_lammpstrj(output_dir + output_traj)
