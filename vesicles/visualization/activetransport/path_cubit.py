#!python
#!python
#!python
#!python
#!python
#!python
import time
from math import *

# Do these commands on the CUBIT command line FIRST

# To change the working directory, e.g.: cd "{path_to_my_repo}/STEPS_Validation/vesicles/visualization/activetransport"
# So visuals can be set manually: import abaqus mesh geometry "meshes/sphere_0.5D_2088tets.inp"



ifile_ves=open('data/path_ves.txt', 'r')

ves_pos_data = ifile_ves.readlines()

ntpnts = len(ves_pos_data)

cubit.cmd('color Volume 1 white')

ves_colour1  = ' yellow'

# We start at volume index 2
volindex = 2

ves_radius = '25e-3'

for t in range(ntpnts):
     print (t)
     startvolindex = volindex
     ves1_pos = ves_pos_data[t].split(' ')
     ves1_pos_n = len(ves1_pos)
     i=0
     while(i < ves1_pos_n-1):
          print (i, ves1_pos_n)
          cubit.cmd('create sphere radius ' +ves_radius )
          cubit.cmd('color Volume '+str(volindex)+ ves_colour1)
          posstring = ves1_pos[i]+' '+ves1_pos[i+1]+' '+ves1_pos[i+2]
          cubit.cmd('move volume '+str(volindex)+'  location '+posstring)
          volindex+=1
          i+=3
     cubstring ='hardcopy "'+'images/path_'+str(t)+'.jpg" jpg'
     cubit.cmd(cubstring)
     for vidx in range(startvolindex, volindex): cubit.silent_cmd('delete volume '+str(vidx))














