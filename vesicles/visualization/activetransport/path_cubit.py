#!python
#!python
#!python
#!python
#!python
#!python
import time
from math import *
# Do this on the CUBIT command line FIRST so visuals can be set manually: import abaqus mesh geometry "/Users/iain/GIT/STEPS_Validation/vesicles/visualization/activetransport/meshes/sphere_0.5D_2088tets.inp"



ifile_ves=open('/Users/iain/GIT/STEPS_Validation/vesicles/visualization/activetransport/data/path_ves.txt', 'r')

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
     cubstring ='hardcopy "'+'/Users/iain/GIT/STEPS_Validation/vesicles/visualization/activetransport/images/path_'+str(t)+'.jpg" jpg'
     cubit.cmd(cubstring)
     for vidx in range(startvolindex, volindex): cubit.silent_cmd('delete volume '+str(vidx))














