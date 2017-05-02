# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# STEPS - STochastic Engine for Pathway Simulation
# Copyright (C) 2007-2013 Okinawa Institute of Science and Technology, Japan.
# Copyright (C) 2003-2006 University of Antwerp, Belgium.
#
# See the file AUTHORS for details.
#
# This file is part of STEPS.
#
# STEPS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# STEPS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import steps.mpi
import steps.model as smodel
import steps.utilities.geom_decompose as gd
import steps.mpi.solver as solvmod

import datetime
import steps.model as smodel
import numpy as np
import steps.geom as stetmesh
import steps.rng as srng
import steps.utilities.meshio as meshio
import time

from tol_funcs import *

rng = srng.create('r123', 512) 
rng.initialize(steps.mpi.rank + 1000)

NITER = 10
DT = 0.01
INT = 0.05

CONC =  50.0e-6	# the number of initial molecules

DCST = 0.1e-9

SAMPLE = 12033	

MESHFILE = 'cyl_diam0_4__len10_12Ktets'

# <1% fail with a tolerance of 5%
tolerance = 5.0/100


# create the array of tet indices to be found at random
tetidxs = np.zeros(SAMPLE, dtype = 'int')
# further create the array of tet barycentre distance to centre
tetrads = np.zeros(SAMPLE)

beg_time = time.time()

########################################################################
####                     ERROR FUNCTION STUFF                       ####
########################################################################

# Arguments are the x value to find, and the number of bins to iterate over
def erfunc(x, num = 1000):
	erf = 0.0
	
	place = 0.0
	ds= x/num
	for i in range(num):
		nowx = (i*x)/num 
		nextx = ((i+1)*x)/num
		goodx = (nowx+nextx)/2.0
		erf+=(ds*np.exp(-(goodx*goodx)))
	
	return 1 -(2*(erf/np.sqrt(np.pi)))


def getConc(Cs, D, x, t):
	return (Cs*erfunc(x/(np.sqrt(4*D*t))))

########################################################################

def gen_model():
    
	mdl = smodel.Model()
	X = smodel.Spec('X', mdl)
	cytosolv = smodel.Volsys('cytosolv', mdl)
	dif_X = smodel.Diff('diffX', cytosolv, X, 0)
	
	return mdl

########################################################################

def gen_geom():
    mesh = meshio.loadMesh('meshes/' +MESHFILE)[0]
    
    ntets = mesh.countTets()
    
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    comp.addVolsys('cytosolv')
    
    # Now fill the array holding the tet indices to sample at random
    assert(SAMPLE == ntets)
    
    numfilled = 0
    while (numfilled < SAMPLE):
        tetidxs[numfilled] = numfilled
        numfilled +=1
    
    # Now find the distance of the centre of the tets to the Z lower face
    for i in range(SAMPLE):
        baryc = mesh.getTetBarycenter(int(tetidxs[i]))
        min = mesh.getBoundMin()
        r = baryc[2] - min[2]
        # Convert to microns
        tetrads[i] = r*1.0e6
    
    return mesh

########################################################################

m = gen_model()
g = gen_geom()

# And fetch the total number of tets to make the data structures
ntets = g.countTets()

tet_hosts = gd.binTetsByAxis(g, steps.mpi.nhosts)
if steps.mpi.rank ==0:
    gd.printPartitionStat(tet_hosts)

sim = solvmod.TetOpSplit(m, g, rng, False, tet_hosts)

tpnts = np.arange(0.0, INT, DT)
ntpnts = tpnts.shape[0]


#Create the big old data structure: iterations x time points x concentrations
if steps.mpi.rank ==0:
    res = np.zeros((NITER, ntpnts, SAMPLE))

# Find the tets connected to the bottom face
# First find all the tets with ONE face on a boundary
boundtets = []

# store the 0to3 index of the surface triangle for each of these boundary tets
bt_srftriidx = []

for i in range(ntets):
	tettemp = g.getTetTetNeighb(i)
	if (tettemp[0] ==-1 or tettemp[1] == -1 or tettemp[2] == -1 or tettemp[3] == -1): 
		boundtets.append(i)
		templist = []
		if (tettemp[0] == -1): 
			templist.append(0)
		if (tettemp[1] == -1): 
			templist.append(1)
		if (tettemp[2] == -1): 
			templist.append(2)
		if (tettemp[3] == -1): 
			templist.append(3)
		bt_srftriidx.append(templist)

assert (boundtets.__len__() == bt_srftriidx.__len__())

minztets = []
boundminz = g.getBoundMin()[2] + 0.01e-06
num2s=0
for i in range(boundtets.__len__()):
	# get the boundary triangle
	if (bt_srftriidx[i].__len__() == 2): num2s+=1
	for btriidx in bt_srftriidx[i]:
		zminboundtri = True
		tribidx = g.getTetTriNeighb(boundtets[i])[btriidx]
		tritemp = g.getTri(tribidx)
		trizs = [0.0, 0.0, 0.0]
		trizs[0] = g.getVertex(tritemp[0])[2]
		trizs[1] = g.getVertex(tritemp[1])[2]
		trizs[2] = g.getVertex(tritemp[2])[2]
		for j in range(3):
			if (trizs[j]>boundminz): zminboundtri = False
		if (zminboundtri): minztets.append(boundtets[i])

nztets = minztets.__len__()
volztets = 0.0
for z in minztets:
	volztets += g.getTetVol(z)

for j in range(NITER):
    #if steps.mpi.rank == 0: print "iteration: ", j , "/", NITER
    sim.reset()
    sim.setCompDiffD('cyto', 'diffX', DCST)
    totset = 0
    for k in minztets:
        sim.setTetConc(k, 'X', CONC)
        sim.setTetClamped(k, 'X', True)
        totset+=sim.getTetCount(k, 'X')    
    for i in range(ntpnts):
        sim.run(tpnts[i])
        for k in range(SAMPLE):
            count = sim.getTetCount(int(tetidxs[k]), 'X')
            if steps.mpi.rank ==0:
                res[j, i, k] = count
#print '%d / %d' % (j + 1, NITER)
ndiff = sim.getDiffExtent()
niteration = sim.getNIteration()
nreac = sim.getReacExtent()

########################################################################
if steps.mpi.rank == 0:
    passed = False
    force_end = False
    
    itermeans = np.mean(res, axis = 0)
    tpnt_compare = [3, 4]
    max_err = 0.0

    for t in tpnt_compare:
        if force_end: break
        NBINS=10
        radmax = 0.0
        radmin = 11.0
        for r in tetrads:
            if (r > radmax): radmax = r
            if (r < radmin) : radmin = r
        
        rsec = (radmax-radmin)/NBINS
        binmins = np.zeros(NBINS+1)
        tetradsbinned = np.zeros(NBINS)
        r = radmin
        bin_vols = np.zeros(NBINS)
        
        for b in range(NBINS+1):
            binmins[b] = r
            if (b!=NBINS): tetradsbinned[b] = r +rsec/2.0
            r+=rsec
        
        bin_counts = [None]*NBINS
        for i in range(NBINS):
            bin_counts[i] = []
        filled = 0
        
        for i in range(itermeans[t].size):
            irad = tetrads[i]
            
            for b in range(NBINS):
                if(irad>=binmins[b] and irad<binmins[b+1]):
                    bin_counts[b].append(itermeans[t][i])
                    bin_vols[b]+=sim.getTetVol(int(tetidxs[i]))
                    filled+=1.0
                    break
        bin_concs = np.zeros(NBINS)
        for c in range(NBINS): 
            for d in range(bin_counts[c].__len__()):
                bin_concs[c] += bin_counts[c][d]
            bin_concs[c]/=(bin_vols[c])
            bin_concs[c]*=(1.0e-3/6.022e23)*1.0e6
        
        for i in range(NBINS):
            if (tetradsbinned[i] > 1 and tetradsbinned[i] < 4):
                rad = tetradsbinned[i]*1.0e-6
                det_conc =   (getConc(CONC*6.022e26, DCST, rad, tpnts[t])/6.022e26)*1.0e6         
                steps_conc = bin_concs[i]
                if tolerable(det_conc, steps_conc, tolerance):
                    passed = True
                else:
                    passed = False
                    force_end = True
                    break
                if (abs(2*(det_conc-steps_conc)/(det_conc+steps_conc)) > max_err): max_err = abs(2*(det_conc-steps_conc)/(det_conc+steps_conc))
    #print "Error:", abs(2*(det_conc-steps_conc)/(det_conc+steps_conc))*100, "%"
    print "Max error:", max_err*100, "%"
