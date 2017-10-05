########################################################################

# 1D diffusion in a semi-infinite tube with clamped source. 
# RESTORE

# AIMS: to verify checkpointing and restoring of the spatial stochastic 
# solver 'Tetexact' in the context of the Clamped-source Diffusion model 
# (see validation/csd_clamp.py)

########################################################################

from __future__ import print_function, absolute_import

import datetime
import os.path as osp
import steps.model as smodel
import numpy as np
import steps.solver as solvmod
import steps.geom as stetmesh
import steps.rng as srng
import steps.utilities.meshio as meshio
import time
import os

from . import tol_funcs
from . import csd_clamp_cp
from .. import configuration

print("Diffusion - Clamped:")

dir_checkpoint = "validation_cp/cp"
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)

rng = srng.create('mt19937', 512) 
rng.initialize(int(time.time()%4294967295)) # The max unsigned long

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
    dif_X = smodel.Diff('diffX', cytosolv, X)
    dif_X.setDcst(DCST)
    
    return mdl

########################################################################

def gen_geom():
    mesh = meshio.loadMesh(configuration.mesh_path(MESHFILE))[0]
    
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

def test_csdclamp():

    m = gen_model()
    g = gen_geom()

    # And fetch the total number of tets to make the data structures
    ntets = g.countTets()

    sim = solvmod.Tetexact(m, g, rng)

    tpnts = np.arange(0.0, INT, DT)
    ntpnts = tpnts.shape[0]


    #Create the big old data structure: iterations x time points x concentrations
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
        sim.restore('./validation_cp/cp/csd_clamp')
        for i in range(ntpnts):
            sim.run(tpnts[i])
            for k in range(SAMPLE):
                res[j, i, k] = sim.getTetCount(int(tetidxs[k]), 'X')
    #print('{0} / {1}'.format(j + 1, NITER))

    itermeans = np.mean(res, axis = 0)


    ########################################################################

    tpnt_compare = [3, 4]
    passed = True
    max_err = 0.0

    for t in tpnt_compare:
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
                assert(tol_funcs.tolerable(det_conc, steps_conc, tolerance))

