########################################################################

# 1D diffusion in a finite tube with fully-reflective boundary.
# Circular "point source" from one face. 

# AIMS: to verify STEPS spatial-stochastic solver 'Tetexact' supports 
# local initial condition on boundary, calculates volume diffusion 
# rates correctly and imposes reflective boundary condition.

# STEPS equivalent model: 3D Stochastic diffusion in a cylinder; source 
# from tetrahedrons on boundary of one circular face.

# For a more detailed description of the analytical system and 
# equivalent STEPS model see:
# http://www.biomedcentral.com/content/supplementary/1752-0509-6-36-s4.pdf
# "1D diffusion in a finite tube from a point source at one end"

# Verification also takes place of model and mesh construction 
# components, particularly mesh loading and manipulation capabilities 
# with functions such as steps.utilities.meshio.loadMesh and 
# steps.geom.Tetmesh.getTetTriNeighb, steps.geom.Tetmesh.getVertex etc. 
# Localised recording by steps.solver.Tetexact.getTetCount is also verified. 

# A 5% tolerance is imposed when comparing the mean output from 10 
# stochastic simulations of the STEPS model to the analytical solution. 
# There is an expected probability of failure of < 1%.
  
########################################################################

import datetime
import time
import numpy as np
import unittest

try:
    from steps.geom import UNKNOWN_TET
except ImportError:
    UNKNOWN_TET  = -1
import steps.model as smodel
import steps.solver as solvmod
import steps.utilities.meshio as meshio
import steps.geom as stetmesh
import steps.rng as srng


from . import tol_funcs
from ..config import Configuration

configuration = Configuration(__file__)

########################################################################

NITER = 10
DT = 0.01
INT = 0.11

global NINJECT, DCST, tolerance, SAMPLE, MESHFILE, tetidxs, tetrads

# The number of initial molecules:
NINJECT = 10000	

DCST = 0.2e-9

# In tests, with good code, <1% fail with a tolerance of 5%
tolerance = 5.0/100

# The number of tets to sample at random:
SAMPLE = 1060	

MESHFILE = 'cyl_diam2__len10_1060tets'

# create the array of tet indices to be found at random
tetidxs = np.zeros(SAMPLE, dtype = 'int')
# further create the array of tet barycentre distance to centre
tetrads = np.zeros(SAMPLE)

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
    
    a = mesh.getBoundMax()[2]-mesh.getBoundMin()[2]
    area = mesh.getMeshVolume()/a
    
    ntets = mesh.countTets()
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    comp.addVolsys('cytosolv')
    
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
    
    return mesh, area, a

########################################################################

class TestBoundedDiffusion(unittest.TestCase):

    def test_bounddiff(self):
        "Diffusion - Bounded (Tetexact)"
        rng = srng.create('r123', 512) 
        rng.initialize(1000) # The max unsigned long

        m = gen_model()
        g, area, a = gen_geom()

        # And fetch the total number of tets to make the data structures
        ntets = g.countTets()

        sim = solvmod.Tetexact(m, g, rng)

        tpnts = np.arange(0.0, INT, DT)
        ntpnts = tpnts.shape[0]


        # Create the big old data structure: iterations x time points x concentrations
        res = np.zeros((NITER, ntpnts, SAMPLE))

        # Find the tets connected to the bottom face
        # First find all the tets with ONE face on a boundary
        boundtets = []
        #store the 0to3 index of the surface triangle for each of these boundary tets
        bt_srftriidx = []

        for i in range(ntets):
            tettemp = g.getTetTetNeighb(i)
            templist = [t for t in range(4) if tettemp[t] == UNKNOWN_TET]
            boundtets.append(i)
            bt_srftriidx.append(templist)

        assert len(boundtets) == len(bt_srftriidx)

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
        conc = NITER*6.022e23*1.0e-3/volztets

        for j in range(NITER):
            sim.reset()
            tetcount = int((1.0*NINJECT)/nztets)
            totset = 0
            for k in minztets:
                sim.setTetSpecCount(k, 'X', tetcount)
                totset+=tetcount
            for i in range(ntpnts):
                sim.run(tpnts[i])
                for k in range(SAMPLE):
                    res[j, i, k] = sim.getTetSpecCount(int(tetidxs[k]), 'X')

        itermeans = np.mean(res, axis = 0)

        ########################################################################

        D = DCST
        pi = np.pi
        nmax = 1000
        N = NINJECT
        N = int((1.0*NINJECT)/nztets)*nztets
        def getprob(x,t):
                if(x>a): 
                        print('x out of bounds')
                        return
                p=0.0
                for n in range(nmax):
                        if (n==0): A = np.sqrt(1.0/a)
                        else : A = np.sqrt(2.0/a)
                        p+= np.exp(-D*np.power((n*pi/a), 2)*t)*A*np.cos(n*pi*x/a)*A*a
                
                return p*N/a

        tpnt_compare = [6, 8, 10]
        passed = True
        max_err = 0.0

        for t in tpnt_compare:
            NBINS = 5
            
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
                if (tetradsbinned[i] > 2 and tetradsbinned[i] < 8):
                    rad = tetradsbinned[i]*1.0e-6
                    det_conc = (getprob(rad, tpnts[t])/area)*(1.0/6.022e20)
                    steps_conc = bin_concs[i]
                    assert tol_funcs.tolerable(det_conc, steps_conc, tolerance)

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(TestBoundedDiffusion))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
