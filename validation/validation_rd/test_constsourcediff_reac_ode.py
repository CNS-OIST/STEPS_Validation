########################################################################

# 1D diffusion in a finite tube with constant influx at both ends. 

# AIMS: to verify STEPS spatial-deterministic solver 'TetODE' supports 
# localisation of reactions (here to boundary tetrahedrons), and
# calculates 1st-order reaction rates and diffusion rates correctly.

# STEPS equivalent model: 3D deterministic diffusion in a cylinder. 
# Constant influx source is modelled as a 1st-order deterministic reaction,
# localised to tetrahedrons on boundary of both circular faces.

# For a more detailed description of the analytical system and 
# equivalent STEPS model see:
# http://www.biomedcentral.com/content/supplementary/1752-0509-6-36-s4.pdf
# "1D diffusion in a finite tube with constant influx at both ends"

# Verification takes place of model and mesh construction 
# components, particularly mesh loading and manipulation capabilities 
# with functions such as steps.utilities.meshio.loadMesh and 
# steps.geom.Tetmesh.getBoundMin/Max, steps.geom.Tetmesh.getVertex etc. 
# Localised recording by steps.solver.TetODE.getTetCount is also verified. 

# Even though this is a deterministic model, a tolerance of 3% is 
# permitted. This is because an infinitely thin place source is 
# not replicated in STEPS and a small error is introduced 
# by the finite tetrahedral source.
  
########################################################################

import datetime
import time
import numpy as np
import unittest

try:
    from steps.geom import UNKNOWN_TET
except ImportError:
    UNKNOWN_TET = -1
import steps.geom as stetmesh
import steps.model as smodel
import steps.rng as srng
import steps.solver as solvmod
import steps.utilities.meshio as meshio

from . import tol_funcs
from ..config import Configuration

configuration = Configuration(__file__)

########################################################################



DT = 0.001
INT = 0.025

DCST = 0.05e-9
FLUX = 300000.0	# The flux at the boundaries, 

SAMPLE = 12033	# All tets

MESHFILE = 'cyl_diam0_4__len10_12Ktets_STEPS'

# Small expected error from non plane source
tolerance = 3.0/100

# create the array of tet indices to be found at random
tetidxs = np.zeros(SAMPLE, dtype = 'int')
# further create the array of tet barycentre distance to centre
tetrads = np.zeros(SAMPLE)

	
########################################################################

def gen_model():
   
    mdl = smodel.Model()
    
    X = smodel.Spec('X', mdl)
    A = smodel.Spec('A', mdl)
    # Vol/surface systems
    cytosolv = smodel.Volsys('cytosolv', mdl)
    
    dif_X = smodel.Diff('diffX', cytosolv, X)
    dif_X.setDcst(DCST)
    
    reac_X = smodel.Reac('reacX', cytosolv, lhs = [A], rhs=[A,X])
    
    return mdl

########################################################################

def gen_geom():
    mesh = meshio.loadMesh(configuration.mesh_path(MESHFILE))[0]
    
    ntets = mesh.countTets()
    
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    comp.addVolsys('cytosolv')
    
    # Now fill the array holding the tet indices to sample at random
    assert(SAMPLE <= ntets)
    numfilled=0
    while (numfilled < SAMPLE):
        tetidxs[numfilled] = numfilled
        numfilled+=1
    tetidxs.sort()

    # Now find the distance of the centre of the tets to the center
    for i in range(SAMPLE):
        baryc = mesh.getTetBarycenter(int(tetidxs[i]))
        min = (mesh.getBoundMin()[2] + mesh.getBoundMax()[2])/2.0
        r = baryc[2] - min
        # Convert to microns
        tetrads[i] = r*1.0e6
    
    return mesh

########################################################################

class TestConstSourceDiffReacODE(unittest.TestCase):
    def test_constsourcediff_reac_ode(self):
        "Reaction-diffusion - Constant source from reaction (TetODE)"

        rng = srng.create('r123', 512) 
        rng.initialize(1000) # The max unsigned long

        m = gen_model()
        g = gen_geom()
        real_vol = g.getMeshVolume()

        #Effective area
        area = real_vol/10e-6

        # And fetch the total number of tets to make the data structures
        ntets = g.countTets()

        sim = solvmod.TetODE(m, g)
        sim.setTolerances(1e-3, 1e-3)

        tpnts = np.arange(0.0, INT, DT)
        ntpnts = tpnts.shape[0]


        #Create the data structure: time points x concentrations
        res = np.zeros((ntpnts, SAMPLE))

        # Find the tets connected to the bottom face
        # First find all the tets with ONE face on a boundary

        boundtets = []
        #store the 0to3 index of the surface triangle for each of these boundary tets
        bt_srftriidx = []

        for i in range(ntets):
            tettemp = g.getTetTetNeighb(i)
            templist = [t for t in range(4) if tettemp[t] == UNKNOWN_TET]
            if templist:
                boundtets.append(i)
                bt_srftriidx.append(templist)

        assert len(boundtets) == len(bt_srftriidx)
                        
        minztets = []
        maxztets = []
        boundminz = g.getBoundMin()[2] + 0.01e-06
        boundmaxz = g.getBoundMax()[2] -0.01e-06
        minztris=[]
        maxztris=[]

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
                if (zminboundtri): 
                    minztets.append(boundtets[i])
                    minztris.append(zminboundtri)
                    continue
                
                zmaxboundtri = True
                for j in range(3):
                    if(trizs[j] < boundmaxz): zmaxboundtri = False
                if (zmaxboundtri):
                    maxztets.append(boundtets[i])
                    maxztris.append(zmaxboundtri)

        nminztets = minztets.__len__()
        nmaxztets = maxztets.__len__()


        totset = 0
        for k in minztets:    
            sim.setTetReacK(k, 'reacX', FLUX/nminztets)		
            sim.setTetSpecCount(k, 'A', 1)
            totset+=sim.getTetSpecCount(k, 'X')
        for l in maxztets:
            sim.setTetReacK(l, 'reacX', FLUX/nmaxztets)
            sim.setTetSpecCount(l, 'A', 1)
            totset+=sim.getTetSpecCount(l, 'X')
            
        for i in range(ntpnts):
            sim.run(tpnts[i])
            for k in range(SAMPLE):
                res[i, k] = sim.getTetSpecCount(int(tetidxs[k]), 'X')*1.0e6

        ########################################################################

        L = 5.0e-6
        a = 10e-6
        D = DCST
        pi = np.pi
        J = FLUX/area

        tpnt_compare = [12, 24]

        for t in tpnt_compare:
            NBINS = 50
            
            radmax = 0.0
            radmin = 0.0
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
            
            for i in range(res[t].size):
                irad = tetrads[i]
                
                for b in range(NBINS):
                    if(irad>=binmins[b] and irad<binmins[b+1]):
                        bin_counts[b].append(res[t][i])
                        bin_vols[b]+=sim.getTetVol(int(tetidxs[i]))
                        filled+=1.0
                        break
            bin_concs = np.zeros(NBINS)
            for c in range(NBINS): 
                for d in range(bin_counts[c].__len__()):
                    bin_concs[c] += bin_counts[c][d]
                bin_concs[c]/=(bin_vols[c])
                bin_concs[c]*=(1.0e-3/6.022e23)
            
            for i in range(NBINS):
                if (tetradsbinned[i] > -5 and tetradsbinned[i] < -3) \
                    or (tetradsbinned[i] > 3 and tetradsbinned[i] < -5):
                    rad = tetradsbinned[i]*1.0e-6
                    nsum = 0.0
                    nmax=100
                    for n in range(1,nmax):
                        nsum+= (np.power(-1., n)/np.power(n, 2))*np.exp(-D*(np.power(((n*pi)/L), 2)*\
                    tpnts[t]))*np.cos((n*pi*rad)/L)
                    det_conc = (1.0/6.022e20)*((J*L)/D)*(((D*tpnts[t])/np.power(L, 2))+((3*np.power(rad, 2) - np.power(L, 2))/(6*np.power(L, 2))) -(2/np.power(pi, 2))*nsum)
                    steps_conc = bin_concs[i]
                    assert tol_funcs.tolerable(det_conc, steps_conc, tolerance)
                    
########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.makeSuite(TestConstSourceDiffReacODE, "test"))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
