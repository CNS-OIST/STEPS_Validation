########################################################################

# Deterministic degradation-diffusion process.

# AIMS: to verify STEPS spatial deterministic solver 'TetODE' 
# supports spatially-separated initial conditions and computes
# diffusion and reaction rates correctly when applied to a 
# degradation-diffusion process. 

# For a more detailed description of the analytical system and 
# equivalent STEPS model see:
# http://www.biomedcentral.com/content/supplementary/1752-0509-6-36-s4.pdf
# "Degradation-diffusion process with initially separated reactants"

# Verification also takes place of the necessary steps to build the model, 
# such as loading a mesh and recording from tetrahedrons. 

# Even though this is a deterministic model, a small tolerance of 1.5% is 
# permitted. This is because the model is not an exact replica of the 
# analytical system and small inaccuracies are produced from a finite 
# "reaction zone".
  
########################################################################

from __future__ import print_function, absolute_import

import steps.model as smod
import steps.geom as sgeom
import steps.rng as srng
import steps.solver as ssolv

import time 
import numpy as np
import steps.utilities.meshio as meshio
from . import tol_funcs

########################################################################

def test_kis_ode():
    "Reaction-diffusion - Degradation-diffusion (TetODE)"

    NITER = 1           # The number of iterations
    DT = 0.01           # Sampling time-step
    INT = 0.11          # Sim endtime

    DCSTA = 400*1e-12
    DCSTB = DCSTA
    RCST = 100000.0e6

    NA0 = 10000         # Initial number of A molecules
    NB0 = NA0           # Initial number of B molecules

    SAMPLE = 5000

    # create the array of tet indices to be found at random
    tetidxs = np.zeros(SAMPLE, dtype = 'int')
    # further create the array of tet barycentre distance to centre
    tetrads = np.zeros(SAMPLE)

    #Small expected error
    tolerance = 1.5/100

    ########################################################################
    rng = srng.create('r123', 512) 
    rng.initialize(1000) # The max unsigned long


    mdl  = smod.Model()

    A = smod.Spec('A', mdl)
    B = smod.Spec('B', mdl)

    volsys = smod.Volsys('vsys',mdl)

    R1 = smod.Reac('R1', volsys, lhs = [A,B], rhs = [])
    R1.setKcst(RCST)

    D_a = smod.Diff('D_a', volsys, A)
    D_a.setDcst(DCSTA)
    D_b = smod.Diff('D_b', volsys, B)
    D_b.setDcst(DCSTB)

    mesh = meshio.loadMesh('validation_rd/meshes/brick_40_4_4_STEPS')[0]
    ntets = mesh.countTets()

    VOLA = mesh.getMeshVolume()/2.0
    VOLB = VOLA

    comp1 = sgeom.TmComp('comp1', mesh, range(ntets))
    comp1.addVolsys('vsys')

    # Now fill the array holding the tet indices to sample at random
    assert(SAMPLE <= ntets)

    numfilled = 0
    while (numfilled < SAMPLE):
            if (ntets != SAMPLE):
                    max = mesh.getBoundMax()
                    min = mesh.getBoundMin()
                    
                    rnx = rng.getUnfII()
                    rny = rng.getUnfII()
                    rnz = rng.getUnfII()
                    
                    xpnt = min[0] + (max[0]-min[0])*rnx
                    ypnt = min[1] + (max[1]-min[1])*rny
                    zpnt = min[2] + (max[2]-min[2])*rnz
                    
                    idx = mesh.findTetByPoint([xpnt, ypnt, zpnt])
                    
                    if (idx == -1): continue
                    if (idx not in tetidxs): 
                            tetidxs[numfilled] = idx
                            numfilled += 1
            else : 
                    tetidxs[numfilled] = numfilled
                    numfilled +=1
    tetidxs.sort()

    # Now find the distance of the centre of the tets to the Z lower face

    for i in range(SAMPLE):
            baryc = mesh.getTetBarycenter(int(tetidxs[i]))
            r = baryc[0]
            tetrads[i] = r*1e6


    Atets = []
    Btets = []

    for t in range(ntets):
            baryx = mesh.getTetBarycenter(t)[0]
            if (baryx < 0.0): 
                    Atets.append(t)
                    continue
            if (baryx >= 0.0): 
                    Btets.append(t)
                    continue
            assert(False)


    sim = ssolv.TetODE(mdl, mesh, rng)
    sim.setTolerances(1.0e-3, 1.0e-3)

    tpnts = np.arange(0.0, INT, DT)
    ntpnts = tpnts.shape[0]

    resA = np.zeros((NITER, ntpnts, SAMPLE))
    resB = np.zeros((NITER, ntpnts, SAMPLE))


    for i in range (0, NITER):
        sim.setCompCount('comp1', 'A', 2*NA0)
        sim.setCompCount('comp1', 'B', 2*NB0)
        
        for t in Btets:
            sim.setTetCount(t, 'A', 0)
        for t in Atets:
            sim.setTetCount(t, 'B', 0)    

        for t in range(0, ntpnts):
            sim.run(tpnts[t])
            for k in range(SAMPLE):
                resA[i,t,k] = sim.getTetCount(int(tetidxs[k]), 'A')
                resB[i,t,k] = sim.getTetCount(int(tetidxs[k]), 'B')
            
    itermeansA = np.mean(resA, axis=0)
    itermeansB = np.mean(resB, axis=0)

    def getdetc(t, x):
        N = 1000        # The number to represent infinity in the exponential calculation
        L = 20e-6
        
        concA  = 0.0
        for n in range(N):
            concA+= ((1.0/(2*n +1))*np.exp((-(DCSTA/(20.0e-6))*np.power((2*n +1), 2)*np.power(np.pi, 2)*t)/(4*L))*np.sin(((2*n +1)*np.pi*x)/(2*L)))
        concA*=((4*NA0/np.pi)/(VOLA*6.022e26))*1.0e6    
        
        return concA


    tpnt_compare = [5, 10]

    for tidx in tpnt_compare:
        NBINS=50
        radmax = 0.0
        radmin = 10.0
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
        
        bin_countsA = [None]*NBINS
        bin_countsB = [None]*NBINS
        for i in range(NBINS):
            bin_countsA[i] = []
            bin_countsB[i] = []
        filled = 0
        
        for i in range(itermeansA[tidx].size):
            irad = tetrads[i]
            
            for b in range(NBINS):
                if(irad>=binmins[b] and irad<binmins[b+1]):
                    bin_countsA[b].append(itermeansA[tidx][i])
                    bin_vols[b]+=sim.getTetVol(int(tetidxs[i]))
                    filled+=1.0
                    break
        filled = 0
        for i in range(itermeansB[tidx].size):
            irad = tetrads[i]
            
            for b in range(NBINS):
                if(irad>=binmins[b] and irad<binmins[b+1]):
                    bin_countsB[b].append(itermeansB[tidx][i])
                    filled+=1.0
                    break
        
        bin_concsA = np.zeros(NBINS)
        bin_concsB = np.zeros(NBINS)
        
        for c in range(NBINS): 
            for d in range(bin_countsA[c].__len__()):
                bin_concsA[c] += bin_countsA[c][d]
            for d in range(bin_countsB[c].__len__()):
                bin_concsB[c] += bin_countsB[c][d]        
            
            bin_concsA[c]/=(bin_vols[c])
            bin_concsA[c]*=(1.0e-3/6.022e23)*1.0e6    
            bin_concsB[c]/=(bin_vols[c])
            bin_concsB[c]*=(1.0e-3/6.022e23)*1.0e6 
        
        for i in range(NBINS):
            rad = abs(tetradsbinned[i])*1.0e-6
            
            if (tetradsbinned[i] < -5):
                # compare A
                det_conc = getdetc(tpnts[tidx], rad)
                steps_conc = bin_concsA[i]
                assert tol_funcs.tolerable(det_conc, steps_conc, tolerance)

            if (tetradsbinned[i] > 5):
                # compare B
                det_conc = getdetc(tpnts[tidx], rad)
                steps_conc = bin_concsB[i]
                assert tol_funcs.tolerable(det_conc, steps_conc, tolerance)

########################################################################
# END
