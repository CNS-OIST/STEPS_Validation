########################################################################

# Spatial stochastic production and degradation reaction-diffusion.

# AIMS: to verify STEPS spatial stochastic solver 'Tetexact' 
# solves the master equation closely when applied to a  
# zero-order production and second-order degradation reaction-diffusion
# process.

# For a more detailed description of the analytical system and 
# equivalent STEPS model see:
# http://www.biomedcentral.com/content/supplementary/1752-0509-6-36-s4.pdf
# "Lack of effect of slow diffusion on discrete production-degradation system"

# Verification also takes place of the necessary steps to build the model, 
# such as mesh-based compartment creation, random-number generator 
# construction and initialization, and recording from mesh-based compartment. 

# A 7.5% tolerance is imposed when comparing the stationary distribution 
# from 50000s of STEPS stochastic simulation to the analytical solution 
# to the chemical master equation, in the range 5-10 molecules. 
# There is an expected probability of failure of < 1%.
  
########################################################################

from __future__ import print_function, absolute_import

import numpy as np
import time 

import steps.model as smod
import steps.geom as sgeom
import steps.rng as srng
import steps.solver as ssolv
import steps.utilities.meshio as meshio

from . import tol_funcs
from .. import configuration

########################################################################

def test_masteq_diff():
    "Reaction-diffusion - Production and second order degradation (Tetexact)"

    ### NOW   A+B-> B,  0->A (see Erban and Chapman, 2009)

    ########################################################################
    SCALE = 1.0

    KCST_f = 100e6*SCALE			# The reaction constant, degradation
    KCST_b = (20.0e-10*SCALE) 		# The reaction constant, production

    DCST_A = 20e-12
    DCST_B = 20e-12

    B0 = 1   # The number of B moleucles

    DT = 0.1			# Sampling time-step
    INT = 50000.1 		# Sim endtime

    filename = 'cube_1_1_1_73tets.inp'

    # A tolerance of 7.5% will fail <1% of the time
    tolerance = 7.5/100

    ########################################################################

    mdl  = smod.Model()

    A = smod.Spec('A', mdl)
    B = smod.Spec('B', mdl)

    volsys = smod.Volsys('vsys',mdl)

    diffA = smod.Diff('diffA', volsys, A)
    diffA.setDcst(DCST_A)
    diffB = smod.Diff('diffB', volsys, B)
    diffB.setDcst(DCST_B)

    # Production
    R1 = smod.Reac('R1', volsys, lhs = [A, B], rhs = [B], kcst = KCST_f)
    R2 = smod.Reac('R2', volsys, lhs = [], rhs = [A], kcst = KCST_b)

    geom = meshio.loadMesh(configuration.mesh_path(filename))[0]

    comp1 = sgeom.TmComp('comp1', geom, range(geom.ntets))
    comp1.addVolsys('vsys')

    rng = srng.create('r123', 512)
    rng.initialize(1000)

    sim = ssolv.Tetexact(mdl, geom, rng)
    sim.reset()

    tpnts = np.arange(0.0, INT, DT)
    ntpnts = tpnts.shape[0]

    res = np.zeros([ntpnts])
    res_std1 = np.zeros([ntpnts])
    res_std2 = np.zeros([ntpnts])

    sim.reset()
    sim.setCompSpecCount('comp1', 'A', 0)
    sim.setCompSpecCount('comp1', 'B', B0)

    b_time = time.time()
    for t in range(0, ntpnts):
        sim.run(tpnts[t])
        res[t] = sim.getCompSpecCount('comp1', 'A')


    def fact(x): return (1 if x==0 else x * fact(x-1))

    # Do cumulative count, but not comparing them all. 
    # Don't get over 50 (I hope)
    steps_n_res = np.zeros(50)
    for r in res: steps_n_res[int(r)]+=1
    for s in range(50): steps_n_res[s] = steps_n_res[s]/ntpnts

    k1 = KCST_f/6.022e23
    k2 = KCST_b*6.022e23
    v = comp1.getVol()*1.0e3 # litres

    for m in range(5, 11):
        analy = (1.0/fact(m))*np.power((k2*v*v)/(B0*k1), m)*np.exp(-((k2*v*v)/(k1*B0)))
        assert tol_funcs.tolerable(steps_n_res[m], analy, tolerance)

########################################################################
# END
