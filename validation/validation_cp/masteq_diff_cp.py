########################################################################

# Spatial stochastic production and degradation reaction-diffusion.
# CHECKPOINT

# AIMS: to verify checkpointing and restoring of the spatial stochastic 
# solver 'Tetexact' in the context of the Production-Degradation 
# Reaction-Diffusion model 
# (see validation/masteq_diff.py)
  
########################################################################

import math
import numpy
import time 
from pylab import *

import steps.model as smod
import steps.geom as sgeom
import steps.rng as srng
import steps.solver as ssolv
import steps.utilities.meshio as meshio

from tol_funcs import *

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

geom = meshio.loadMesh('./validation_rd/meshes/'+filename)[0]

comp1 = sgeom.TmComp('comp1', geom, range(geom.ntets))
comp1.addVolsys('vsys')

rng = srng.create('mt19937', 512)
rng.initialize(int(time.time()%4294967295))

sim = ssolv.Tetexact(mdl, geom, rng)
sim.reset()

tpnts = numpy.arange(0.0, INT, DT)
ntpnts = tpnts.shape[0]

res = numpy.zeros([ntpnts])
res_std1 = numpy.zeros([ntpnts])
res_std2 = numpy.zeros([ntpnts])

sim.reset()
sim.setCompCount('comp1', 'A', 0)
sim.setCompCount('comp1', 'B', B0)

sim.checkpoint('./validation_cp/cp/masteq_diff')


