from __future__ import print_function # for backward compatibility with Py2
import steps.model as smodel
import steps.geom as sgeom
import steps.rng as srng
# Parallel TetOpSplit
import steps.mpi
import steps.mpi.solver as parallel_solver
import steps.utilities.geom_decompose as gd
import steps.utilities.meshio as meshio
import numpy as np
import math
import time
from random import *
import sys

from steps.geom import UNKNOWN_TET, UNKNOWN_TRI,INDEX_DTYPE

import matplotlib.pyplot as plt


initial = 0
# The number of simulation time-points
N_timepoints = 100

# The simulation dt
DT_sim = 1e-4 # seconds

TEMPERATURE = 30

# The membrane resistivity, ohm.m^2 (1/conductivity)
R_ro = 1.0/1.0

# The leak reversal potential. Koch uses 'relative to rest', so make this 0
Rrev = -65.0e-3

# Membrane capacitance
C_ro = 1

Avogadro = 6.02214179e23

# Concentration of Na:    Hille
# Outer compartment


# Na_count = 40e6
comp1_conc = 2e-19#1*Na_count/Avogadro
outer_conc = 10*2e-19#1*Na_count/Avogadro
P = 1e-14

syn_count = 1e9

Na_G = 1

mdl = smodel.Model()
ssys = smodel.Surfsys('ssys', mdl)
vsys = smodel.Volsys('vsys', mdl)

#syn = smodel.Chan('syn', mdl)
#syn_cond = smodel.ChanState('syn_cond', mdl, syn)

Na = smodel.Spec('Na', mdl)
D = smodel.Spec('D', mdl)
E = smodel.Spec('E', mdl)
Na.setValence(2)

Na2Dreac = smodel.Reac('Na2Dreac', vsys, lhs=[Na], rhs=[D], kcst=1.3e-3)
D2Ereac = smodel.Reac('D2Ereac', vsys, lhs=[D], rhs=[E], kcst=1.3e-3)
E2Nareac = smodel.Reac('E2Nareac', vsys, lhs=[E], rhs=[Na], kcst=1.3e-3)

NaChan = smodel.Chan('NaChan', mdl)
NaChan_0 = smodel.ChanState('NaChan_0', mdl, NaChan)

GHKcurr = smodel.GHKcurr('GHKcurr', ssys, NaChan_0, Na, virtual_oconc = outer_conc, computeflux = True)
GHKcurr.setP(P)


LChan = smodel.Chan('LChan', mdl)
LChan_0 = smodel.ChanState('LChan_0', mdl, LChan)
OC_LeakChan = smodel.OhmicCurr('OC_LeakChan', ssys, chanstate = LChan_0, erev = Rrev, g = 1/R_ro )


mesh, node_proxy, tet_proxy, tri_proxy = meshio.importGmsh(
    '/home/katta/projects/HBP_STEPS/test/mesh/2tets_2patches_1comp.msh', 1)
tet_groups = tet_proxy.getGroups()
inner_tets = tet_groups[(0,'comp1')]
# outer_tets = tet_groups[(0,'comp_o')]
tri_groups = tri_proxy.getGroups()
patch_tris = tri_groups[(0,'patch1')]

comp1 = sgeom.TmComp('comp1', mesh, inner_tets)
# comp_o = sgeom.TmComp('comp_o', mesh, outer_tets)
patch1 = sgeom.TmPatch('patch1', mesh, patch_tris, comp1)

print ('Comp 1 volume: '+str(comp1.getVol()))
# print ('Comp o volume: '+str(comp_o.getVol()))

comp1.addVolsys('vsys')
# comp_o.addVolsys('vsys')
patch1.addSurfsys('ssys')

memb_area = patch1.getArea()
print ('Patch 1 area: '+str(memb_area))

R = R_ro / memb_area
C = C_ro * memb_area

memb1 = sgeom.Memb('memb1', mesh, [patch1], opt_method=1)


tet_hosts = gd.linearPartition(mesh, [1, 1, steps.mpi.nhosts])
tri_hosts = gd.partitionTris(mesh, tet_hosts, patch_tris)


# Create the random number generator
r = srng.create('mt19937', 512)
r.initialize(int(sys.argv[1]))
#
# Create solver object
sim = parallel_solver.TetOpSplit(mdl, mesh, r, parallel_solver.EF_DV_PETSC, tet_hosts, tri_hosts)
sim.setTemp(TEMPERATURE + 273)
sim.setEfieldDT(1e-5)

sim.setMembPotential('memb1', Rrev)
#
sim.setMembCapac('memb1', C_ro)
#
sim.setMembVolRes('memb1', R_ro)





# sim.setMembRes('memb1', R_ro, Rrev)

# sim.setCompCount('comp1', 'Na', Na_count)
sim.setCompConc('comp1', 'Na', comp1_conc)
# sim.setCompCount('comp_o', 'Na', Na_ocount)
sim.setPatchCount('patch1', 'NaChan_0', syn_count)
sim.setPatchCount('patch1', 'LChan_0', syn_count)

count1 = []
t = np.arange(N_timepoints + 1) * DT_sim
patch_tris_v_np = np.array(patch_tris, dtype = INDEX_DTYPE)
patch_tris_v = np.zeros(patch_tris_v_np.size, dtype=np.double)
V_min = []
V_max = []
for i in t:
    sim.run(i)
    # count1.append(sim.getCompConc('comp1', 'Na'))
    count1.append(sim.getCompCount('comp1', 'Na'))
    sim.getBatchTriVsNP(patch_tris_v_np, patch_tris_v)

    V_min.append(np.amin(patch_tris_v))
    V_max.append(np.amax(patch_tris_v))





# plt.clf()
# # print(count1)
#
# plt.plot(t, count1, label='comp1')
# plt.legend(loc='best')
# plt.xlabel('Time (msec)')
# plt.show()
#
# print(t)


import pandas as pd

df = pd.DataFrame({"t":t, "count1":count1, "V_min":V_min, "V_max":V_max})



df.to_csv(f'/home/katta/projects/HBP_STEPS_local_sims/GHKEfieldUnitTest/results/steps3/res{int(sys.argv[1])}.txt', sep=" ",
          index=False)
