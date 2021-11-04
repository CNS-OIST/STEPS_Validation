from __future__ import print_function # for backward compatibility with Py2
import steps.model as smodel
import steps.geom as sgeom
import steps.rng as srng
import steps.solver as ssolver
import steps.utilities.meshio as meshio
import numpy as np
import math
import time
from random import *

import matplotlib.pyplot as plt

initial = 0
# The number of simulation time-points
N_timepoints = 100

# The simulation dt
DT_sim = 1e-5 # seconds

TEMPERATURE = 30

# The membrane resistivity, ohm.m^2 (1/conductivity)
R_ro = 1.0/1.0

# The leak reversal potential. Koch uses 'relative to rest', so make this 0
Rrev = -65.0e-3

# Membrane capacitance
C_ro = 0

# Concentration of Na:    Hille
# Outer compartment
Na_oconc = 40e-6
# Inner compartment
Na_iconc = 20e-6

syn_count = 1

Na_G = 1

mdl = smodel.Model()
ssys = smodel.Surfsys('ssys', mdl)
vsys = smodel.Volsys('vsys', mdl)

#syn = smodel.Chan('syn', mdl)
#syn_cond = smodel.ChanState('syn_cond', mdl, syn)

Na = smodel.Spec('Na', mdl)
Na.setValence(2)
Na_diff = smodel.Diff('Na_diff', vsys, Na, 1e-9)

NaChan = smodel.Chan('NaChan', mdl)
NaChan_state_0 = smodel.ChanState('NaChan_state_0', mdl, NaChan)

GHKcurr = smodel.GHKcurr('GHKcurr', ssys, NaChan_state_0, Na, True)
GHKcurr.setP(1e-14)

mesh, node_proxy, tet_proxy, tri_proxy = meshio.importGmsh('comp2.msh', 1e-6)
tet_groups = tet_proxy.getGroups()
inner_tets = tet_groups[(0,'comp_i')]
outer_tets = tet_groups[(0,'comp_o')]
tri_groups = tri_proxy.getGroups()
patch_tris = tri_groups[(0,'patch')]

comp_i = sgeom.TmComp('comp_i', mesh, inner_tets)
comp_o = sgeom.TmComp('comp_o', mesh, outer_tets)
patch = sgeom.TmPatch('patch', mesh, patch_tris, comp_i, comp_o)

print ('Comp i volume: '+str(comp_i.getVol()))
print ('Comp o volume: '+str(comp_o.getVol()))

comp_i.addVolsys('vsys')
comp_o.addVolsys('vsys')
patch.addSurfsys('ssys')

memb_area = patch.getArea()

R = R_ro / memb_area
C = C_ro * memb_area

membrane = sgeom.Memb('membrane', mesh, [patch], opt_method=1)

# Create the random number generator
r = srng.create('mt19937', 512)
r.initialize(int(time.time() % 10000))
#

# Create solver object
sim = ssolver.Tetexact(mdl, mesh, r, True)
sim.setTemp(TEMPERATURE + 273)
sim.setEfieldDT(1e-7)

sim.setMembPotential('membrane', Rrev)
#
sim.setMembCapac('membrane', C_ro)
#
sim.setMembVolRes('membrane', R_ro)
sim.setMembRes('membrane', R_ro, Rrev)

sim.setCompConc('comp_i', 'Na', Na_iconc)
sim.setCompConc('comp_o', 'Na', Na_oconc)
sim.setPatchCount('patch', 'NaChan_state_0', syn_count)

count_i = [sim.getCompCount('comp_i', 'Na')]
count_o = [sim.getCompCount('comp_o', 'Na')]

for i in range(1, N_timepoints):
    sim.run(DT_sim *i)
    count_i.append(sim.getCompCount('comp_i', 'Na'))
    count_o.append(sim.getCompCount('comp_o', 'Na'))

plt.clf()
plt.plot(np.arange(N_timepoints) * DT_sim * 1e3, count_i, label='comp_i')
plt.plot(np.arange(N_timepoints) * DT_sim * 1e3, count_o, label='comp_o')
plt.legend(loc='best')
plt.xlabel('Time (msec)')
plt.show()

