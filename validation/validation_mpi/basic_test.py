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

import steps
import steps.mpi
import steps.model as smodel
import steps.rng as srng
import steps.geom as sgeom
import steps.mpi.solver as solvmod
import steps.utilities.meshio as smeshio
import steps.utilities.meshctrl as meshctrl
import math
import numpy

MESHFILE = "meshes/3tets.inp"
DCST = 0.02e-9
RCST = 100000.0e6
SRCST = RCST
########################################################################

mdl = smodel.Model()
A = smodel.Spec('A', mdl)
B = smodel.Spec('B', mdl)
C = smodel.Spec('C', mdl)
D_surf = smodel.Spec('D_surf', mdl)

comp_vsys = smodel.Volsys('comp_vsys', mdl)
dif_A = smodel.Diff('diffA', comp_vsys, A, DCST)
dif_B = smodel.Diff('diffB', comp_vsys, B, DCST)
dif_C = smodel.Diff('diffC', comp_vsys, C, DCST)
reac_AB_C = smodel.Reac('reacAB_C', comp_vsys, [A, B], [C], kcst = RCST)

ssys = smodel.Surfsys('ssys', mdl)
sreac_CD_C = smodel.SReac('sreac_CD_C', ssys, ilhs = [C], slhs = [D_surf], srhs = [D_surf], orhs = [C], kcst = SRCST)
sdiff_D = smodel.Diff('sdiffD', ssys, D_surf, DCST)

########################################################################

mesh, nodeproxy, tetproxy, triproxy = smeshio.importAbaqus(MESHFILE, 1e-6)
# block connectivity 1-5-4
# tet connectivity 2-1-0
# host 2 add sync_host 3
# host 1 add sync_host 3
comp1 = sgeom.TmComp('comp1', mesh, [0, 1])
comp1.addVolsys('comp_vsys')
comp2 = sgeom.TmComp('comp2', mesh, [2])
comp2.addVolsys('comp_vsys')
patch_tri = meshctrl.findOverlapTris(mesh, [2], [0, 1])
patch = sgeom.TmPatch('patch', mesh, patch_tri, icomp = comp1, ocomp = comp2)
patch.addSurfsys('ssys')
tet_hosts = [0, 1, 2]
# patch tri id 5
tri_hosts = {patch_tri[0]:3}

rng = srng.create('r123', 512)
rng.initialize(steps.mpi.rank + 1000)

sim = solvmod.TetOpSplit(mdl, mesh, rng, False, tet_hosts, tri_hosts)

# test diffusion
sim.setCompCount("comp1", "A", 20)
sim.setCompCount("comp1", "B", 10)
sim.setPatchCount("patch", "D_surf", 5)

a_comp1 = sim.getCompCount("comp1", "A")
b_comp1 = sim.getCompCount("comp1", "B")
c_comp1 = sim.getCompCount("comp1", "C")
c_comp2 = sim.getCompCount("comp2", "C")
d_patch = sim.getPatchCount("patch", "D_surf")

if steps.mpi.rank == 0:
    print "A in comp1", a_comp1
    print "B in comp1", b_comp1
    print "C in comp1", c_comp1
    print "C in comp2", c_comp2
    print "D in patch", d_patch

sim.run(0.005)

a_comp1 = sim.getCompCount("comp1", "A")
b_comp1 = sim.getCompCount("comp1", "B")
c_comp1 = sim.getCompCount("comp1", "C")
c_comp2 = sim.getCompCount("comp2", "C")
d_patch = sim.getPatchCount("patch", "D_surf")

if steps.mpi.rank == 0:
    print "A in comp1", a_comp1
    print "B in comp1", b_comp1
    print "C in comp1", c_comp1
    print "C in comp2", c_comp2
    print "D in patch", d_patch




