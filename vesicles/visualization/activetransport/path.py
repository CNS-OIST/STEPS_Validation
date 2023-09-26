
from numpy import linspace, array

import steps.model as smodel
import steps.mpi
import steps.mpi.solver as ssolv
import steps.geom as stetmesh
import steps.rng as srng

import steps.utilities.meshio as smeshio
import steps.utilities.meshctrl as smeshctrl


myrank = steps.mpi.rank


scale = 1e-6
NTPNTS = 1000
DT=0.002

ves_diam=50e-9

########################################################################

mdl = smodel.Model()

V = smodel.Vesicle('Ves', mdl, ves_diam, 0)

Spec1 = smodel.Spec('Spec1', mdl)

ves_ssys = smodel.VesSurfsys('ves_ssys', mdl)


volsys = smodel.Volsys('volsys', mdl)



surf_sys =smodel.Surfsys('surf_sys', mdl)

s1diff = smodel.Diff('Spec1_diff', surf_sys, Spec1, 0)

endoA = smodel.Exocytosis('exo', ves_ssys, kcst=1000000, deps = [Spec1])


V.addVesSurfsys('ves_ssys')


########################################################################


mesh = smeshio.importAbaqus('meshes/sphere_0.5D_2088tets.inp', scale)[0]

# Find the total number of tetrahedrons in the mesh
ntets = mesh.countTets()

comp1 = stetmesh.TmComp('comp1', mesh, range(ntets))
comp1.addVolsys('volsys')


patch1 = stetmesh.TmPatch('patch1', mesh, mesh.getSurfTris(), comp1)
patch1.addSurfsys('surf_sys')


rng = srng.create('mt19937', 512) 
rng.initialize(43907)


sim = ssolv.TetVesicle(mdl, mesh, rng, ssolv.EF_NONE)
sim.setOutputSync(False, 0)


sim.createPath('actin1')


sim.addPathPoint('actin1', 1, [0,-0.2e-6, 0])
sim.addPathPoint('actin1', 2, [0, 0.22e-6, 0])

sim.addPathBranch('actin1', 1, {2:1})


sim.addPathVesicle('actin1', 'Ves', 0.4e-6, stoch_stepsize  = [18e-9,0.7])


tpnts = linspace(0,NTPNTS*DT, NTPNTS+1)


if myrank == 0:
    ofile_ves = open('data/path_ves.txt', 'w')

path_inj = range(0,NTPNTS,200)

for i in range(len(tpnts)):
    if (i in path_inj):
        idx = sim.addCompVesicle('comp1', 'Ves')
        sim.setCompSingleVesiclePos('comp1', 'Ves', idx, (0,-0.2e-6, 0))

    sim.run(tpnts[i])

    for v in sim.getCompVesicleIndices('comp1', 'Ves'):
        vpos = sim.getSingleVesiclePos('Ves', v)
        if (vpos[1] > 0.0): sim.setSingleVesicleSurfaceSpecCount('Ves', v, 'Spec1', 20)
        pos = array(vpos)/scale
        if myrank == 0:
            ofile_ves.write(str(pos[0])+' '+str(pos[1])+' '+str(pos[2])+' ')
    if myrank == 0:
        ofile_ves.write('\n')

if myrank == 0:
    ofile_ves.close()
