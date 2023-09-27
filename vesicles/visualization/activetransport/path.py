import steps.interface

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.sim import *
from steps.saving import *

import numpy as np
import pickle
import os

os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)

scale = 1e-6
NTPNTS = 1000
DT = 0.002

ves_diam = 50e-9

########################################################################

model = Model()

with model:
    Spec1 = Species.Create()

    ssys = SurfaceSystem.Create()
    with ssys:
        Diffusion(Spec1, 0)

    vssys = VesicleSurfaceSystem.Create()
    with vssys:
        exo = Exocytosis.Create(1000000, deps=Spec1)

    Ves1 = Vesicle.Create(ves_diam, 0, vssys)

########################################################################

mesh = TetMesh.LoadAbaqus('meshes/sphere_0.5D_2088tets.inp', scale)

with mesh:
    comp = Compartment.Create(mesh.tets)
    patch = Patch.Create(mesh.surface, comp, None, ssys)

########################################################################

rng = RNG('mt19937', 512, 43909)

sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

act1 = sim.addVesiclePath('actin1')

act1.addPoint([0,-0.2e-6, 0])
act1.addPoint([0, 0.22e-6, 0])

act1.addBranch(1, {2: 1})

act1.addVesicle(Ves1, 0.4e-6, stoch_stepsize=(18e-9, 0.7))

rs = ResultSelector(sim)

vesPos = rs.comp.VESICLES().Pos

sim.toSave(vesPos, dt=DT)

########################################################################

tpnts = np.linspace(0, NTPNTS*DT, NTPNTS+1)
path_inj = range(0, NTPNTS, 200)
with XDMFHandler('data/path') as hdf:
    sim.toDB(hdf, 'path')
    sim.newRun()
    for i in range(len(tpnts)):
        if i in path_inj:
            v = sim.comp.addVesicle(Ves1)
            v.setPos((0, -0.2e-6, 0))

        sim.run(tpnts[i])

        for v in VesicleList(sim.comp.Ves1):
            if v.Pos[1] > 0:
                v('surf').Spec1.Count = 20

if MPI.rank == 0:
    with HDF5Handler('data/path') as hdf:
        group = hdf['path']
        vesPos, = group.results
        with open('data/path.pkl', 'wb') as f:
            pickle.dump(vesPos.data[0,:,0], f)
