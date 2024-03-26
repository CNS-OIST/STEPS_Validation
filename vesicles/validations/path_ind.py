import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"


scale = 1e-6
ENDT = 5
DT = 0.001

ves_diam = 50e-9

########################################################################

model = Model()
with model:
    ssys = SurfaceSystem.Create()
    vsys = VolumeSystem.Create()
    ves_ssys = VesicleSurfaceSystem.Create()

    Spec1, Spec2 = Species.Create()

    Ves1 = Vesicle.Create(ves_diam, 1e-12, ves_ssys)

    with vsys:
        Diffusion(Spec1, 1e-12)

########################################################################

mesh = TetMesh.LoadAbaqus('meshes/sphere_0.5D_2088tets.inp', scale)

with mesh:
    comp1 = Compartment.Create(mesh.tets, vsys)
    patch1 = Patch.Create(mesh.surface, comp1, None, ssys)

########################################################################

rng = RNG('mt19937', 512, 987)
sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

act1 = sim.addVesiclePath('actin1')
act2 = sim.addVesiclePath('actin2')


act1.addPoint([0, 0, -1.5e-7])
act1.addPoint([0, 0, 0])
act1.addPoint([0, 0, 2.0e-7])

act1.addBranch(1, {2: 1})
act1.addBranch(2, {3: 1})

act1.addVesicle(Ves1, 1e-6)

ves = sim.comp1.addVesicle(Ves1)

rs = ResultSelector(sim)

vesPos = rs.VESICLE(ves).Pos

sim.toSave(vesPos, dt=DT)

with HDF5Handler('data/path_ind') as hdf:
    sim.toDB(hdf, f'path_ind')

    sim.newRun(reset=False)

    sim.run(ENDT)

if MPI.rank == 0:
    with HDF5Handler('data/path_ind') as hdf:
        vesPos, = hdf['path_ind'].results
        plt.plot(vesPos.time[0], vesPos.data[0, :, 0][:, 2])
        plt.xlabel('Time (s)')
        plt.ylabel('z position ($\mu$m)')
        fig = plt.gcf()
        fig.set_size_inches(7, 3.5)
        fig.savefig("plots/path_ind.pdf", dpi=300, bbox_inches='tight')
        plt.close()
