########################################################################

# Test endocytosis

########################################################################

import steps.interface

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.sim import *
from steps.saving import *

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

########################################################################

# Simulation parameters
scale = 1e-6

########################################################################

# First order irreversible parameters
NITER = 1000
KCST = 10
spec_N = 10

########################################################################

# Raft and vesicle-related parameters
vesicle_diam = 5e-9
DCST = 1e-12

########################################################################

INT = 0.31
DT = 0.01

AVOGADRO = 6.022e23
LINEWIDTH = 2

########################################################################

model = Model()

with model:
    ssys = SurfaceSystem()
    spec = Species.Create()
    vesicle = Vesicle.Create(vesicle_diam, DCST)

    with ssys:
        endo = Endocytosis.Create(vesicle, KCST, spec, True)

########################################################################

mesh = TetMesh.LoadAbaqus('meshes/sphere_0.5D_577tets.inp', scale)

with mesh:
    cyto = Compartment.Create(mesh.tets)
    memb = Patch.Create(mesh.surface, cyto, None, ssys)

    with memb:
        for tri in memb.tris:
            EndocyticZone([tri])

########################################################################

rng = RNG('mt19937', 512, 100)

sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)

rs = ResultSelector(sim)

ves_count = rs.cyto.vesicle.Count
spec_count = rs.memb.spec.Count

sim.toSave(ves_count, spec_count, dt=DT)

with HDF5Handler('data/endocytosis') as hdf:
    sim.toDB(hdf, 'endocytosis')

    for i in range(NITER):
        if MPI.rank == 0:
            print(i + 1, 'of', NITER)

        sim.newRun()

        sim.memb.spec.Count = spec_N

        sim.run(INT)

if MPI.rank == 0:
    with HDF5Handler('data/endocytosis') as hdf:
        ves_count, spec_count = hdf['endocytosis'].results

        plt.subplot(121)

        tpnts = spec_count.time[0]
        mean_res = np.mean(spec_count.data, axis=0).flatten()
        std_res = np.std(spec_count.data, axis=0).flatten()

        analy = spec_N * np.exp(-KCST * tpnts)
        std = np.sqrt(spec_N * np.exp(-KCST * tpnts) * (1 - np.exp(-KCST * tpnts)))

        plt.errorbar(
            tpnts,
            analy,
            std,
            color='black',
            label='analytical',
            linewidth=LINEWIDTH,
        )
        p = plt.errorbar(
            tpnts + DT / 3,
            mean_res,
            std_res,
            color='cyan',
            ls='--',
            label='STEPS',
            linewidth=LINEWIDTH,
        )
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Dep. species number')

        plt.subplot(122)

        tpnts = ves_count.time[0]
        mean_res_v = np.mean(ves_count.data, axis=0).flatten()
        analy = spec_N * (1 - np.exp(-KCST * tpnts))

        plt.plot(
            tpnts,
            analy,
            color='black',
            label='analytical',
            linewidth=LINEWIDTH,
        )
        plt.plot(
            tpnts,
            mean_res_v,
            'c--',
            label='STEPS',
            linewidth=LINEWIDTH,
        )
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Vesicle number')
        fig = plt.gcf()
        fig.set_size_inches(7, 3.5)
        fig.savefig('plots/endocytosis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
