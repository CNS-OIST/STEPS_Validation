########################################################################

# Test generation and dissociation rate of Rafts

########################################################################

import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

from matplotlib import pyplot as plt
import numpy as np

########################################################################

# Simulation parameters
scale = 1e-6

########################################################################

# The reaction parameters
KCST_f = 10.0
KCST_b = 1.0

# Raft-related parameters
raft_diam = 10e-9
DCST = 1e-12

########################################################################

INT = 100000.1
DT = 0.1

AVOGADRO = 6.022e23
LINEWIDTH = 3

########################################################################

mdl = Model()

with mdl:
    surfsys = SurfaceSystem.Create()
    raftsys = RaftSurfaceSystem.Create()

    raft = Raft.Create(raft_diam, DCST, raftsys)
    spec = Species.Create()

    with surfsys:
        Diffusion(spec, 0)
        raftgen = RaftGen.Create(raft, KCST_f, spec)

    with raftsys:
        raftdis = RaftDis.Create(KCST_b, spec)

########################################################################

mesh = TetMesh.LoadAbaqus('meshes/sphere_0.5D_577tets.inp', scale)

with mesh:
    cyto = Compartment.Create(mesh.tets)
    memb = Patch.Create(mesh.surface, cyto, None, surfsys)

########################################################################

rng = RNG('mt19937', 512, 100)

sim = Simulation('TetVesicle', mdl, mesh, rng, MPI.EF_NONE)

rs = ResultSelector(sim)

raft_count = rs.memb.raft.Count

sim.toSave(raft_count, dt=DT)

with HDF5Handler('data/raft_gendis') as hdf:
    sim.toDB(hdf, f'raft_gendis')
    sim.newRun()

    sim.memb.spec.Count = 1
    sim.memb.spec.Clamped = True

    sim.run(INT)

if MPI.rank == 0:
    with HDF5Handler('data/raft_gendis') as hdf:
        raft_count, = hdf['raft_gendis'].results

        def fact(x):
            return 1 if x == 0 else x * fact(x - 1)

        steps_n_res = np.zeros(50)
        analy = np.zeros(50)
        k1 = KCST_b
        k2 = KCST_f

        res = np.array(raft_count.data).flatten().astype(int)
        for r in res:
            if r < 50:
                steps_n_res[r] += 1
        ntpnts = INT / DT
        for s in range(50):
            steps_n_res[s] = steps_n_res[s] / ntpnts
            analy[s] = 1.0 / fact(s) * np.power(k2 / k1, s) * np.exp(-k2 / k1)

        plt.bar(range(50), analy, label='analytical')
        plt.bar(range(50), steps_n_res, label='STEPS', alpha=0.5)

        plt.legend()
        plt.xlabel('Raft number')
        plt.ylabel('Probability')
        plt.xlim(0, 24)
        fig = plt.gcf()
        fig.set_size_inches(3.4, 3.4)
        fig.savefig('plots/raft_gendis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
