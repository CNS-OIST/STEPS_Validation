########################################################################

# surface diffusion of rafts on mesh surface from central source.

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

# Number of iterations; plotting dt; sim endtime:
NITER = 1000

DT = 0.01
INT = 0.16

# Number of molecules injected in center; diff constant; number of tets sampled:
NINJECT = 100

DCST = 10e-12

raft_diam = 1.0e-8

########################################################################

model = Model()

with model:
    ssys = SurfaceSystem.Create()

    raft = Raft.Create(raft_diam, DCST)
    X = Species.Create()

    with ssys:
        Diffusion(X, 0)

########################################################################

mesh = TetMesh.Load(f'meshes/coin_10r_1h_13861')

with mesh:
    cyto = Compartment.Create(mesh.tets)

    patch_tris = TriList(
        [tri for tri in mesh.surface if all(v.z > 0 for v in tri.verts)])

    memb = Patch.Create(patch_tris, cyto, None, ssys)

    ctri = (patch_tris & mesh.tets[0, 0, 0.5e-6].faces)[-1]

    trirads = [np.linalg.norm(tri.center - ctri.center) for tri in memb.tris]
    triareas = [tri.Area for tri in memb.tris]

########################################################################

rng = RNG('r123', 1024, 100)
sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

rs = ResultSelector(sim)

rafts = rs.TRIS(memb.tris).raft.Count

sim.toSave(rafts, dt=DT)

for j in range(NITER):
    if MPI.rank == 0:
        print(j, 'of', NITER)
    sim.newRun()
    sim.TRI(ctri).raft.Count = NINJECT

    sim.run(INT)

tpnt_compare = range(3, 13, 2)

if MPI.rank == 0:
    first = True
    for t in tpnt_compare:
        bin_n = 20

        bins = np.histogram_bin_edges(trirads, bin_n)
        tri_bins = np.digitize(trirads, bins)

        # Compute average position for each bin
        bins_pos = np.bincount(tri_bins,
                               weights=trirads) / np.bincount(tri_bins)
        bin_areas = np.bincount(tri_bins, weights=triareas)

        bin_counts = np.bincount(tri_bins,
                                 weights=np.mean(rafts.data[:, t, :], axis=0))
        bin_concs = bin_counts / (bin_areas * 1e12)

        det_conc = 1.0e-12 * (NINJECT /
                              (4 * np.pi * DCST * rafts.time[0, t])) * (np.exp(
                                  (-1.0 * (bins_pos**2)) /
                                  (4 * DCST * rafts.time[0, t])))

        if first:
            plt.plot(bins_pos * 1e6,
                     det_conc,
                     'k-',
                     label='analytical',
                     linewidth=3)
            plt.plot(bins_pos * 1e6,
                     bin_concs,
                     'r--',
                     label='STEPS',
                     linewidth=3)
            first = False
        else:
            plt.plot(bins_pos * 1e6, det_conc, 'k-', linewidth=3)
            plt.plot(bins_pos * 1e6, bin_concs, 'r--', linewidth=3)

    plt.xlabel('Distance from origin ($\mu$m)')
    plt.ylabel('Raft density ($\mu$m$^{-2}$)')
    plt.legend()
    plt.xlim(0, 5)
    fig = plt.gcf()
    fig.set_size_inches(3.4, 3.4)
    fig.savefig("plots/raft_diff.pdf", dpi=300, bbox_inches='tight')
    plt.close()

########################################################################
# END