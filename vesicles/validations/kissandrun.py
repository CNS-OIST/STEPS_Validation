########################################################################

# Test kiss and run exocytosis

########################################################################

import steps.interface

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.sim import *
from steps.saving import *

from matplotlib import pyplot as plt
import numpy as np

########################################################################

# Simulation parameters
scale = 1e-6

########################################################################

# First order irreversible parameters
NITER = 500
KCST = 5
ves_N = 10
glu_N = 100 # per vesicle

# Raft and vesicle-related parameters
vesicle_diam = 40e-9
DCST = 1e-12

########################################################################

INT = 1.04
DT = 0.04

AVOGADRO = 6.022e23
LINEWIDTH = 2

########################################################################

for pr in [0.1, 0.4, 0.7, 1.0]:
    model = Model()

    with model:
        glu, Spec1, Spec2 = Species.Create()
        vesicle = Vesicle.Create(vesicle_diam, DCST)
        vssys = VesicleSurfaceSystem.Create()

        with vssys:
            exo = Exocytosis.Create(KCST, deps=Spec1.v, kissAndRun=True, knrSpecChanges=[(Spec1, Spec2)], knrPartialRelease=pr)

    ########################################################################

    mesh = TetMesh.LoadAbaqus('meshes/sphere_0.5D_577tets.inp', scale)

    with mesh:
        cyto = Compartment.Create(mesh.tets)
        memb = Patch.Create(mesh.surface, cyto)

    ########################################################################

    rng = RNG('mt19937', 512, int(pr*100))

    sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

    rs = ResultSelector(sim)

    ves_count = rs.cyto.vesicle.Count
    glu_count = rs.cyto.vesicle('in').glu.Count

    sim.toSave(ves_count, glu_count, dt=DT)

    with HDF5Handler('data/kissandrun') as hdf:
        sim.toDB(hdf, f'kissandrun_pr{pr}', pr=pr)
        for i in range(NITER):
            if MPI.rank == 0:
                print(i + 1, 'of', NITER)

            sim.newRun()

            sim.cyto.vesicle.Count = ves_N
            sim.cyto.VESICLES()('in').glu.Count = glu_N
            sim.cyto.VESICLES()('surf').Spec1.Count = 100

            sim.run(INT)


if MPI.rank == 0:
    sp_counter = 1
    with HDF5Handler('data/kissandrun') as hdf:
        for pr in sorted(hdf.parameters['pr']):
            ves_count, glu_count = hdf.get(pr=pr).results

            tpnts = ves_count.time[0]
            mean_res = np.mean(glu_count.data[:NITER, ...], axis=0).flatten()
            std_res = np.std(glu_count.data[:NITER, ...], axis=0).flatten()

            analy = pr * glu_N * ves_N * np.exp(-KCST * tpnts) + ((1-pr) * glu_N * ves_N)
            
            plt.subplot(2,2,sp_counter)

            plt.plot(tpnts,
                     mean_res,
                     color='red',
                     ls='-',
                     label='STEPS, pr: '+str(pr),
                     linewidth=LINEWIDTH)
                        
            plt.plot(tpnts,
                     analy,
                     color='black',
                     ls='--',
                     label='analytical',
                     linewidth=LINEWIDTH)

            plt.legend()
            if sp_counter in (3,4):
                plt.xlabel('Time (s)')
            if sp_counter in (1,3):
                plt.ylabel('Intraluminal species')
            sp_counter+=1

    fig = plt.gcf()
    fig.set_size_inches(6.8, 6.8)
    fig.savefig('plots/kissandrun.pdf', dpi=300, bbox_inches='tight')
    plt.close()
