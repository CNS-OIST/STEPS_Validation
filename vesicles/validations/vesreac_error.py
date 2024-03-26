########################################################################

# Test 4 different types of vesicle surface reaction: 2 first-order
# reactions and 3 second-order reactions interacting with cytosolic species

########################################################################

import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

import itertools
import math
import os
import sys
import time

########################################################################


def run_sim(hdfPath, ntets, D, NITER, use_foi, use_soAA, seed):

    # Vesicle-related parameters
    ves_N = 10
    ves_diam = 40e-9

    ########################################################################

    # First order irreversible parameters
    KCST_foi = 10
    spec_A_foi_number_perves = 10
    spec_A_foi_N = spec_A_foi_number_perves * ves_N

    ########################################################################

    # Second order irreversible AA parameters
    KCST_soAA = 2e6
    spec_A_soAA_number_perves = 100
    spec_B_soAA_number_incomp = ves_N * spec_A_soAA_number_perves

    INT = 0.21
    DT = 0.01
    AVOGADRO = 6.022e23

    ########################################################################

    model = Model()
    r = ReactionManager()

    with model:
        vsys = VolumeSystem.Create()
        vssys = VesicleSurfaceSystem.Create()

        ves = Vesicle.Create(ves_diam, D*1e-12, vssys)
        A_foi = Species.Create()
        A_soAA, B_soAA, C_soAA = Species.Create()

        with vssys:
            # First order irreversible
            A_foi.v > r[1] > None
            r[1].K = KCST_foi

            # Second order irreversible AA
            B_soAA.o + A_soAA.v > r[1] > C_soAA.v
            r[1].K = KCST_soAA

        with vsys:
            Diffusion(B_soAA, 10e-12)

    scale, diam = (1e-6, '0.5D') if ntets in [577, 2088] else (0.25e-6, '2D')
    mesh = TetMesh.LoadAbaqus(
        'meshes/sphere_'+diam+'_'+str(ntets)+'tets.inp', scale)

    with mesh:
        comp = Compartment.Create(mesh.tets, vsys)
        memb = Patch.Create(mesh.surface, comp, None)

    rng = RNG('mt19937', 512, seed)
    sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

    CONCA_soAA = (ves_N * spec_A_soAA_number_perves) / \
        (AVOGADRO * comp.Vol * 1e3)
    CONCB_soAA = spec_B_soAA_number_incomp / (AVOGADRO * comp.Vol * 1e3)

    rs = ResultSelector(sim)

    volfact = AVOGADRO * comp.Vol * 1e3

    rs_foi = rs.comp.ves('surf').A_foi.Count
    rs_soAA = rs.comp.ves('surf').A_soAA.Count / volfact

    sim.toSave(rs_foi, rs_soAA, dt=DT)

    with HDF5Handler(hdfPath) as hdf:
        sim.toDB(hdf, f'sim_{ntets}_{D}',
                 ntets=ntets, D=D, spec_A_foi_N=spec_A_foi_N, CONCA_soAA=CONCA_soAA, KCST_foi=KCST_foi,
                 KCST_soAA=KCST_soAA, use_foi=use_foi, use_soAA=use_soAA
                 )
        for i in range(NITER):

            sim.newRun()

            sim.comp.ves.Count = ves_N

            if use_foi:
                sim.comp.VESICLES()('surf').A_foi.Count = spec_A_foi_number_perves

            if use_soAA:
                sim.comp.B_soAA.Count = spec_B_soAA_number_incomp
                sim.comp.VESICLES()('surf').A_soAA.Count = spec_A_soAA_number_perves

            sim.run(INT)


if __name__ == '__main__':
    if sys.argv[1] == 'runSingle':
        hdfPath, ntets, D, NITER, use_foi, use_soAA, seed = sys.argv[2:]
        run_sim(hdfPath, int(ntets), float(D), int(NITER),
                use_foi == 'True', use_soAA == "True", int(seed))
    elif sys.argv[1] == 'runGrid':
        import shlex
        import subprocess

        nmpi, totproc, nrunsPerProc, baseseed = map(int, sys.argv[2:])
        hdfPrefix = 'data/vesreac_error/vesreac_error'
        mesh_ntets = [291, 577, 991, 2088, 3414, 11773, 41643, 265307]
        DVals = [0, 0.1]
        Ntotal_runs = 1000

        nsplit_runs = math.ceil(Ntotal_runs / nrunsPerProc)
        # Scheduling
        processes = []
        for i, (ntets, D, rind) in enumerate(itertools.product(mesh_ntets, DVals, range(nsplit_runs))):
            hdfPath = hdfPrefix + f'_{i}'
            if (rind + 1) * nrunsPerProc <= Ntotal_runs:
                NITER = nrunsPerProc
            else:
                NITER = Ntotal_runs % nrunsPerProc
            seed = hash((baseseed, ntets, D, rind, i)) % int(1e10)
            command = f'mpirun --bind-to none -n {nmpi} python3 {__file__} runSingle {hdfPath} {ntets} {D} {NITER} {D == 0} {True} {seed}'
            print('Running', i, '/', len(mesh_ntets)
                  * len(DVals) * nsplit_runs)
            processes.append((i, subprocess.Popen(shlex.split(
                command), env=os.environ, shell=False, stdout=subprocess.DEVNULL)))
            while len(processes) >= totproc // nmpi:
                remainingProcs = []
                for j, proc in processes:
                    if proc.poll() is None:
                        remainingProcs.append((j, proc))
                    else:
                        print('Run', j, 'finished')
                    processes = remainingProcs
                    time.sleep(0.5)
        # Wait until all runs are finished
        for j, p in processes:
            p.wait()
            print('Run', j, 'finished')
