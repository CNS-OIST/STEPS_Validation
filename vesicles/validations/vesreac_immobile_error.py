import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *
from steps.simcheck import *

import numpy as np
import os
import time
import sys
import itertools

scale = 1e-6
ves_diam = 1e-12

DT = 0.01
NITER = 20
INT = 0.1

params = dict(
    KCST = 5e8,
    CONC_INIT = 1e-6,
    NB_A = 100,
)
globals().update(params)

AVOGADRO = 6.022e23

def run_sim(hdfPath, meshPath, DCST, specDcst, vesDt, nr):
    model = Model()
    r = ReactionManager()

    with model:
        SA, SB = Species.Create()
        vsys = VolumeSystem.Create()
        vssys = VesicleSurfaceSystem.Create()
        ves = Vesicle.Create(ves_diam, DCST, vssys)
        with vssys:
            SB.o + SA.v > r[1] > SA.v
            r[1].K = KCST
        if specDcst > 0:
            with vsys:
                Diffusion(SB, specDcst)

    mesh = TetMesh.LoadAbaqus(meshPath, scale)
    with mesh:
        comp = Compartment.Create(mesh.tets, vsys)
    voltet = comp.Vol / len(mesh.tets)
    volfact = AVOGADRO * comp.Vol * 1e3

    rng = RNG('mt19937', 512, hash((meshPath, DCST, vesDt, nr)) % int(1e10))
    sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

    rs = ResultSelector(sim)

    concs = rs.comp.SB.Conc

    sim.toSave(concs, dt=DT)

    with HDF5Handler(hdfPath) as hdf:
        sim.toDB(hdf, f'sim_{vesDt}_{DCST}_{len(mesh.tets)}', 
            vesDt=vesDt, volfact=volfact, voltet=voltet, ntets=len(mesh.tets), DCST=DCST, **params
        )
        sim.newRun()

        sim.setVesicleDT(vesDt)
        sim.comp.ves.Count = NB_A
        sim.comp.VESICLES()('surf').SA.Count = 1

        sim.comp.SB.Conc = CONC_INIT

        sim.run(INT)


if __name__ == '__main__':
    if sys.argv[1] == 'runSingle':
        hdfPath, meshPath, DCST, specDcst, vesDt, nr = sys.argv[2:]
        run_sim(hdfPath, meshPath, float(DCST), float(specDcst), float(vesDt), nr)
    elif sys.argv[1] == 'runGrid':
        import shlex, subprocess

        hdfPrefix = 'data/vesreac_immobile_reactants/vesreac_immobile_reactants'
        specDcst = 0
        nmpi, totproc = map(int, sys.argv[2:4])
        if len(sys.argv) > 4:
            hdfPrefix = sys.argv[4]
        if len(sys.argv) > 5:
            specDcst = float(sys.argv[5])

        meshDir = 'meshes'
        meshes = [
            'sphere_2D_11773tets.inp', 'sphere_2D_265307tets.inp', 'sphere_2D_291tets.inp',
            'sphere_2D_3414tets.inp', 'sphere_2D_368452tets.inp', 'sphere_2D_41643tets.inp',
            'sphere_2D_991tets.inp'
        ]
        allMeshes = [os.path.join(meshDir, mesh) for mesh in meshes]

        # Parameters
        DCSTVals = [1e-13]
        vesDtVals = np.logspace(-5, -2, 20)

        # Scheduling
        processes = []
        for i, (meshPath, DCST, vesDt, nr) in enumerate(itertools.product(allMeshes, DCSTVals, vesDtVals, range(NITER))):
            hdfPath = hdfPrefix + f'_{i}'
            command = f'mpirun --bind-to none -n {nmpi} python3 {__file__} runSingle {hdfPath} {meshPath} {DCST} {specDcst} {vesDt} {nr}'
            print('Running', i, '/', len(allMeshes) * len(DCSTVals) * len(vesDtVals) * NITER)
            processes.append((i ,subprocess.Popen(shlex.split(command), env=os.environ, shell=False, stdout=subprocess.DEVNULL)))
            while len(processes) >= totproc // nmpi:
                remainingProcs = []
                for j, proc in processes:
                    if proc.poll() is None:
                        remainingProcs.append((j, proc))
                    else:
                        print('Run', j, 'finished')
                processes = remainingProcs
                time.sleep(0.5)
