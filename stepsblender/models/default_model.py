import steps.interface

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.sim import *
from steps.utils import *
from steps.saving import *

import random
import math
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(FILE_PATH, '..', 'meshes', 'sphere_4ktets.msh')
DATA_PATH = os.path.join(FILE_PATH, '..', 'data', os.path.basename(__file__).split('.')[0])

mdl = Model()
r = ReactionManager()
DCST = 1e-12

with mdl:
    S1, S2, S3 = Species.Create()
    vsys = VolumeSystem.Create()
    L1, L2 = LinkSpecies.Create()

    vssys = VesicleSurfaceSystem.Create()
    V1 = Vesicle.Create(4e-7, 1e-12)
    V1.addSystem(vssys)

    V2 = Vesicle.Create(4e-7, 1e-12)
    V2.addSystem(vssys)

    rssys = RaftSurfaceSystem.Create()

    R1 = Raft.Create(4e-7, 1e-12, rssys)
    ssys = SurfaceSystem.Create()

    with vsys:
        S1 > r[1] > S2
        r[1].K = 1

        ves1bind1 = VesicleBind.Create((V1, V1), (S1, S1), (L1, L1), 0, 1e-7, kcst=1e10)

        ves2unbind1 = VesicleUnbind.Create((V1, V1), (L1, L1), (S2, S2), kcst=0.0001)

        ves1bind2 = VesicleBind.Create((V1, V1), (S2, S2), (L2, L2), 0, 1e-7, kcst=1e10)

        ves1bind3 = VesicleBind.Create((V1, V1), (S1, S2), (L1, L2), 0, 1e-7, kcst=1e10)

        Diffusion(S1, DCST)
        Diffusion(S2, DCST)
        Diffusion(S3, DCST)

    with vssys:

        exo1 = Exocytosis.Create(1, deps=S1, raft=R1)
        exo2 = Exocytosis.Create(2, deps=S1, raft=R1)

        S1.o + S2.v > r[1] > S1.i + S3.v
        r[1].K = Parameter(10, 'uM^-1 s^-1')

    with ssys:
        S1.s > r[1] > S1.s
        r[1].K = 1

        Diffusion(S1, DCST)
        Diffusion(S2, DCST)
        Diffusion(S3, DCST)

        endo1 = Endocytosis(V1, 1)

    with rssys:
        rendo1 = RaftEndocytosis.Create(V1, 2)

mesh = TetMesh.LoadGmsh(MESH_PATH, scale=5e-6)

with mesh:
    middle = (mesh.bbox.max - mesh.bbox.min) / 2 + mesh.bbox.min

    comp1 = Compartment.Create([tet for tet in mesh.tets if tet.center.x <= middle.x], vsys)
    comp2 = Compartment.Create(mesh.tets - comp1.tets, vsys)

    memb1 = Patch.Create(comp1.surface & mesh.surface, comp1, None)
    memb2 = Patch.Create(comp2.surface & mesh.surface, comp2, None)
    patchInter = Patch.Create(comp1.surface & comp2.surface, comp1, comp2, ssys)

    with memb2:
        tets = memb2.tris[0].tetNeighbs[0].toList()
        tets.dilate(3)
        tris = tets.surface & memb2.tris
        endozone1 = EndocyticZone.Create(tris)

rng = RNG('mt19937', 512, 1234)

sim = Simulation('TetVesicle', mdl, mesh, rng, False)

path = sim.addVesiclePath('path1')
points = [path.addPoint(tet.center) for tet in comp1.tets[0:40:10]]
for i, orig in enumerate(points):
    if i + 1 < len(points):
        path.addBranch(orig, {points[i + 1]: 0.5})

rs = ResultSelector(sim)

############################################

from stepsblender.utils import AddBlenderDataSaving

AddBlenderDataSaving(sim, dt=0.05)

############################################

with XDMFHandler(DATA_PATH) as hdf:
    sim.toDB(hdf, 'default')

    sim.newRun()

    sim.comp1.V1.Count = 50
    sim.comp1.V2.Count = 5
    sim.patchInter.S1.Count = 20

    lst = VesicleList(sim.comp1.V1)

    sim.VESICLES(lst('surf')).S1.Count = 50
    sim.VESICLES(lst('in')).S1.Count = 60

    sim.comp1.VESICLES()('surf').S1.Count = 20
    sim.comp1.VESICLES()('surf').S2.Count = 10
    sim.comp1.VESICLES()('in').S2.Count = 10

    sim.comp1.VESICLES(V2)('surf').LIST(S1, S2, S3).Count = 0
    sim.comp1.VESICLES(V2)('in').LIST(S1, S2, S3).Count = 0

    sim.patchInter.R1.Count = 10
    lstraft = RaftList(sim.patchInter.R1)
    sim.RAFTS(lstraft).S1.Count = 2

    sim.patchInter.RAFTS().S3.Count = 5

    for t in range(10):
        if MPI.rank == 0:
            print(f'running {t / 10} / 1s')
        sim.run(t / 10)
