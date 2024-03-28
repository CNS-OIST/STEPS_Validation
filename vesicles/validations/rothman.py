# The mossy fibre terminals model of Rothman et al 2016 (https://doi.org/10.7554/eLife.15133)
# Includes static mitochondria in the mesh, immobile and mobile vesicles in the model.

import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

from matplotlib import pyplot as plt
import numpy as np
import time

########################################################################

# plotting dt; sim endtime:
DT = 0.0002
T_END = 0.3

########################################################################

mito_vol_frac = 0.28
vesicle_vol_frac = 0.17  #  Rothman et al MFT
vesicle_immob_frac = 0.25  #  Rothman et al MFT
vesicle_diam = 49e-9  #  Rothman et al MFT
DCST = 0.06e-12

########################################################################

MESHFILE = 'MFT_wmito0.28_cylinder_565958tets_19042022.inp'
scale = 1

########################################################################

model = Model()

with model:
    vsys = VolumeSystem.Create()

    Ves1 = Vesicle.Create(vesicle_diam, DCST)
    Ves_immob = Vesicle.Create(vesicle_diam, 0)
    X = Species.Create()

    with vsys:
        Diffusion(X, 0)

########################################################################

mesh = TetMesh.LoadAbaqus('meshes/' + MESHFILE, scale * 1e-6)

with mesh:
    cyto = Compartment.Create(mesh.tets, vsys)

########################################################################

rng = RNG('mt19937', 1024, int(time.time() % 4294967295))
sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

tpnts = np.arange(DT, T_END, DT)

# Calculate the number based on volume fractions
vesicle_vol_total = mesh.Vol * vesicle_vol_frac

vesicle_n_immob = (vesicle_immob_frac * vesicle_vol_total) / (
    (4.0 / 3) * np.pi * pow((vesicle_diam / 2.0), 3))
vesicle_n_mob = ((1 - vesicle_immob_frac) * vesicle_vol_total) / (
    (4.0 / 3) * np.pi * pow((vesicle_diam / 2.0), 3))

vesicle_n_mob = int(vesicle_n_mob)
vesicle_n_immob = int(vesicle_n_immob)

sim.setVesicleDT(0.0001)

if MPI.rank == 0:
    print("Set mobile vesicle number to: ", vesicle_n_mob)
    btime = time.time()

sim.cyto.Ves1.Count = vesicle_n_mob
vesicle_n_mob_count = sim.cyto.Ves1.Count

if MPI.rank == 0:
    print("Succesfully set mobile vesicle number to: ", vesicle_n_mob_count)
    print("Set immobile vesicle number to: ", vesicle_n_immob)

sim.cyto.Ves_immob.Count = vesicle_n_immob
vesicle_n_immob_count = sim.cyto.Ves_immob.Count

if MPI.rank == 0:
    print("Succesfully set immobile vesicle number to: ", vesicle_n_immob_count)

# Only record from vesicles within a 1uM radius of centre, to negate boundary effects
allVes = VesicleList(sim.cyto.Ves1)
recordVes = VesicleList(
    [ves for ves in allVes if np.linalg.norm(ves.Pos) < 1e-6])
starting_pos = np.array(sim.VESICLES(recordVes).Pos)

if MPI.rank == 0:
    print("Record vesicle number: ", len(recordVes))

rs = ResultSelector(sim)

positions = rs.VESICLES(recordVes).Pos

sim.toSave(positions, dt=DT)

with HDF5Handler('data/rothman') as hdf:
    group = sim.toDB(hdf, f'rothman', starting_pos=starting_pos,
                     mito_vol_frac=mito_vol_frac)
    if MPI.rank == 0:
        group.staticData['tpnts'] = list(tpnts)

    sim.newRun(reset=False)

    for t in tpnts:
        if MPI.rank == 0:
            print(t, tpnts[-1], time.time() - btime)
        sim.run(t)
