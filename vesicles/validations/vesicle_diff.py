import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

from matplotlib import pyplot as plt
import mpi4py.MPI
import numpy as np
import time

# Number of iterations; plotting dt; sim endtime:
NITER = 10000

ves_diam = 50e-9

v_color = {0.0: 'yellow', 0.2: 'red', 0.4: 'blue', 0.6: 'green'}

adjuster = 0

# These are the excluded volumes
for vol_frac in [0.0, 0.2, 0.4, 0.6]:

    DCST = 1.0e-12

    DT = 5.0e-2

    T_END = 100.0e-2

    ########################################################################

    MESHFILE = 'sphere_rad10_11Ktets'

    ########################################################################

    model = Model()

    with model:
        vsys = VolumeSystem.Create()

        Ves1 = Vesicle.Create(ves_diam, DCST)
        X = Species.Create()

        with vsys:
            Diffusion(X, 0)

    ########################################################################

    mesh = TetMesh.Load(f'meshes/{MESHFILE}')

    with mesh:
        ctet = mesh.tets[0, 0, 0]
        comptets = mesh.tets - TetList([ctet])
        if MPI.rank == 0:
            vol = mesh.Vol
            totVol = vol
            while totVol > vol * (1 - vol_frac):
                tet = comptets[np.random.randint(0, len(comptets))]
                if np.linalg.norm(tet.center) < 10 * ves_diam:
                    print(np.linalg.norm(tet.center), 10 * ves_diam)
                    continue
                comptets.remove(tet)
                totVol -= tet.Vol

            print(
                f'Initial volume: {vol} Final volume: {totVol} Fraction: {totVol/vol}'
            )
        comptets = mpi4py.MPI.COMM_WORLD.bcast(comptets.indices, root=0)

        cyto = Compartment.Create(comptets + [ctet.idx], vsys)

    ########################################################################

    rng = RNG('r123', 1024, int(time.time() % 4294967295))
    sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

    rs = ResultSelector(sim)

    vesPos = rs.cyto.VESICLES(Ves1).Pos

    sim.toSave(vesPos, dt=DT)

    with HDF5Handler('data/vesicle_diff') as hdf:
        sim.toDB(hdf, f'vesicle_diff_volfrac{vol_frac}', vol_frac=vol_frac)
        for i in range(NITER):
            sim.newRun()

            sim.setVesicleDT(DT / 10.0)
            v = sim.cyto.addVesicle(Ves1)
            v.Pos = [0, 0, 0]

            if MPI.rank == 0:
                print(i + 1, 'of', NITER)

            sim.run(T_END)

if MPI.rank == 0:
    with HDF5Handler('data/vesicle_diff') as hdf:
        for vol_frac in sorted(hdf.parameters['vol_frac']):
            vesPos, = hdf.get(vol_frac=vol_frac).results

            data = np.array([[list(cols[0].values()) for cols in row]
                            for row in vesPos.data])
            res = np.mean(np.sum(data**2, axis=3) * 1e12, axis=0)
            plt.plot(vesPos.time[0],
                     res,
                     label=str(vol_frac),
                     color=v_color[vol_frac],
                     linewidth=3)
            plt.plot(vesPos.time[0],
                     vesPos.time[0] * 6 * DCST * 1e12,
                     'k--',
                     linewidth=3)
            adjuster += DT / 10.0

    plt.xlabel('Time (s)')
    plt.ylabel('<r$^2$> ($\mu$$m^2$)')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(3.4, 3.4)
    fig.savefig("plots/vesicle_diff.pdf", dpi=300, bbox_inches='tight')
    plt.close()
