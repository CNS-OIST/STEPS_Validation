########################################################################
# Test the distribution of molecules under reduced volumes by vesicles
########################################################################

import steps.interface

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.sim import *
from steps.saving import *
import steps.simcheck

import matplotlib
from matplotlib import pyplot as plt
import time
import os
import numpy as np
from scipy.stats import binom
from scipy.optimize import curve_fit

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

########################################################################

INT = 1000.001
NMOLCS = 1000
DT = 0.001

########################################################################

ltime = time.localtime()
resdir = f'plots/reducedvol_{ltime[0]}_{ltime[1]}_{ltime[2]}'
if MPI.rank == 0:
    try:
        os.mkdir(resdir)
    except:
        pass

########################################################################


def binom_func(x, P):
    rv = binom(NMOLCS, P)
    return rv.pmf(x)


def runtest():

    vesicle_vol_frac = 0.1
    vesicle_diam = 50e-9

    DCST_ves = 1e-12
    DCST_spec = 10e-12

    MESHFILE = 'sphere_10r_49tets'

    model = Model()
    with model:
        spec1 = Species.Create()
        ves1 = Vesicle.Create(vesicle_diam, DCST_ves)
        vsys = VolumeSystem.Create()
        with vsys:
            spec1_vdiff = Diffusion.Create(spec1, DCST_spec)

    mesh = TetMesh.Load(os.path.join('meshes', MESHFILE))
    mesh_vol = mesh.Vol
    with mesh:
        comp1 = Compartment.Create(mesh.tets, vsys)

    rng = RNG('mt19937', 512, int(time.time() % 4294967295))

    sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)

    rs = ResultSelector(sim)
    spec_count = rs.TETS().spec1.Count
    tet_volred = rs.TETS().ReducedVol
    sim.toSave(spec_count, tet_volred, dt=DT)

    with HDF5Handler('data/reducedvol') as hdf:
        sim.toDB(hdf, f'reducedvol')
        sim.newRun()

        tetvolinit = sim.TETS().Vol
        
        sim.setVesicleDT(5e-4)
        steps.simcheck.Check(sim, True)
        
        sim.comp1.spec1.Count = NMOLCS

        vesicle_vol_total = mesh_vol * vesicle_vol_frac
        vesicle_n = vesicle_vol_total / ((4.0 / 3) * np.pi * np.power(
            (vesicle_diam / 2.0), 3))
        sim.comp1.ves1.Count = int(vesicle_n)

        mesh_vol_reduced = sum(sim.TETS().ReducedVol)

        sim.run(INT)

    if MPI.rank == 0:
        with HDF5Handler('data/reducedvol') as hdf:
            spec_count, tet_volred = hdf['reducedvol'].results
            tetvolmean = np.mean(tet_volred.data[0], axis=0)
            plt.bar(range(len(tetvolinit)),
                    tetvolinit,
                    label='raw volume',
                    color='black')
            plt.bar(range(len(tetvolmean)),
                    tetvolmean,
                    label='mean volume during simulation',
                    color='red')
            plt.xlabel('Tet index')
            plt.ylabel('Volume')
            plt.legend()
            plt.savefig(
                os.path.join(resdir, f'{MESHFILE}_{NMOLCS}_{INT}_meanvolume.png'))
            plt.close()

            plt.bar(range(len(tetvolmean)),
                    tetvolinit,
                    label='raw volume',
                    color='black')
            plt.bar(range(len(tetvolmean)),
                    tetvolmean / tetvolinit,
                    label='fractional mean volume during simulation',
                    color='red')
            plt.xlabel('Tet index')
            plt.ylabel('Volume')
            plt.legend()
            plt.savefig(
                os.path.join(
                    resdir,
                    f'{MESHFILE}_{NMOLCS}_{INT}_meanvolume_fractional.png'))
            plt.close()

            maxn = max(np.array(spec_count.data).flatten())

            errors = []

            for t in range(len(mesh.tets)):
                n_range = [0, maxn]
                norm_hist = plt.hist(spec_count.data[0][:, t],
                                     bins=np.arange(
                                         n_range[0], n_range[1]) - 0.5,
                                     rwidth=0.5,
                                     align='mid',
                                     label='simulation',
                                     density=True,
                                     color='blue')

                avogadro = 6.022141e23

                v = tetvolmean[t]
                V = mesh_vol_reduced
                [n, prob] = [NMOLCS, v / V]

                x = np.arange(0, maxn)
                rv = binom(n, prob)

                popt, pcov = curve_fit(binom_func,
                                       np.arange(n_range[0], n_range[1])[:-1],
                                       norm_hist[0],
                                       p0=[prob])

                if MPI.rank == 0:
                    plt.bar(x,
                            rv.pmf(x),
                            width=0.5,
                            color='red',
                            label='binomial',
                            alpha=0.5)
                    rv_fit = binom(n, popt[0])
                    plt.plot(x,
                             rv_fit.pmf(x),
                             color='black',
                             label='Fit to simulation')
                    plt.xlim(int(n * prob * 0.1), int(n * prob * 2.0))
                    plt.legend(loc='best')
                    plt.xlabel('Number of molecules')
                    plt.ylabel('Probability')
                    plt.ylim(0, 0.14)
                    fig = plt.gcf()
                    fig.set_size_inches(3.4, 3.4)
                    plt.savefig(os.path.join(resdir,
                                             f'{MESHFILE}_{NMOLCS}_{INT}_{t}.pdf'),
                                dpi=300,
                                bbox_inches='tight')
                    plt.close()

                    errors.append(abs(100 * prob / popt[0] - 100))

        plt.hist(errors)
        plt.xlabel('Percentage error in binomial fit')
        plt.ylabel('Number of tetrahedrons')
        fig = plt.gcf()
        fig.set_size_inches(3.4, 3.4)
        plt.savefig(os.path.join(resdir,
                                 f'{MESHFILE}_{NMOLCS}_{INT}_error.pdf'),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()


btime = time.time()
runtest()
if MPI.rank == 0:
    print('Took ', time.time() - btime, 'seconds')
