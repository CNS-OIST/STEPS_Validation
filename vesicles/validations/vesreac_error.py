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

from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.optimize import curve_fit

########################################################################

def run_sim(ntets, D, NITER_foi, NITER_soAA):

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

    NITER_max = max([NITER_foi, NITER_soAA])
    
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
            A_foi.v >r[1]> None
            r[1].K = KCST_foi

            # Second order irreversible AA
            B_soAA.o + A_soAA.v >r[1]> C_soAA.v
            r[1].K = KCST_soAA

        with vsys:
            Diffusion(B_soAA, 10e-12)

    scale, diam = (1e-6, '0.5D') if ntets in [577, 2088] else (0.25e-6, '2D')
    mesh = TetMesh.LoadAbaqus('meshes/sphere_'+diam+'_'+str(ntets)+'tets.inp', scale)

    with mesh:
        comp = Compartment.Create(mesh.tets, vsys)
        memb = Patch.Create(mesh.surface, comp, None)


    rng = RNG('mt19937', 512, 123)
    sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

    CONCA_soAA = (ves_N * spec_A_soAA_number_perves) / (AVOGADRO * comp.Vol * 1e3)
    CONCB_soAA = spec_B_soAA_number_incomp / (AVOGADRO * comp.Vol * 1e3)

    rs = ResultSelector(sim)

    volfact = AVOGADRO * comp.Vol * 1e3
    
    rs_foi = rs.comp.ves('surf').A_foi.Count
    rs_soAA = rs.comp.ves(
        'surf').A_soAA.Count / volfact << rs.comp.B_soAA.Conc << rs.comp.ves(
            'surf').C_soAA.Count / volfact
    
    sim.toSave(rs_foi, rs_soAA, dt=DT)

    btime=time.time()

    for i in range(NITER_max):
        if MPI.rank == 0 and not i%100:
            print(ntets, D, ':', i, 'of', NITER_max)

        sim.newRun()

        sim.comp.ves.Count = ves_N

        if i < NITER_foi:
            sim.comp.VESICLES()('surf').A_foi.Count = spec_A_foi_number_perves

        if i < NITER_soAA:
            sim.comp.B_soAA.Count = spec_B_soAA_number_incomp
            sim.comp.VESICLES()('surf').A_soAA.Count = spec_A_soAA_number_perves
        
        sim.run(INT)
        
        mean_res_foi = np.mean(rs_foi.data[:NITER_foi, ...], axis=0).flatten()
        mean_res_soAA = np.mean(rs_soAA.data[:NITER_soAA, ...], axis=0)
        
        def foi_findrate(x, K): return spec_A_foi_N * np.exp(-K * x)
        def soAA_findrate(x, K): return (1.0 / CONCA_soAA + ((x * K))) * 1e-6
        
        poptfoi, pcovfoi = curve_fit(foi_findrate, rs_foi.time[0], mean_res_foi, p0=KCST_foi)
        poptAAv, pcovAAv = curve_fit(soAA_findrate, rs_soAA.time[0],(1.0 / mean_res_soAA[:, 0]) * 1e-6, p0=KCST_soAA)
        
        fo_error = 100*abs(poptfoi[0]-KCST_foi)/KCST_foi
        so_error = 100*abs(poptAAv[0]-KCST_soAA)/KCST_soAA

    return fo_error, so_error


mesh_ntets = [291,577,991,2088,3414,11773,41643,265307]
mesh_sizetets = [119.77,96.58,81.19,63.76,54.21,36.03,23.68,12.79]


foi_error = []
so_error_D0 = []
so_error_D0_1 = []

for ntets in mesh_ntets:
    error = run_sim(ntets, 0, 1000, 1000)
    foi_error.append(error[0])
    so_error_D0.append(error[1])

for ntets in mesh_ntets:
    error = run_sim(ntets, 0.1, 1, 1000)
    so_error_D0_1.append(error[1])

lw=3

plt.plot(mesh_sizetets, foi_error, linewidth=lw, marker ='o')
plt.xlim(0, 130)
plt.gca().invert_xaxis()
plt.ylim(0,2)
plt.xlabel("Average tetrahedron size (nm)")
plt.ylabel("Error in 2nd order reaction rate (%)")
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/vesreac_error_size_foi.pdf", dpi=300, bbox_inches='tight')
plt.close()

plt.plot(mesh_sizetets, so_error_D0, label='D=0$\mu m^2s^{-1}$', linewidth=lw, marker ='o')
plt.plot(mesh_sizetets, so_error_D0_1, label='D=0.1$\mu m^2s^{-1}$', linewidth=lw, marker ='o')
plt.legend()
plt.xlim(0, 130)
plt.gca().invert_xaxis()
plt.ylim(0,15)
plt.xlabel("Average tetrahedron size (nm)")
plt.ylabel("Error in 2nd order reaction rate (%)")
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/vesreac_error_size.pdf", dpi=300, bbox_inches='tight')
plt.close()

