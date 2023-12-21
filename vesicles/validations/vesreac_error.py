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

# Simulation parameters

mesh_ntets = 291
scale = 0.25e-6 # any '2D' mesh requires a scale of 0.25e-6

D=10 # Diffusion factor

########################################################################

# Vesicle-related parameters
ves_N = 10
ves_diam = 40e-9

########################################################################

# Second order irreversible AA parameters
NITER_soAA = 1000
KCST_soAA = 2e6
spec_A_soAA_number_perves = 100
spec_B_soAA_number_incomp = ves_N * spec_A_soAA_number_perves

# Second order irreversible AB parameters
NITER_soAB = 1000
KCST_soAB = 0.5e6
n_soAB = 2
spec_A_soAB_number_perves = 50
spec_B_soAB_number_incomp = ves_N * spec_A_soAB_number_perves / n_soAB

########################################################################

NITER_max = max([NITER_soAA, NITER_soAB])

INT = 0.21
DT = 0.01

AVOGADRO = 6.022e23
LINEWIDTH = 3

########################################################################

model = Model()
r = ReactionManager()

with model:
    vsys = VolumeSystem.Create()
    vssys = VesicleSurfaceSystem.Create()

    ves = Vesicle.Create(ves_diam, D*1e-12, vssys)
    A_soAA, B_soAA, C_soAA, A_soAB, B_soAB, C_soAB = Species.Create(
    )

    with vssys:

        # Second order irreversible AA
        B_soAA.o + A_soAA.v >r[1]> C_soAA.v
        r[1].K = KCST_soAA

        # Second order irreversible AB
        B_soAB.o + A_soAB.v >r[1]> C_soAB.v
        r[1].K = KCST_soAB

    with vsys:
        Diffusion(B_soAA, 10e-12)
        Diffusion(B_soAB, 10e-12)

########################################################################

mesh = TetMesh.LoadAbaqus('meshes/sphere_2D_'+str(mesh_ntets)+'tets.inp', scale)
if MPI.rank == 0:
    print ("Mesh volume (nm^3):", mesh.Vol*1e27)

with mesh:
    comp = Compartment.Create(mesh.tets, vsys)
    memb = Patch.Create(mesh.surface, comp, None)

########################################################################

rng = RNG('mt19937', 512, 1)
sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

CONCA_soAA = (ves_N * spec_A_soAA_number_perves) / (AVOGADRO * comp.Vol * 1e3)
CONCB_soAA = spec_B_soAA_number_incomp / (AVOGADRO * comp.Vol * 1e3)

CONCA_soAB = (ves_N * spec_A_soAB_number_perves) / (AVOGADRO * comp.Vol * 1e3)
CONCB_soAB = CONCA_soAB / n_soAB

rs = ResultSelector(sim)

volfact = AVOGADRO * comp.Vol * 1e3
rs_soAA = rs.comp.ves(
    'surf').A_soAA.Count / volfact << rs.comp.B_soAA.Conc << rs.comp.ves(
        'surf').C_soAA.Count / volfact
rs_soAB = rs.comp.ves('surf').A_soAB.Count / volfact << rs.comp.B_soAB.Conc

sim.toSave(rs_soAA, rs_soAB, dt=DT)

btime=time.time()

for i in range(NITER_max):
    if MPI.rank == 0:
        print ("So far", time.time()-btime, "s")
        print(i, 'of', NITER_max)
    
    sim.newRun()

    sim.comp.ves.Count = ves_N

    if i < NITER_soAA:
        sim.comp.B_soAA.Count = spec_B_soAA_number_incomp
        sim.comp.VESICLES()('surf').A_soAA.Count = spec_A_soAA_number_perves
    if i < NITER_soAB:
        sim.comp.B_soAB.Count = spec_B_soAB_number_incomp
        sim.comp.VESICLES()('surf').A_soAB.Count = spec_A_soAB_number_perves

    sim.run(INT)

if MPI.rank == 0:

    tpnts = rs_soAA.time[0]

    plt.subplot(223)
    mean_res_soAA = np.mean(rs_soAA.data[:NITER_soAA, ...], axis=0)

    invA = (1.0 / mean_res_soAA[:, 0]) * 1e-6
    invB = (1.0 / mean_res_soAA[:, 1]) * 1e-6
    lineA = (1.0 / CONCA_soAA + ((tpnts * KCST_soAA))) * 1e-6
    lineB = (1.0 / CONCB_soAA + ((tpnts * KCST_soAA))) * 1e-6

    plt.plot(tpnts, lineA, 'k-', linewidth=LINEWIDTH, label='analytical')
    plt.plot(tpnts,
             invA,
             'r--',
             linewidth=LINEWIDTH,
             ms=5,
             label='STEPS, vesicle species ')
    plt.plot(tpnts,
             invB,
             'yo',
             linewidth=LINEWIDTH,
             label='STEPS, cytosolic species')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Inverse concentration (1/$\mu$M)')

    plt.subplot(224)
    mean_res_soAB = np.mean(rs_soAB.data[:NITER_soAB, ...], axis=0)

    A_soAB = mean_res_soAB[:, 0]
    B_soAB = mean_res_soAB[:, 1]
    C_soAB = CONCA_soAB - CONCB_soAB
    lnBA_soAB = np.log(B_soAB / A_soAB)
    lineAB_soAB = np.log(CONCB_soAB / CONCA_soAB) - C_soAB * KCST_soAB * tpnts
    plt.plot(tpnts, lineAB_soAB, 'k-', linewidth=LINEWIDTH, label='analytical')
    plt.plot(tpnts, lnBA_soAB, 'r--', linewidth=LINEWIDTH, label='STEPS')
    plt.legend()
    plt.xlabel('Time (s)')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    fig.savefig("plots/vesreac_error_D"+str(D)+"_"+str(mesh_ntets)+".pdf", dpi=300, bbox_inches='tight')
    plt.close()


    tpnts = rs_soAA.time[0]

    def soAA_func(x, K):
        return (1.0 / CONCA_soAA + ((x * K))) * 1e-6
    
    poptAAv, pcovAAv = curve_fit(soAA_func, tpnts,(1.0 / mean_res_soAA[:, 0]) * 1e-6, p0=KCST_soAA)
    poptAAc, pcovAAc = curve_fit(soAA_func, tpnts,(1.0 / mean_res_soAA[:, 1]) * 1e-6, p0=KCST_soAA)
    

    tpnts = rs_soAB.time[0]

    def soAB_func(x, K):
        return np.log(CONCB_soAB / CONCA_soAB) - C_soAB * K * x
    
    poptAB, pcovAB = curve_fit(soAB_func, tpnts, lnBA_soAB , p0=KCST_soAB)


    print ("soAA error, vesicle species", "soAA error, cytosolic species", "soAB error (%):")
    print(100*abs(poptAAv[0]-KCST_soAA)/KCST_soAA, 100*abs(poptAAc[0]-KCST_soAA)/KCST_soAA, 100*abs(poptAB[0]-KCST_soAB)/KCST_soAB)
