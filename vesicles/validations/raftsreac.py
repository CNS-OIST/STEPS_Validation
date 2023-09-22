########################################################################

# Test 4 different types of raft surface reaction: 2 first-order
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

########################################################################

# Simulation parameters
scale = 1e-6

########################################################################

# Raft-related parameters
raft_N = 10
raft_diam = 10e-8

########################################################################

# First order irreversible parameters
NITER_foi = 500
KCST_foi = 5
spec_A_foi_number_perraft = 5
spec_A_foi_N = spec_A_foi_number_perraft * raft_N

# First order reversible parameters
NITER_for = 10
KCST_f_for = 100.0
KCST_b_for = 20.0
spec_A_for_number_perraft = 1000
spec_A_for_N = spec_A_for_number_perraft * raft_N

# Second order irreversible AA parameters
NITER_soAA = 500
KCST_soAA = 2e6
spec_A_soAA_number_perraft = 100
spec_B_soAA_number_incomp = raft_N * spec_A_soAA_number_perraft

# Second order irreversible AB parameters
NITER_soAB = 500
KCST_soAB = 1.0e6
n_soAB = 2
spec_A_soAB_number_perraft = 50
spec_B_soAB_number_incomp = raft_N * spec_A_soAB_number_perraft / n_soAB

########################################################################

NITER_max = max([NITER_foi, NITER_for, NITER_soAA, NITER_soAB])

INT = 0.21
DT = 0.01

AVOGADRO = 6.022e23
LINEWIDTH = 3

########################################################################

model = Model()
r = ReactionManager()

with model:
    vsys = VolumeSystem.Create()
    rssys = RaftSurfaceSystem.Create()

    raft = Raft.Create(raft_diam, 1e-12, rssys)
    A_foi, A_for, B_for, A_soAA, B_soAA, C_soAA, A_soAB, B_soAB, C_soAB = Species.Create(
    )

    with rssys:
        # First order irreversible
        A_foi.r >r[1]> None
        r[1].K = KCST_foi

        # First order reversible
        A_for.r <r[1]> B_for.r
        r[1].K = KCST_f_for, KCST_b_for

        # Second order irreversible AA
        B_soAA.i + A_soAA.r >r[1]> C_soAA.r
        r[1].K = KCST_soAA

        # Second order irreversible AB
        B_soAB.i + A_soAB.r >r[1]> C_soAB.r
        r[1].K = KCST_soAB

    with vsys:
        Diffusion(B_soAA, 10e-12 * 5)
        Diffusion(B_soAB, 10e-12 * 5)

########################################################################

mesh = TetMesh.LoadAbaqus('meshes/sphere_0.5D_577tets.inp', scale)

with mesh:
    comp = Compartment.Create(mesh.tets, vsys)
    memb = Patch.Create(mesh.surface, comp, None)

########################################################################

rng = RNG('mt19937', 512, 100)
sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

CONCA_soAA = (raft_N * spec_A_soAA_number_perraft) / (AVOGADRO * comp.Vol *
                                                      1e3)
CONCB_soAA = spec_B_soAA_number_incomp / (AVOGADRO * comp.Vol * 1e3)

CONCA_soAB = (raft_N * spec_A_soAB_number_perraft) / (AVOGADRO * comp.Vol *
                                                      1e3)
CONCB_soAB = CONCA_soAB / n_soAB

rs = ResultSelector(sim)

volfact = AVOGADRO * comp.Vol * 1e3
rs_foi = rs.memb.raft.A_foi.Count
rs_for = rs.memb.raft.LIST(A_for, B_for).Count / volfact
rs_soAA = rs.memb.raft.A_soAA.Count / volfact << rs.comp.B_soAA.Conc << rs.memb.raft.C_soAA.Count / volfact
rs_soAB = rs.memb.raft.A_soAB.Count / volfact << rs.comp.B_soAB.Conc

sim.toSave(rs_foi, rs_for, rs_soAA, rs_soAB, dt=DT)

for i in range(NITER_max):
    if MPI.rank == 0:
        print(i, 'of', NITER_max)

    sim.newRun()

    sim.memb.raft.Count = raft_N

    if i < NITER_foi:
        sim.memb.RAFTS().A_foi.Count = spec_A_foi_number_perraft
    if i < NITER_for:
        sim.memb.RAFTS().A_for.Count = spec_A_for_number_perraft
        sim.memb.RAFTS().B_for.Count = 0
    if i < NITER_soAA:
        sim.comp.B_soAA.Count = spec_B_soAA_number_incomp
        sim.memb.RAFTS().A_soAA.Count = spec_A_soAA_number_perraft
    if i < NITER_soAB:
        sim.comp.B_soAB.Count = spec_B_soAB_number_incomp
        sim.memb.RAFTS().A_soAB.Count = spec_A_soAB_number_perraft

    sim.run(INT)

if MPI.rank == 0:
    tpnts = rs_for.time[0]

    plt.subplot(221)
    mean_res_foi = np.mean(rs_foi.data[:NITER_foi, ...], axis=0).flatten()
    std_res_foi = np.std(rs_foi.data[:NITER_foi, ...], axis=0).flatten()

    analy = spec_A_foi_N * np.exp(-KCST_foi * tpnts)
    std = np.sqrt((spec_A_foi_N * (np.exp(-KCST_foi * tpnts)) *
                   (1 - (np.exp(-KCST_foi * tpnts)))))

    plt.errorbar(tpnts,
                 analy,
                 std,
                 color='black',
                 label='analytical',
                 linewidth=LINEWIDTH)
    plt.errorbar(tpnts + DT / 5.0,
                 mean_res_foi,
                 std_res_foi,
                 color='cyan',
                 ls='--',
                 label='STEPS',
                 linewidth=LINEWIDTH)
    plt.legend()
    plt.ylabel('Molecule count')
    plt.xlabel('Time (s)')

    plt.subplot(222)
    mean_res_for = np.mean(rs_for.data[:NITER_for, ...], axis=0)

    eq_A = spec_A_for_N * (KCST_b_for / KCST_f_for) / (
        1 + (KCST_b_for / KCST_f_for)) / (comp.Vol * 6.0221415e26) * 1e6
    eq_B = (spec_A_for_N / (comp.Vol * 6.0221415e26)) * 1e6 - eq_A

    plt.plot((tpnts[0], tpnts[-1]), (eq_A, eq_A),
             'k-',
             label='analytical Aeq',
             linewidth=LINEWIDTH)
    plt.plot((tpnts[0], tpnts[-1]), (eq_B, eq_B),
             'b-',
             label='analytical Beq',
             linewidth=LINEWIDTH)
    plt.plot(tpnts,
             mean_res_for[:, 0] * 1e6,
             '--',
             label='STEPS A',
             linewidth=LINEWIDTH)
    plt.plot(tpnts,
             mean_res_for[:, 1] * 1e6,
             '--',
             label='STEPS B',
             linewidth=LINEWIDTH)
    plt.legend()
    plt.ylabel('Concentration ($\mu$M)')
    plt.xlabel('Time (s)')

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
             label='STEPS, raft species ')
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
    fig.savefig("plots/raftsreac.pdf", dpi=300, bbox_inches='tight')
    plt.close()
