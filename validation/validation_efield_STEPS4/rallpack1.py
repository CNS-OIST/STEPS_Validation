# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# # Tests adapted from rallpack 1
# # Original Rallpack 1 author: Iain Hepburn
# # Test suite author: Alessandro Cattabiani
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import print_function, absolute_import
import steps.interface

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.sim import *
from steps.saving import *

import math
import os
import pandas as pd
import sys
import copy

postproc_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "postproc",
)
sys.path.append(postproc_path)
from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator

from .. import configuration


def run_sim(USE_STEPS_4):

    # # # # # # # # # # # # # # # # SIMULATION CONTROLS # # # # # # # # # # # # # #

    # Sim end time (seconds)
    SIM_END = 0.25

    # The current injection in amps
    Iinj = 0.1e-9

    EF_DT = 1e-6
    SAVE_DT = 5e-6

    SEED = 1

    # # # # # # # # # # # # # # # # PARAMETERS # # # # # # # # # # # # # #

    # Leak conductance, Siemens/m^2
    L_G = 0.25

    # Leak reveral potential, V
    leak_rev = -65.0e-3

    # Total leak conductance for ideal cylinder:
    surfarea_cyl = 1.0 * math.pi * 1000 * 1e-12
    L_G_tot = L_G * surfarea_cyl

    # Ohm.m
    Ra = 1.0

    # # # # # # # # # # # # # DATA COLLECTION # # # # # # # # # # # # # # # # # #

    # record potential at the two extremes along (z) axis
    POT_POS = [0.0, 1.0e-03]

    # Mesh geometry

    mesh_path = configuration.path(
        os.path.join(
            "validation_efield_STEPS4",
            "meshes",
            "axon_cube_L1000um_D866nm_1135tets.msh",
        )
    )
    mesh = DistMesh(mesh_path) if USE_STEPS_4 else TetMesh.LoadGmsh(mesh_path)

    with mesh:
        if USE_STEPS_4:
            __MESH__ = Compartment.Create()

            memb = Patch.Create(__MESH__, None, "ssys")
            z_min = Patch.Create(__MESH__, None, "ssys")
            z_max = Patch.Create(__MESH__, None, "ssys")
        else:
            __MESH__ = Compartment.Create(mesh.tets)

            memb = Patch.Create(mesh.triGroups[(0, "memb")], __MESH__, None, "ssys")
            z_min = Patch.Create(mesh.triGroups[(0, "z_min")], __MESH__, None, "ssys")
            z_max = Patch.Create(mesh.triGroups[(0, "z_max")], __MESH__, None, "ssys")

        surfarea_mesh = memb.Area
        corr_fac_area = surfarea_mesh / surfarea_cyl

        vol_cyl = math.pi * 0.5 * 0.5 * 1000 * 1e-18
        vol_mesh = __MESH__.Vol
        corr_fac_vol = vol_mesh / vol_cyl

        if USE_STEPS_4:
            membrane = Membrane.Create([memb], capacitance=0.01 / corr_fac_area)
            __MESH__.Conductivity = 1 / (Ra * corr_fac_vol)
        else:
            membrane = Membrane.Create([memb])

        # The tetrahedrons from which to record potential
        POT_TET = TetList(mesh.tets[0, 0, z] for z in POT_POS)

    # Model # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    mdl = Model()
    r = ReactionManager()
    with mdl:
        ssys = SurfaceSystem.Create()

        # Leak
        leaksus = SubUnitState.Create()
        Leak = Channel.Create([leaksus])

        with ssys:
            # Set the single-channel conductance:
            g_leak_sc = L_G_tot / len(membrane.tris)
            OC_L = OhmicCurr.Create(Leak[leaksus], g_leak_sc, leak_rev)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Create the solver objects
    rng = RNG("r123", 512, SEED)
    # rng = RNG('mt19937', 512, SEED)

    if USE_STEPS_4:
        sim = Simulation(
            "DistTetOpSplit",
            mdl,
            mesh,
            rng,
            searchMethod=NextEventSearchMethod.GIBSON_BRUCK,
        )  # , searchMethod=NextEventSearchMethod.GIBSON_BRUCK)
    else:
        part = LinearMeshPartition(mesh, 1, 1, MPI.nhosts)
        sim = Simulation("TetOpSplit", mdl, mesh, rng, MPI.EF_DV_PETSC, part)

    # Data saving
    rs = ResultSelector(sim)

    Vrs = rs.TETS(POT_TET).V

    sim.toSave(Vrs, dt=SAVE_DT)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    sim.newRun()

    sim.TRIS(membrane.tris).Leak[leaksus].Count = 1

    sim.membrane.Potential = -65e-3

    minzverts = list(set([v for t in z_min.tris for v in t.verts]))
    for v in minzverts:
        sim.solver.setVertIClamp(v.idx, Iinj / len(minzverts))

    if not USE_STEPS_4:
        sim.membrane.Capac = 0.01 / corr_fac_area
        sim.membrane.VolRes = Ra * corr_fac_vol

    sim.EfieldDT = EF_DT

    progress_dt = SAVE_DT * 10
    nsteps = math.floor(SIM_END / progress_dt) + 1

    for i in range(nsteps):
        current_time = i * progress_dt
        sim.run(current_time)
        print(
            f"Progress: {round(1e4 * (i / nsteps)) / 1e2}%, current time/SIM_END: "
            f"{round(1e4*current_time)/1e4}/{SIM_END}"
        )

    """Record"""
    folder_path = configuration.path(
        os.path.join(
            "validation_efield_STEPS4", "results", f"STEPS{4 if USE_STEPS_4 else 3}"
        )
    )
    os.makedirs(folder_path, exist_ok=True)
    df = pd.DataFrame(
        {"t": Vrs.time[0], "V zmin": Vrs.data[0, :, 0], "V zmax": Vrs.data[0, :, 1]}
    )

    df.to_csv(
        folder_path + f"/res{SEED}_STEPS{4 if USE_STEPS_4 else 3}.txt",
        sep=" ",
        index=False,
    )

    return df


def check_results(db_STEPS3=None, db_STEPS4=None):
    """Benchmark_analytic"""
    multi = 1000
    traces_benchmark_analytic = []
    traces_benchmark_analytic.append(Trace("t", "ms", multi=multi))
    traces_benchmark_analytic.append(
        Trace(
            "V zmin",
            "mV",
            multi=multi,
            reduce_ops={
                "amin": [],
                "amax": [],
            },
        )
    )
    traces_benchmark_analytic.append(copy.deepcopy(traces_benchmark_analytic[-1]))
    traces_benchmark_analytic[-1].name = "V zmax"
    benchmark_analytic = TraceDB(
        "analytic",
        traces_benchmark_analytic,
        folder_path=configuration.path(
            os.path.join("validation_efield_STEPS4", "data", "analytical")
        ),
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """Sample_STEPS3"""
    multi = 1000
    traces_sample_STEPS3 = []
    traces_sample_STEPS3.append(Trace("t", "ms", multi=multi))

    traces_sample_STEPS3.append(
        Trace(
            "V zmin",
            "mV",
            multi=multi,
            reduce_ops={
                "amin": [],
                "amax": [],
            },
        )
    )
    traces_sample_STEPS3.append(copy.deepcopy(traces_sample_STEPS3[-1]))
    traces_sample_STEPS3[-1].name = "V zmax"
    if db_STEPS3 is not None:
        for i in traces_sample_STEPS3:
            i.add_raw_trace("", db_STEPS3[i.name])

    """Create the sample database"""
    sample_STEPS3 = TraceDB(
        "STEPS3",
        traces_sample_STEPS3,
        folder_path="",  # configuration.path(os.path.join('validation_efield_STEPS4', 'results', 'STEPS3')),
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
        save_raw_traces_cache=False,
        save_refined_traces_cache=False,
    )

    """Sample_STEPS4"""
    multi = 1000
    traces_sample_STEPS4 = []
    traces_sample_STEPS4.append(Trace("t", "ms", multi=multi))

    traces_sample_STEPS4.append(
        Trace(
            "V zmin",
            "mV",
            multi=multi,
            reduce_ops={
                "amin": [],
                "amax": [],
            },
        )
    )
    traces_sample_STEPS4.append(copy.deepcopy(traces_sample_STEPS4[-1]))
    traces_sample_STEPS4[-1].name = "V zmax"
    if db_STEPS4 is not None:
        for i in traces_sample_STEPS4:
            i.add_raw_trace("", db_STEPS4[i.name])

    """Create the sample database"""
    sample_STEPS4 = TraceDB(
        "STEPS4",
        traces_sample_STEPS4,
        folder_path="",  # configuration.path(os.path.join('validation_efield_STEPS4', 'results', 'STEPS4')),
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
        save_raw_traces_cache=False,
        save_refined_traces_cache=False,
    )

    """comparator"""

    comp = Comparator(traceDBs=[benchmark_analytic, sample_STEPS3, sample_STEPS4])

    for i in ["V zmin", "V zmax"]:
        comp.plot(
            trace_name_b=i,
            savefig_path=configuration.path(
                os.path.join("validation_efield_STEPS4", "results", "pics")
            ),
            isdiff=False,
            suffix="",
            interactive=False,
        )

    """Compute the mse"""
    for tDBnames, mse_tests in comp.mse_refactored(normalized=False).items():
        print(tDBnames)
        for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
            # the std. mesh is quite coarse and the error with the analytic solution may be considered still big.
            # However, we are also comparing with STEPS 3 where we can be much more strict.
            err = 1e-1 if "analytic" in tDBnames else 1e-12
            print(k, *v.items(), f" Max accepted err: {err}")

            assert v["amax"] < err


def test_rallpack1():
    db_STEPS3 = run_sim(USE_STEPS_4=False)
    db_STEPS4 = run_sim(USE_STEPS_4=True)
    check_results(db_STEPS3, db_STEPS4)
