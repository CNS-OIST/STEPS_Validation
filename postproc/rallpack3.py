from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy
import numpy

"""Create the sample traces

Tell the program what is inside the raw data and what alaiyses you want to perform for each trace
"""
multi = 1
traces_sample_STEPS4 = []
traces_sample_STEPS4.append(Trace("t", "s", multi=multi))

traces_sample_STEPS4.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_peak_y', {i}]": [] for i in range(9)},
            **{f"['i_peak_t', {i}]": [] for i in range(9)},
            "freq": [],
        },
    )
)
traces_sample_STEPS4.append(copy.deepcopy(traces_sample_STEPS4[-1]))
traces_sample_STEPS4[-1].name = "V_z_max"

"""Create the sample database"""
sample_STEPS4 = TraceDB(
    "STEPS4",
    traces_sample_STEPS4,
    "/home/katta/projects/STEPS4Models/Validations/rallpack3/sample_STEPS4/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


##########################################

"""Create the benchmark STEPS 3 traces """
multi = 1
traces_benchmark_STEPS3 = []
traces_benchmark_STEPS3.append(Trace("t", "s", multi=1))

traces_benchmark_STEPS3.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_peak_y', {i}]": [] for i in range(9)},
            **{f"['i_peak_t', {i}]": [] for i in range(9)},
            "freq": [],
        },
    )
)
traces_benchmark_STEPS3.append(copy.deepcopy(traces_benchmark_STEPS3[-1]))
traces_benchmark_STEPS3[-1].name = "V_z_max"

"""Create the benchmark database"""
benchmark_STEPS3 = TraceDB(
    "STEPS3",
    traces_benchmark_STEPS3,
    "/home/katta/projects/STEPS4Models/Validations/rallpack3/benchmark_STEPS3/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)

##########################################

"""Create the benchmark NEURON traces """
multi = 1e-3
traces_benchmark_NEURON = []
traces_benchmark_NEURON.append(Trace("t", "s", multi=multi))

traces_benchmark_NEURON.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_peak_y', {i}]": [] for i in range(9)},
            **{f"['i_peak_t', {i}]": [] for i in range(9)},
            "freq": [],
        },
    )
)
traces_benchmark_NEURON.append(copy.deepcopy(traces_benchmark_NEURON[-1]))
traces_benchmark_NEURON[-1].name = "V_z_max"

"""Create the benchmark database"""
benchmark_NEURON = TraceDB(
    "NEURON",
    traces_benchmark_NEURON,
    "rallpack3/benchmark_NEURON",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)

##########################################

"""Create the comparator for advanced studies"""
comp = Comparator(traceDBs=[sample_STEPS4, benchmark_STEPS3, benchmark_NEURON])

"""Perform the ks test"""

for tDBnames, ks_tests in comp.test_ks().items():
    print(tDBnames)
    for k, v in sorted(ks_tests.items(), key=lambda k: k[0]):
        print(k, v)

# this can take some time. Commented for now
# """Compute the mse"""
# for tDBnames, mse_tests in comp.mse_refactored().items():
#     print(tDBnames)
#     for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
#         print(k, *v.items())

"""Plots"""

bindwidth_y = 0.0005
bindwidth_t = 0.0005
bindwidth_Hz = 1

suffix = "rallpack3_STEPS3_vs_STEPS4"
savefig_path = "rallpack3/pics"
comp.distplot(
    "V_z_min",
    f"freq",
    binwidth=bindwidth_Hz,
    savefig_path=savefig_path,
    suffix=suffix,
)

comp.distplot(
    "V_z_max",
    f"freq",
    binwidth=bindwidth_Hz,
    savefig_path=savefig_path,
    suffix=suffix,
)

for i in range(9):
    comp.distplot(
        "V_z_min",
        f"['i_peak_y', {i}]",
        binwidth=bindwidth_y,
        savefig_path=savefig_path,
        suffix=suffix,
        traceDB_names=["STEPS3", "STEPS4"],
    )
    comp.distplot(
        "V_z_min",
        f"['i_peak_t', {i}]",
        binwidth=bindwidth_t,
        savefig_path=savefig_path,
        suffix=suffix,
        traceDB_names=["STEPS3", "STEPS4"],
    )
    comp.distplot(
        "V_z_max",
        f"['i_peak_y', {i}]",
        binwidth=bindwidth_y,
        savefig_path=savefig_path,
        suffix=suffix,
        traceDB_names=["STEPS3", "STEPS4"],
    )
    comp.distplot(
        "V_z_max",
        f"['i_peak_t', {i}]",
        binwidth=bindwidth_t,
        savefig_path=savefig_path,
        suffix=suffix,
        traceDB_names=["STEPS3", "STEPS4"],
    )
