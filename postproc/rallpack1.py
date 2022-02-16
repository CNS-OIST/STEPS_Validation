from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy
import numpy

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

"""Create the sample database"""
sample_STEPS4 = TraceDB(
    "STEPS4",
    traces_sample_STEPS4,
    "rallpack1/sample_STEPS4/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


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


"""Create the sample database"""
benchmark_analytic = TraceDB(
    "analytic",
    traces_benchmark_analytic,
    "rallpack1/benchmark_analytic/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Benchmark_STEPS3"""

multi = 1000
traces_benchmark_STEPS3 = []
traces_benchmark_STEPS3.append(Trace("t", "ms", multi=multi))
traces_benchmark_STEPS3.append(
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
traces_benchmark_STEPS3.append(copy.deepcopy(traces_benchmark_STEPS3[-1]))
traces_benchmark_STEPS3[-1].name = "V zmax"

"""Create the sample database"""
benchmark_STEPS3 = TraceDB(
    "STEPS3",
    traces_benchmark_STEPS3,
    "rallpack1/benchmark_STEPS3/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""comparator"""

comp = Comparator(traceDBs=[sample_STEPS4, benchmark_STEPS3, benchmark_analytic])
comp.plot(trace_name_b="V zmin", savefig_path="rallpack1/pics", isdiff=False)
comp.plot(trace_name_b="V zmax", savefig_path="rallpack1/pics", isdiff=False)

"""Compute the mse"""
for tDBnames, mse_tests in comp.mse_refactored(normalized=False).items():
    # if tDBnames == "STEPS4_vs_analytic":
    print(tDBnames)
    for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
        print(k, *v.items())


# """Mesh scaling"""
# import matplotlib.pyplot as plt
# dofs = [42, 267, 2595, 12180]
# V zmax_mse = [4.90431547267612e-07, 4.3761326885036406e-11, 1.0878045760284713e-12, 3.6528686422550846e-13]
# plt.loglog(dofs, V zmax_mse,marker='o')
# plt.ylabel("mse")
# plt.xlabel("DoFs")
# plt.savefig("rallpack1/pics/mesh_scaling")
# plt.show()
