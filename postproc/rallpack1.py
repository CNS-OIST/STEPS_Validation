from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy
import numpy

"""Sample_STEPS4"""
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
        },
    )
)
traces_sample_STEPS4.append(copy.deepcopy(traces_sample_STEPS4[-1]))
traces_sample_STEPS4[-1].name = "V_z_max"

"""Create the sample database"""
sample_STEPS4 = TraceDB(
    "STEPS4",
    traces_sample_STEPS4,
    "rallpack1/sample_STEPS4/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Benchmark_analytic"""

multi = 1
traces_benchmark_analytic = []
traces_benchmark_analytic.append(Trace("t", "s", multi=multi))
traces_benchmark_analytic.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
        },
    )
)
traces_benchmark_analytic.append(copy.deepcopy(traces_benchmark_analytic[-1]))
traces_benchmark_analytic[-1].name = "V_z_max"


"""Create the sample database"""
benchmark_analytic = TraceDB(
    "analytic",
    traces_benchmark_analytic,
    "rallpack1/benchmark_analytic/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Benchmark_STEPS3"""

multi = 1
traces_benchmark_STEPS3 = []
traces_benchmark_STEPS3.append(Trace("t", "s", multi=multi))
traces_benchmark_STEPS3.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
        },
    )
)
traces_benchmark_STEPS3.append(copy.deepcopy(traces_benchmark_STEPS3[-1]))
traces_benchmark_STEPS3[-1].name = "V_z_max"

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
comp.plot(
    trace_name_b="V_z_min", savefig_path="rallpack1/pics", isdiff=False, istitle=False
)
comp.plot(
    trace_name_b="V_z_max", savefig_path="rallpack1/pics", isdiff=False, istitle=False
)

"""Compute the mse"""
for tDBnames, mse_tests in comp.mse_refactored(normalized=False).items():
    print(tDBnames)
    for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
        print(k, *v.items())
