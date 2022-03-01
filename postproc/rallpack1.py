from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy

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
    "rallpack1/raw_traces/analytical",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
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

"""Create the sample database"""
sample_STEPS4 = TraceDB(
    "STEPS4",
    traces_sample_STEPS4,
    "rallpack1/raw_traces/STEPS4",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""comparator"""

comp = Comparator(traceDBs=[benchmark_analytic, sample_STEPS4])
comp.plot(trace_name_b="V zmin", savefig_path="rallpack1/pics", isdiff=False, suffix="")
comp.plot(trace_name_b="V zmax", savefig_path="rallpack1/pics", isdiff=False, suffix="")

"""Compute the mse"""
for tDBnames, mse_tests in comp.mse_refactored(normalized=False).items():
    print(tDBnames)
    for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
        print(k, *v.items())
