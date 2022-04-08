import copy
import sys

import matplotlib.pyplot as plt

from postproc.comparator import Comparator
from postproc.figure import Figure
from postproc.traceDB import TraceDB
from postproc.trace import Trace


def check(
    analytical_raw_traces_folder="rallpack1/raw_traces/analytical",
    STEPS3_raw_traces_folder="rallpack1/raw_traces/STEPS3",
    STEPS4_raw_traces_folder="rallpack1/raw_traces/STEPS4",
    fig_folder="rallpack1/pics",
):

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
        analytical_raw_traces_folder,
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

    """Create the sample database"""
    sample_STEPS3 = TraceDB(
        "STEPS3",
        traces_sample_STEPS3,
        STEPS3_raw_traces_folder,
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
        STEPS4_raw_traces_folder,
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """comparator"""

    comp = Comparator(traceDBs=[benchmark_analytic, sample_STEPS3, sample_STEPS4])

    # We present the pictures only for analytic vs STEPS4 as it is the most relevant
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    comp._plot(
        benchmarkDB_name="analytic",
        sampleDB_name="STEPS4",
        trace_name_b="V zmin",
        savefig_path=fig_folder,
        isdiff=False,
        pplot=ax[0],
    )
    ax[0].set_title("A\n", loc="left", fontweight="bold")
    comp._plot(
        benchmarkDB_name="analytic",
        sampleDB_name="STEPS4",
        trace_name_b="V zmax",
        savefig_path=fig_folder,
        isdiff=False,
        pplot=ax[1],
    )
    ax[1].set_title("B\n", loc="left", fontweight="bold")
    fig.tight_layout()
    Figure.savefig(savefig_path=fig_folder, file_name="traces", fig=fig)
    fig.show()

    """Compute the mse"""
    for tDBnames, mse_tests in comp.mse_refactored(normalized=False).items():
        print(tDBnames)
        for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
            # the std. mesh is quite coarse and the error with the analytic solution may be considered still big.
            # However, we are also comparing with STEPS 3 where we can be much more strict.
            err = 1e-1 if "analytic" in tDBnames else 1e-3
            print(k, *v.items(), f" Max accepted err: {err}")

            assert v["amax"] < err


if __name__ == "__main__":
    check(*sys.argv[1:])
