import copy
import sys

import matplotlib.pyplot as plt

from postproc.comparator import Comparator
from postproc.figure import Figure
from postproc.traceDB import TraceDB
from postproc.trace import Trace
from postproc.utils import Utils


def check(
    sample_0_raw_traces_folder="rallpack1/raw_traces/STEPS4",
    sample_1_raw_traces_folder="rallpack1/raw_traces/STEPS3",
    analytical_raw_traces_folder="rallpack1/raw_traces/analytical",
    savefig_path="rallpack1/pics",
):

    sample_names = Utils.autonaming_after_folders(
        sample_0_raw_traces_folder, sample_1_raw_traces_folder
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
        analytical_raw_traces_folder,
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """sample_1"""
    multi = 1000
    traces_sample_1 = []
    traces_sample_1.append(Trace("t", "ms", multi=multi))
    traces_sample_1.append(
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
    traces_sample_1.append(copy.deepcopy(traces_sample_1[-1]))
    traces_sample_1[-1].name = "V zmax"

    """Create the sample database"""
    sample_1 = TraceDB(
        sample_names[1],
        traces_sample_1,
        sample_1_raw_traces_folder,
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """sample_0"""
    multi = 1000
    traces_sample_0 = []
    traces_sample_0.append(Trace("t", "ms", multi=multi))
    traces_sample_0.append(
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
    traces_sample_0.append(copy.deepcopy(traces_sample_0[-1]))
    traces_sample_0[-1].name = "V zmax"

    """Create the sample database"""
    sample_0 = TraceDB(
        sample_names[0],
        traces_sample_0,
        sample_0_raw_traces_folder,
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """comparator"""

    comp = Comparator(traceDBs=[benchmark_analytic, sample_1, sample_0])

    # We present the pictures only for analytic vs STEPS4 as it is the most relevant
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    comp._plot(
        benchmarkDB_name="analytic",
        sampleDB_name=sample_names[0],
        trace_name_b="V zmin",
        savefig_path=savefig_path,
        isdiff=False,
        pplot=ax[0],
    )
    ax[0].set_title("A\n", loc="left", fontweight="bold")
    comp._plot(
        benchmarkDB_name="analytic",
        sampleDB_name=sample_names[0],
        trace_name_b="V zmax",
        savefig_path=savefig_path,
        isdiff=False,
        pplot=ax[1],
    )
    ax[1].set_title("B\n", loc="left", fontweight="bold")
    fig.tight_layout()
    Figure.savefig(savefig_path=savefig_path, file_name="traces", fig=fig)
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
