import copy
import sys

import matplotlib.pyplot as plt
import pandas

from postproc.comparator import Comparator
from postproc.figure import Figure
from postproc.trace import Trace
from postproc.traceDB import TraceDB
from postproc.utils import Utils


def check(
    sample_0_raw_traces_folder="rallpack1/raw_traces/STEPS4",
    sample_1_raw_traces_folder="rallpack1/raw_traces/STEPS3",
    analytical_raw_traces_folder="rallpack1/raw_traces/analytical",
    savefig_path="rallpack1/pics",
):

    benchmark_analytic, sample_1, sample_0, sample_names = create_base_DBs(
        sample_0_raw_traces_folder,
        sample_1_raw_traces_folder,
        analytical_raw_traces_folder,
    )

    diff_analytic_STEPS4 = create_diff_DB(benchmark_analytic, sample_1)

    renamed_analytic = rename_traces(benchmark_analytic)
    renamed_sample_1 = rename_traces(sample_1)

    """comparator"""

    comp = Comparator(traceDBs=[benchmark_analytic, sample_1, sample_0])
    for comp_name, c in comp.mse_refactored(normalized=False).items():
        print(comp_name)
        for trace, res in c.items():
            print(trace, res)

    fig, ax = plt.subplots(1, 1)
    inset_ax = fig.add_axes([0.25, 0.5, 0.5, 0.4])
    fmt = 0
    fmt = renamed_analytic.plot(title="", fmt=fmt, pplot=inset_ax, finalize=False)

    renamed_sample_1.plot(
        title="", fmt=fmt, pplot=inset_ax, finalize=True, with_legend=True
    )

    diff_analytic_STEPS4.plot(pplot=ax, ylim=[-1, 3], with_legend=True, title="")

    fig.tight_layout()
    Figure.savefig(savefig_path=savefig_path, file_name="traces", fig=fig)
    fig.show()


def create_base_DBs(
    sample_0_raw_traces_folder,
    sample_1_raw_traces_folder,
    analytical_raw_traces_folder,
):
    """ Create the standard trace databases required for postprocessing """

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
        clear_refined_traces_cache=True,
        keep_raw_traces=True,
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
        clear_refined_traces_cache=True,
        keep_raw_traces=True,
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
        clear_refined_traces_cache=True,
        keep_raw_traces=True,
    )

    return benchmark_analytic, sample_0, sample_1, sample_names


def create_diff_DB(DB0, DB1):
    """ Create the database of the differences of the traces """

    traces_diff_analytic_STEPS4 = []
    traces_diff_analytic_STEPS4.append(Trace("t", "mS", multi=1))

    trace_name = "diff V zmin"
    trace_name_old = "V zmin"
    traces_diff_analytic_STEPS4.append(Trace(trace_name, "mV", multi=1))
    interp_sample, interp_benchmark, interp_time = Utils._format_traces(
        DB1.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB1.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
        DB0.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB0.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
    )
    traces_diff_analytic_STEPS4[-1].raw_traces = pandas.DataFrame(
        {trace_name: interp_benchmark - interp_sample}
    )
    traces_diff_analytic_STEPS4[0].add_raw_trace(trace_name, interp_time)
    trace_name = "diff V zmax"
    trace_name_old = "V zmax"
    traces_diff_analytic_STEPS4.append(Trace(trace_name, "mV", multi=1))
    interp_sample, interp_benchmark, _ = Utils._format_traces(
        DB1.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB1.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
        DB0.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB0.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
    )
    traces_diff_analytic_STEPS4[-1].raw_traces = pandas.DataFrame(
        {trace_name: interp_benchmark - interp_sample}
    )
    traces_diff_analytic_STEPS4[0].add_raw_trace(trace_name, interp_time)

    diff_DB0_DB1 = TraceDB(
        "diffs",
        traces_diff_analytic_STEPS4,
        clear_refined_traces_cache=True,
        keep_raw_traces=True,
    )

    return diff_DB0_DB1


def rename_traces(DB):
    traces = []
    traces.append(Trace("t", "mS", multi=1))
    for i in DB.traces.values():
        if i.name == "t":
            continue

        name = f"{DB.name}, {i.name}"
        traces.append(Trace(i.name, i.unit))

        traces[-1].add_raw_trace(name, i.raw_traces.iloc[:, 0])
        traces[0].add_raw_trace(name, DB.get_time_trace().raw_traces.iloc[:, 0])

    return TraceDB(
        f"{DB.name}",
        traces,
        clear_refined_traces_cache=True,
        keep_raw_traces=True,
    )


if __name__ == "__main__":
    check(*sys.argv[1:])
