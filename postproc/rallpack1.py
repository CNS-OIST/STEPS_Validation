import copy
import sys
import os

import matplotlib.image as image
import matplotlib.pyplot as plt
import pandas

from postproc.comparator import Comparator
from postproc.trace import Trace
from postproc.traceDB import TraceDB
from postproc.utils import Utils


def check(
    # sample_0_raw_traces_folder="/home/katta/projects/STEPS4ModelRelease/rallpack1/raw_traces/STEPS4",
    sample_0_raw_traces_folder="rallpack1/raw_traces/STEPS4",
    sample_1_raw_traces_folder="rallpack1/raw_traces/STEPS3",
    analytical_raw_traces_folder="rallpack1/raw_traces/analytical",
    savefig_path="rallpack1/pics",
):
    with_title = False

    benchmark_analytic, sample_1, sample_0, sample_names = create_base_DBs(
        sample_0_raw_traces_folder,
        sample_1_raw_traces_folder,
        analytical_raw_traces_folder,
    )

    compare_raw_traces(benchmark_analytic, sample_0, sample_1, savefig_path, with_title)

    mesh_scaling_plot(benchmark_analytic, savefig_path)

def compare_raw_traces(benchmark_analytic, sample_0, sample_1, savefig_path, with_title):
    diff_analytic_STEPS4 = create_diff_DB(benchmark_analytic, sample_1)

    renamed_analytic = rename_traces(benchmark_analytic)
    renamed_sample_1 = rename_traces(sample_1)

    """comparator"""
    comp = Comparator(traceDBs=[benchmark_analytic, sample_1, sample_0])
    for comp_name, c in comp.mse_refactored(normalized=False).items():
        print(comp_name)
        for trace, res in c.items():
            print(trace, res)

    """Traces figure"""
    fig, axtot = plt.subplots(
        1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [2, 4]}
    )
    ax = axtot[0]
    ax.imshow(image.imread("rallpack1/base_pics/rallpack1_setup.jpg"))
    ax.axis("off")
    Utils.set_subplot_title(0, 0, 2, ax, f"rallpack1 setup" if with_title else None)

    ax = axtot[1]
    diff_analytic_STEPS4.plot(ax=ax, fmt=[{"linestyle": "-"}, {"linestyle": "--"}])
    ax.set_ylim([-0.9, 2.5])
    ax.legend(["V zmin, analytic - STEPS4", "V zmax, analytic - STEPS4"], loc=4)
    ax.set_xlabel("ms")
    ax.set_ylabel("mV")
    Utils.set_subplot_title(0, 1, 2, ax, f"analytic - STEPS4" if with_title else None)

    inset_ax = fig.add_axes([0.5, 0.5, 0.46, 0.4])
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fmt = [
        {"linestyle": "-", "color": default_colors[0]},
        {"linestyle": "-.", "color": default_colors[2]},
    ]
    renamed_analytic.plot(ax=inset_ax, fmt=fmt)
    fmt = [
        {"linestyle": "--", "color": default_colors[1]},
        {"linestyle": ":", "color": default_colors[3]},
    ]
    renamed_sample_1.plot(ax=inset_ax, fmt=fmt)
    inset_ax.set_xlabel("ms")
    inset_ax.set_ylabel("mV")
    inset_ax.legend(
        ["V zmin, analytic", "V zmax analytic", "V zmin, STEPS4", "V zmax STEPS4"]
    )

    fig.tight_layout()
    Utils.savefig(savefig_path, "traces", fig)
    fig.show()


def mesh_scaling_plot(benchmark_analytic, savefig_path):
    traces_path = "rallpack1/raw_traces/STEPS4/mesh_scaling/"

    vzmin = []
    vzmax = []
    x = [42, 267, 2595, 12180]
    for dir in x:

        """Create the standard trace databases required for postprocessing"""

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
            dir,
            traces_sample_1,
            traces_path + str(dir),
            clear_refined_traces_cache=True,
            keep_raw_traces=True,
            save_refined_traces_cache=False,
        )

        """comparator"""

        comp = Comparator(traceDBs=[benchmark_analytic, sample_1])
        for comp_name, c in comp.mse_refactored(normalized=False).items():
            # print(comp_name)
            vzmin.append(c["V zmin"]["mean"])
            vzmax.append(c["V zmax"]["mean"])

    fig, ax = plt.subplots()

    ax.loglog(x, vzmin, label="V zmin", marker='x')
    ax.loglog(x, vzmax, label="V zmax", marker='x')
    ax.legend()

    # diff_analytic_STEPS4.plot(ax=ax, fmt=[{"linestyle": "-"}, {"linestyle": "--"}])
    # ax.set_ylim([-0.9, 2.5])
    # ax.legend(["V zmin, analytic - STEPS4", "V zmax, analytic - STEPS4"], loc=4)
    # ax.set_xlabel("ms")
    # ax.set_ylabel("mV")
    # Utils.set_subplot_title(0, 1, 2, ax, f"analytic - STEPS4" if with_title else None)
    #
    # inset_ax = fig.add_axes([0.5, 0.5, 0.46, 0.4])
    # default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # fmt = [
    #     {"linestyle": "-", "color": default_colors[0]},
    #     {"linestyle": "-.", "color": default_colors[2]},
    # ]
    # renamed_analytic.plot(ax=inset_ax, fmt=fmt)
    # fmt = [
    #     {"linestyle": "--", "color": default_colors[1]},
    #     {"linestyle": ":", "color": default_colors[3]},
    # ]
    # renamed_sample_1.plot(ax=inset_ax, fmt=fmt)
    # inset_ax.set_xlabel("ms")
    # inset_ax.set_ylabel("mV")
    # inset_ax.legend(
    #     ["V zmin, analytic", "V zmax analytic", "V zmin, STEPS4", "V zmax STEPS4"]
    # )

    # fig.tight_layout()
    Utils.savefig(savefig_path, "mesh_scaling", fig)
    # fig.show()
    # print(vzmin)
    # print(vzmax)




def create_base_DBs(
    sample_0_raw_traces_folder,
    sample_1_raw_traces_folder,
    analytical_raw_traces_folder,
):
    """Create the standard trace databases required for postprocessing"""

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
        save_refined_traces_cache=False,
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
        save_refined_traces_cache=False,
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
        save_refined_traces_cache=True,
    )

    return benchmark_analytic, sample_0, sample_1, sample_names


def create_diff_DB(DB0, DB1):
    """Create the database of the differences of the traces"""

    traces_diff_analytic_STEPS4 = []
    traces_diff_analytic_STEPS4.append(Trace("t", "mS", multi=1))

    trace_name = f"V zmin, {DB0.name} - {DB1.name}"
    trace_name_old = "V zmin"
    traces_diff_analytic_STEPS4.append(Trace(trace_name, "mV", multi=1))

    interp0, interp1, interp_time = Utils._format_traces(
        DB0.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB0.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
        DB1.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB1.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
    )

    traces_diff_analytic_STEPS4[-1].raw_traces = pandas.DataFrame(
        {trace_name: interp0 - interp1}
    )
    traces_diff_analytic_STEPS4[0].add_raw_trace(trace_name, interp_time)
    trace_name = f"V zmax, {DB0.name} - {DB1.name}"
    trace_name_old = "V zmax"
    traces_diff_analytic_STEPS4.append(Trace(trace_name, "mV", multi=1))
    interp0, interp1, _ = Utils._format_traces(
        DB0.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB0.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
        DB1.traces["t"].raw_traces.iloc[:, 0].to_numpy(),
        DB1.traces[trace_name_old].raw_traces.iloc[:, 0].to_numpy(),
    )
    traces_diff_analytic_STEPS4[-1].raw_traces = pandas.DataFrame(
        {trace_name: interp0 - interp1}
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
