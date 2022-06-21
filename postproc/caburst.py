import logging
import sys

import matplotlib.pyplot as plt

from postproc.comparator import Comparator
from postproc.trace import Trace
from postproc.traceDB import TraceDB
from postproc.utils import Utils

logging.basicConfig(level=logging.WARNING)


def check(
    sample_0_raw_traces_folder="caburst/raw_traces/STEPS4",
    sample_1_raw_traces_folder="caburst/raw_traces/STEPS3",
    savefig_path="caburst/pics",
):
    filter = []
    goodness_of_fit_test_type = "ks"
    with_title = False

    sample_names, sample_0_DB, sample_1_DB = create_base_DBs(
        sample_0_raw_traces_folder,
        sample_1_raw_traces_folder,
    )

    """Create the comparator for advanced studies"""
    comp = Comparator(traceDBs=[sample_1_DB, sample_0_DB])

    Utils.pretty_print_goodness_of_fit(comp, goodness_of_fit_test_type, filter)

    """Plots"""

    # plot_avg_and_std(comp, savefig_path, with_title)

    fig, axtot = plt.subplots(3, 2, figsize=(9, 10))
    for i, membrane in enumerate(["smooth", "spiny"]):
        for j, op in enumerate(["min", "max"]):
            ax = axtot[i][j]
            comp.avgplot_raw_traces(
                trace_name=f"{membrane} {op} V",
                std=False,
                ax=ax,
            )
            ax.set_xlabel("ms")
            ax.set_ylabel("mV")
            Utils.set_subplot_title(
                i,
                j,
                2,
                ax,
                f"avg. and conf. int. {membrane}, {op}" if with_title else None,
            )
            ax.legend(["avg. and conf. int. STEPS3", "avg. and conf. int. STEPS4"])

    ax = axtot[2][0]
    comp.avgplot_raw_traces(
        trace_name=f"spiny min V",
        std=False,
        ax=ax,
        conf_int_fill_between_kwargs={"alpha": 0.3},
    )
    ax.set_xlabel("ms")
    ax.set_ylabel("mV")
    ax.set_xlim([35, 37])
    ax.set_ylim([-39, -41])
    Utils.set_subplot_title(2, 0, 2, ax, f"Focus of panel C" if with_title else None)
    ax.legend(["avg. and conf. int. STEPS3", "avg. and conf. int. STEPS4"])

    fig.delaxes(axtot[2][1])
    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="avg_and_conf_int", fig=fig)
    fig.show()


def plot_avg_and_std(comp, savefig_path, with_title):
    fig, axtot = plt.subplots(2, 2)
    for i, membrane in enumerate(["smooth", "spiny"]):
        for j, op in enumerate(["max", "min"]):
            ax = axtot[i][j]
            comp.avgplot_raw_traces(
                trace_name=f"{membrane} {op} V",
                conf_lvl=0,
                ax=ax,
                std_fill_between_kwargs={"alpha": 0.5},
            )
            ax.set_xlabel("ms")
            ax.set_ylabel("mV")
            Utils.set_subplot_title(
                i, j, 2, ax, f"avg. and std. {membrane}, {op}" if with_title else None
            )
            ax.legend(["avg. and std. STEPS3", "avg. and std. STEPS4"])
    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="avg_and_std", fig=fig)
    fig.show()


def create_base_DBs(
    sample_0_raw_traces_folder,
    sample_1_raw_traces_folder,
):

    sample_names = Utils.autonaming_after_folders(
        sample_0_raw_traces_folder, sample_1_raw_traces_folder
    )

    """Create the benchmark traces"""
    multi = 1000
    traces_sample_1 = []

    for membrane in ["smooth", "spiny"]:
        for op in ["max", "min"]:
            traces_sample_1.append(
                Trace(
                    f"{membrane} {op} V",
                    "mV",
                    multi=multi,
                    reduce_ops={
                        "amin": [],
                        "amax": [],
                        "['i_peak_t', 0]": [],
                        "['i_peak_y', 0]": [],
                        # "['i_peak_y', 1]": [],
                        "['val', 0.038]": [],
                        "n_peaks": [],
                    },
                )
            )
    traces_sample_1.append(Trace("t", "ms", multi=multi))

    """ create the benchmark database"""
    sample_1_DB = TraceDB(
        sample_names[1],
        traces_sample_1,
        sample_1_raw_traces_folder,
        clear_refined_traces_cache=True,
        save_refined_traces_cache=True,
        keep_raw_traces=True,
    )

    """Create the sample traces"""
    multi = 1000
    traces_sample_0 = []

    for membrane in ["smooth", "spiny"]:
        for op in ["max", "min"]:
            traces_sample_0.append(
                Trace(
                    f"{membrane} {op} V",
                    "mV",
                    multi=multi,
                    reduce_ops={
                        "amin": [],
                        "amax": [],
                        "['i_peak_t', 0]": [],
                        "['i_peak_y', 0]": [],
                        # "['i_peak_y', 1]": [],
                        "['val', 0.038]": [],
                        "n_peaks": [],
                    },
                )
            )
    traces_sample_0.append(Trace("t", "ms", multi=multi))

    """Create the sample database"""
    sample_0_DB = TraceDB(
        sample_names[0],
        traces_sample_0,
        sample_0_raw_traces_folder,
        clear_refined_traces_cache=True,
        save_refined_traces_cache=True,
        keep_raw_traces=True,
    )

    return sample_names, sample_0_DB, sample_1_DB


if __name__ == "__main__":
    check(*sys.argv[1:])
