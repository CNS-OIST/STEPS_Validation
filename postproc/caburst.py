import logging
import sys

import matplotlib.pyplot as plt

from postproc.comparator import Comparator
from postproc.figure import Figure
from postproc.traceDB import TraceDB
from postproc.trace import Trace
from postproc.utils import Utils

logging.basicConfig(level=logging.WARNING)


def check(
    sample_0_raw_traces_folder="caburst/raw_traces/STEPS4",
    sample_1_raw_traces_folder="caburst/raw_traces/STEPS3",
    savefig_path="caburst/pics",
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
        clear_raw_traces_cache=False,
        clear_refined_traces_cache=False,
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
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """Create the comparator for advanced studies"""
    comp = Comparator(traceDBs=[sample_1_DB, sample_0_DB])

    """Perform the ks test"""
    filter = []
    for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
        print(tDBnames)
        for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
            for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
                print(t, k, v)

    """Plots"""

    fig, ax = plt.subplots(2, 2)
    subplot_label = "A"
    for i, membrane in enumerate(["smooth", "spiny"]):
        for j, op in enumerate(["max", "min"]):
            comp.avgplot_raw_traces(
                trace_name=f"{membrane} {op} V",
                conf_lvl=0,
                savefig_path="caburst/pics",
                suffix="",
                pplot=ax[i][j],
                legendfontsize=5,
            )
            ax[i][j].set_title(subplot_label + "\n", loc="left", fontweight="bold")
            subplot_label = chr(ord(subplot_label) + 1)
    fig.tight_layout()
    Figure.savefig(savefig_path=savefig_path, file_name="avg_and_std", fig=fig)
    fig.show()

    fig, ax = plt.subplots(3, 2, figsize=(9, 10))
    subplot_label = "A"
    legendfontsize = 8
    for i, membrane in enumerate(["smooth", "spiny"]):
        for j, op in enumerate(["min", "max"]):
            comp.avgplot_raw_traces(
                trace_name=f"{membrane} {op} V",
                std=False,
                savefig_path="caburst/pics",
                suffix="",
                pplot=ax[i][j],
                legendfontsize=legendfontsize,
            )
            ax[i][j].set_title(subplot_label + "\n", loc="left", fontweight="bold")
            subplot_label = chr(ord(subplot_label) + 1)

    comp.avgplot_raw_traces(
        trace_name=f"spiny min V",
        std=False,
        savefig_path="caburst/pics",
        suffix="",
        title=r"Focus of panel C",
        pplot=ax[2][0],
        legendfontsize=legendfontsize,
        xlim=[35, 37],
        ylim=[-39, -41],
    )
    ax[2][0].set_title(subplot_label + "\n", loc="left", fontweight="bold")
    fig.delaxes(ax[2][1])
    subplot_label = chr(ord(subplot_label) + 1)
    fig.tight_layout()
    Figure.savefig(savefig_path=savefig_path, file_name="avg_and_conf_int", fig=fig)
    fig.show()


if __name__ == "__main__":
    check(*sys.argv[1:])
