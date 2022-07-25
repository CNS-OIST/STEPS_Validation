import logging
import sys

import matplotlib.image as image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

    # plot_raw_traces(sample_0_DB, savefig_path, with_title)

    # plot_avg_and_conf_int_and_inset_spiny_min(comp, savefig_path, with_title)

    plot_avg_and_conf_int_and_diff(comp, savefig_path, with_title)

    # plot_avg_and_std(comp, savefig_path, with_title)

    # plot_avg_and_conf_int(comp, savefig_path, with_title)


def plot_raw_traces(DB, savefig_path, with_title):

    fig, axtot = plt.subplots(1, 2, figsize=(8, 4))

    ax = axtot[0]
    ax.imshow(image.imread("caburst/base_pics/Purkinje_structure.png"))
    ax.axis("off")
    Utils.set_subplot_title(0, 0, 2, ax, f"Purkinje structure" if with_title else None)

    ax = axtot[1]
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = [
        [
            {
                "color": default_colors[i],
                "alpha": 0.035,
                "linestyle": ["-", "--", "-.", ":"][i],
            }
        ]
        for i in range(4)
    ]
    DB.plot(ax=ax, fmt=styles)
    ax.set_xlabel("ms")
    ax.set_ylabel("mV")
    leg = ax.legend(["smooth max V", "smooth min V", "spiny max V", "spiny min V"])
    for idx, lh in enumerate(leg.legendHandles):
        lh.set_color(styles[idx][0]["color"])
        lh.set_linestyle(styles[idx][0]["linestyle"])
        lh.set_alpha(1)

    Utils.set_subplot_title(0, 1, 2, ax, f"raw traces" if with_title else None)
    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="raw_traces", fig=fig)
    fig.show()


def plot_avg_and_conf_int_and_inset_spiny_min(comp, savefig_path, with_title):

    fig, ax = plt.subplots(1, 1)
    membrane = "spiny"
    op = "min"
    comp.avgplot_raw_traces(
        trace_name=f"{membrane} {op} V",
        std=False,
        ax=ax,
        conf_int_fill_between_kwargs={"alpha": 0.3},
    )
    ax.set_xlabel("ms")
    ax.set_ylabel("mV")
    ax.set_ylim([-63, -5])
    if with_title:
        ax.set_title(f"avg. and conf. int. {membrane}, {op}")
    ax.legend(["STEPS3", "STEPS4"], loc=2)

    axins = ax.inset_axes([0.5, 0.6, 0.47, 0.37])
    comp.avgplot_raw_traces(
        trace_name=f"{membrane} {op} V",
        std=False,
        ax=axins,
        conf_int_fill_between_kwargs={"alpha": 0.3},
    )
    axins.set_xlim([35, 37])
    axins.set_ylim([-40.5, -38.5])
    axins.set_xlabel("ms")
    axins.set_ylabel("mV")
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="avg_and_conf_int_spiny_min", fig=fig)
    fig.show()


def plot_avg_and_conf_int_and_diff(comp, savefig_path, with_title):
    with_title = True

    fig, axtot = plt.subplots(4, 2, figsize=(10, 10))
    membranes = ["smooth", "spiny"]
    ops = ["min", "max"]
    for i, membrane in enumerate(membranes):
        for j, op in enumerate(ops):

            ax = axtot[i * 2][j]
            comp.avgplot_raw_traces(
                trace_name=f"{membrane} {op} V",
                std=False,
                ax=ax,
                conf_int_fill_between_kwargs={"alpha": 0.3},
            )
            ax.set_xlabel("ms")
            ax.set_ylabel("mV")
            if with_title:
                ax.set_title(f"avg. and conf. int. {membrane}, {op}")

            if i == 0 and j == 1:
                ax.legend(["STEPS3", "STEPS4"])

            ax = axtot[i * 2 + 1][j]
            comp.avgplot_raw_traces(
                trace_name=f"{membrane} {op} V",
                std=False,
                baselineDB="STEPS3",
                ax=ax,
                conf_int_fill_between_kwargs={"alpha": 0.3},
            )
            ax.set_xlabel("ms")
            ax.set_ylabel("mV")

    fig.tight_layout()
    Utils.savefig(path=savefig_path, name=f"avg_and_conf_int_and_diff", fig=fig)
    fig.show()


def plot_avg_and_conf_int(comp, savefig_path, with_title):

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
            ax.legend(["STEPS3", "STEPS4"])

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
    ax.legend(["STEPS3", "STEPS4"])

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
        clear_refined_traces_cache=False,
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
        clear_refined_traces_cache=False,
        save_refined_traces_cache=True,
        keep_raw_traces=True,
    )

    return sample_names, sample_0_DB, sample_1_DB


if __name__ == "__main__":
    check(*sys.argv[1:])
