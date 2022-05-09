import copy
import sys

import matplotlib.pyplot as plt

from postproc.comparator import Comparator
from postproc.figure import Figure
from postproc.traceDB import TraceDB
from postproc.trace import Trace
from postproc.utils import Utils


def check(
    STEPS4_raw_traces_folder="rallpack3/raw_traces/STEPS4/testing",
    STEPS3_raw_traces_folder="rallpack3/raw_traces/STEPS3/testing",
    savefig_path="rallpack3/pics",
):

    npeaks = 17
    multi_t = 1000
    multi_y = 1000
    filter = []  # ["n_peaks", 17]

    # ##########################################

    """Create the benchmark traces. How do you want to refine the data? Usually exactly like the sample traces"""
    traces_STEPS3 = []
    traces_STEPS3.append(Trace("t", "ms", multi=multi_t))

    traces_STEPS3.append(
        Trace(
            "V zmin",
            "mV",
            multi=multi_y,
            reduce_ops={
                "amin": [],
                "amax": [],
                **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
                **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
                f"['freq', {1/multi_y}, {1/multi_t}]": [],
                "n_peaks": [],
                "peaks_y": [],
                "peaks_t": [],
            },
        )
    )
    traces_STEPS3.append(copy.deepcopy(traces_STEPS3[-1]))
    traces_STEPS3[-1].name = "V zmax"

    """Create the benchmark database"""
    STEPS3_DB = TraceDB(
        "STEPS3",
        traces_STEPS3,
        STEPS3_raw_traces_folder,
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """Create the sample traces. How do you want to refine the data?"""
    traces_STEPS4 = []
    traces_STEPS4.append(Trace("t", "ms", multi=multi_t))
    traces_STEPS4.append(
        Trace(
            "V zmin",
            "mV",
            multi=multi_y,
            reduce_ops={
                "amin": [],
                "amax": [],
                **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
                **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
                f"['freq', {1/multi_y}, {1/multi_t}]": [],
                "n_peaks": [],
                "peaks_y": [],
                "peaks_t": [],
            },
        )
    )
    traces_STEPS4.append(copy.deepcopy(traces_STEPS4[-1]))
    traces_STEPS4[-1].name = "V zmax"

    """Create the sample database"""
    STEPS4_DB = TraceDB(
        "STEPS4",
        traces_STEPS4,
        STEPS4_raw_traces_folder,
        clear_raw_traces_cache=True,
        clear_refined_traces_cache=True,
    )

    """Create the comparator for advanced studies
    
    Note: anywhere is relevant, the first traceDB is considered the benchmark. The others are samples
    """
    comp = Comparator(traceDBs=[STEPS3_DB, STEPS4_DB])

    """p value statistics and graphs
    
    create a database using the refined data produced before as raw data for the new database
    """
    pvalues = {
        "V zmin": {
            "i_peak_y": [],
            "i_peak_t": [],
        },
        "V zmax": {
            "i_peak_y": [],
            "i_peak_t": [],
        },
    }
    # the ks tests are our new raw data
    for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
        for t, d in ks_tests.items():
            for k, v in d.items():
                if t in pvalues:
                    for k_slim in pvalues[t]:
                        if k_slim in k:
                            pvalues[t][k_slim].append(v.pvalue)

    pvalues_traces = [Trace(k, "mV", reduce_ops=v) for k, v in pvalues.items()]

    """Create a database"""
    pvalues_traceDB = TraceDB("p values", pvalues_traces, is_refine=False)

    comp_pvalues = Comparator(traceDBs=[pvalues_traceDB])

    # filter data out

    """Perform the ks test"""
    for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
        print(tDBnames)
        for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
            for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
                print(t, k, v)

    """Plots"""

    # print(STEPS4_DB)
    # exit()

    bindwidth_y = 0.0005 * multi_y
    bindwidth_t = 0.001 * multi_t
    bindwidth_Hz = 0.1

    # ### this works only if we use standard units: s, V
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    for i, tracename in enumerate(["V zmin", "V zmax"]):
        comp.distplot(
            tracename,
            f"['freq', {1/multi_y}, {1/multi_t}]",
            binwidth=bindwidth_Hz,
            savefig_path=savefig_path,
            xlabel="Hz",
            filter=filter,
            pplot=ax[0][i],
        )
        ax[0][i].set_title(f"{'A' if i == 0 else 'B'}\n", loc="left", fontweight="bold")
        comp.distplot(
            tracename,
            f"n_peaks",
            binwidth=1,
            binrange=[12.5, 19.5],
            savefig_path=savefig_path,
            filter=filter,
            xlabel="n peaks",
            pplot=ax[1][i],
        )
        ax[1][i].set_title(f"{'C' if i == 0 else 'D'}\n", loc="left", fontweight="bold")
    fig.tight_layout()
    Figure.savefig(savefig_path=savefig_path, file_name="npeaks_and_freq", fig=fig)
    fig.show()

    exit()

    for op_tuple in [("peaks_t", "ms", bindwidth_t), ("peaks_y", "mV", bindwidth_y)]:
        op, label, binwidth = op_tuple
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        for i, tracename in enumerate(["V zmin", "V zmax"]):
            comp.distplot(
                tracename,
                op,
                binwidth=binwidth,
                filter=filter,
                xlabel=label,
                pplot=ax[0][i],
            )
            ax[0][i].set_title(
                f"{'A' if i == 0 else 'B'}\n", loc="left", fontweight="bold"
            )

            comp.avgplot_refined_traces(
                tracename,
                [f"['i_{op.replace('s', '')}', {q}]" for q in range(npeaks)],
                xlabel="peak n",
                ylabel=label,
                savefig_path=savefig_path,
                title=f"{tracename} {op} avg. and std.",
                pplot=ax[1][i],
            )
            ax[1][i].set_title(
                f"{'C' if i == 0 else 'D'}\n", loc="left", fontweight="bold"
            )
        fig.tight_layout()
        Figure.savefig(savefig_path=savefig_path, file_name=op, fig=fig)
        fig.show()

    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    subplot_label = "A"
    for i, tname in enumerate(["V zmax", "V zmin"]):
        for j, op in enumerate(["i_peak_y", "i_peak_t"]):
            comp_pvalues.distplot(
                tname,
                op,
                binwidth=0.1,
                binrange=[0, 1],
                xlabel="p values",
                savefig_path=savefig_path,
                pplot=ax[i][j],
            )
            ax[i][j].set_title(subplot_label + "\n", loc="left", fontweight="bold")
            subplot_label = chr(ord(subplot_label) + 1)
    fig.tight_layout()
    Figure.savefig(savefig_path=savefig_path, file_name="p_values", fig=fig)
    fig.show()


if __name__ == "__main__":
    check(*sys.argv[1:])
