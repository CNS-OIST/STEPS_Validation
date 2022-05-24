import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from postproc.comparator import Comparator
from postproc.figure import Figure
from postproc.trace import Trace
from postproc.traceDB import TraceDB
from postproc.utils import Utils


def check(
    sample_1_raw_traces_folder="rallpack3/raw_traces/STEPS4",
    sample_0_raw_traces_folder="rallpack3/raw_traces/STEPS3",
):
    sample_names = Utils.autonaming_after_folders(
        sample_0_raw_traces_folder, sample_1_raw_traces_folder
    )

    npeaks = 17
    multi_t = 1000
    multi_y = 1000
    filter = []  # ["n_peaks", 17]
    clear_all_caches = False  # True is used for debugging
    savefig_path = "rallpack3/pics"

    # ##########################################

    """Create the benchmark traces. How do you want to refine the data? Usually exactly like the sample traces"""
    traces_sample_1 = []
    traces_sample_1.append(Trace("t", "ms", multi=multi_t))

    traces_sample_1.append(
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
    traces_sample_1.append(copy.deepcopy(traces_sample_1[-1]))
    traces_sample_1[-1].name = "V zmax"

    """Create the benchmark database"""
    sample_1_DB = TraceDB(
        sample_names[1],
        traces_sample_1,
        sample_1_raw_traces_folder,
        clear_raw_traces_cache=clear_all_caches,
        clear_refined_traces_cache=clear_all_caches,
    )

    """Create the sample traces. How do you want to refine the data?"""
    traces_sample_0 = []
    traces_sample_0.append(Trace("t", "ms", multi=multi_t))
    traces_sample_0.append(
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
    traces_sample_0.append(copy.deepcopy(traces_sample_0[-1]))
    traces_sample_0[-1].name = "V zmax"

    """Create the sample database"""
    sample_0_DB = TraceDB(
        sample_names[0],
        traces_sample_0,
        sample_0_raw_traces_folder,
        clear_raw_traces_cache=clear_all_caches,
        clear_refined_traces_cache=clear_all_caches,
    )

    """Create the comparator for advanced studies

    Note: anywhere is relevant, the first traceDB is considered the benchmark. The others are samples
    """
    comp = Comparator(traceDBs=[sample_1_DB, sample_0_DB])

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
                            pvalues[t][k_slim] = [
                                *pvalues[t][k_slim],
                                *[pp.pvalue for pp in v],
                            ]

    pvalues_traces = [Trace(k, "mV", reduce_ops=v) for k, v in pvalues.items()]

    """Create a database"""
    pvalues_traceDB = TraceDB("p values", pvalues_traces, is_refine=False)
    comp_pvalues = Comparator(traceDBs=[pvalues_traceDB])

    """Perform the ks test"""
    for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
        print(tDBnames)
        for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
            for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
                print(t, k, v)

    """Plots"""
    bindwidth_y = 0.0005 * multi_y
    bindwidth_t = 0.001 * multi_t
    bindwidth_Hz = 0.1

    ### this works only if we use standard units: s, V
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
            title="freq",
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

    """Add pvalue reference and produce the boxplot"""
    pvalues_traces.append(Trace("ref", "", reduce_ops={"": pvalues_reference()}))
    pvalues_traceDB = TraceDB("p values", pvalues_traces, is_refine=False)
    comp_pvalues = Comparator(traceDBs=[pvalues_traceDB])
    comp_pvalues.boxplot_refined_traces(
        ylabel="p values",
        savefig_path=savefig_path,
        title="boxplot p values",
    )

    # """Chunked pvalues"""
    # ip = 0
    # nsections = 10
    # chunked_pvalues = {
    #     "V zmin": {
    #         **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
    #         **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
    #     },
    #     "V zmax": {
    #         **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
    #         **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
    #     },
    # }
    #
    # # the ks tests are our new raw data
    # for tDBnames, ks_tests in comp.test_ks(filter=filter, nsections=nsections).items():
    #     for t, d in ks_tests.items():
    #         for k, v in d.items():
    #             if t in chunked_pvalues:
    #                 for k_slim in chunked_pvalues[t]:
    #                     if k_slim in k:
    #                         chunked_pvalues[t][k_slim] = [*chunked_pvalues[t][k_slim], *[pp.pvalue for pp in v]]
    #
    # chunked_pvalues_traces = [Trace(k, "mV", reduce_ops=v) for k, v in chunked_pvalues.items()]
    #
    # """Create a database"""
    # chunked_pvalues_traceDB = TraceDB("p values", chunked_pvalues_traces, is_refine=False)
    # chunked_comp_pvalues = Comparator(traceDBs=[chunked_pvalues_traceDB])
    #
    # chunked_comp_pvalues.boxplot_refined_traces(
    #     ylabel="p values",
    #     savefig_path=savefig_path,
    #     title="boxplot p values",
    # )


def pvalues_reference(npvalues=100, mean=1, sigma=0.1, size=1000):
    """P values from comparing identical lognormal distributions

    Input:
    - npvalues: number of pvalues. (output size)
    - mean, sigma, size: inputs of numpy.random.lognormal
    """

    pvalues = []
    for i in range(npvalues):
        Y = {}
        Y["Y1"] = np.random.lognormal(mean=mean, sigma=sigma, size=size)
        Y["Y2"] = np.random.lognormal(mean=mean, sigma=sigma, size=size)
        pvalue = sp.stats.ks_2samp(Y["Y1"], Y["Y2"])[1]
        pvalues.append(pvalue)
    return pvalues


if __name__ == "__main__":
    check(*sys.argv[1:])
