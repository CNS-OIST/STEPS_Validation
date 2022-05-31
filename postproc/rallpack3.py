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

import logging
logging.basicConfig(level=logging.INFO)


def check(
    sample_1_raw_traces_folder="rallpack3/raw_traces/STEPS4/testing_highChannelDensity_long",
    sample_0_raw_traces_folder="rallpack3/raw_traces/STEPS3/testing_highChannelDensity_long",
    # sample_1_raw_traces_folder="rallpack3/raw_traces/STEPS4",
    # sample_0_raw_traces_folder="rallpack3/raw_traces/STEPS3",
):
    sample_names = Utils.autonaming_after_folders(
        sample_0_raw_traces_folder, sample_1_raw_traces_folder
    )

    npeaks = 100
    multi_t = 1000
    multi_y = 1000
    filter = []  # ["n_peaks", 17]
    clear_all_caches = True  # True is used for debugging
    savefig_path = "rallpack3/pics"

    # ##########################################

    logging.info("Process sample 1")

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
                f"peaks_y": [],
                f"peaks_t": [],
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
        clear_refined_traces_cache=clear_all_caches,
    )

    logging.info("Process sample 0")

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
                f"peaks_y": [],
                f"peaks_t": [],
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
        clear_refined_traces_cache=clear_all_caches,
    )

    logging.info("Comparator")

    """Create the comparator for advanced studies

    Note: anywhere is relevant, the first traceDB is considered the benchmark. The others are samples
    """
    comp = Comparator(traceDBs=[sample_1_DB, sample_0_DB])

    logging.info("P values")

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

    logging.info("P value comparator")

    """Create a database"""
    pvalues_traceDB = TraceDB(
        "p values",
        pvalues_traces,
        clear_refined_traces_cache=True,
        save_refined_traces_cache=False,
    )
    comp_pvalues = Comparator(traceDBs=[pvalues_traceDB])

    logging.info("Ks tests")

    """Perform the ks test"""
    for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
        print(tDBnames)
        for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
            for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
                print(t, k, v)

    logging.info("Plots")

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
            # binrange=[npeaks-2.5, npeaks+2.5],
            savefig_path=savefig_path,
            filter=filter,
            xlabel="n peaks",
            pplot=ax[1][i],
        )
        ax[1][i].set_title(f"{'C' if i == 0 else 'D'}\n", loc="left", fontweight="bold")
    fig.tight_layout()
    Figure.savefig(savefig_path=savefig_path, file_name="npeaks_and_freq", fig=fig)
    fig.show()

    neuron_results = [
        [
            [
                1.62,
                16.3,
                30.83,
                45.355,
                59.875,
                74.395,
                88.915,
                103.435,
                117.96,
                132.48,
                147,
                161.52,
                176.04,
                190.565,
                205.085,
                219.605,
                234.125,
                248.645,
            ],
            [
                4.29,
                18.895,
                33.43,
                47.955,
                62.475,
                76.995,
                91.515,
                106.035,
                120.56,
                135.08,
                149.6,
                164.12,
                178.64,
                193.165,
                207.685,
                222.205,
                236.725,
            ],
        ],
        [
            [
                41.4922,
                23.2034,
                22.4812,
                22.427,
                22.424,
                22.4241,
                22.4241,
                22.4236,
                22.423,
                22.4238,
                22.4241,
                22.424,
                22.4236,
                22.423,
                22.4238,
                22.4241,
                22.424,
                22.4236,
            ],
            [
                44.5447,
                46.2315,
                46.1529,
                46.1471,
                46.1474,
                46.1474,
                46.147,
                46.1463,
                46.1469,
                46.1473,
                46.1474,
                46.147,
                46.1462,
                46.1469,
                46.1473,
                46.1474,
                46.147,
            ],
        ],
    ]
    for j, op_tuple in enumerate(
        [("peaks_t", "ms", bindwidth_t), ("peaks_y", "mV", bindwidth_y)]
    ):
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
                title=f"{tracename} {op} avg.-Neuron avg. and std.",
                pplot=ax[1][i],
                mean_offset=neuron_results[j][i],
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
    pvalues_traceDB = TraceDB(
        "p values",
        pvalues_traces,
        clear_refined_traces_cache=True,
        save_refined_traces_cache=False,
    )
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
