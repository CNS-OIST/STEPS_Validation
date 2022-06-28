import copy
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from postproc.comparator import Comparator
from postproc.trace import Trace
from postproc.traceDB import TraceDB
from postproc.utils import Utils

logging.basicConfig(level=logging.INFO)


def check(
    sample_1_raw_traces_folder="rallpack3/raw_traces/STEPS4/ref_2022-06-16_highChannelDensity2",
    sample_0_raw_traces_folder="rallpack3/raw_traces/STEPS3/ref_2022-06-16_highChannelDensity2",
):

    npeaks = 17
    npeaks_focus = min(npeaks, 17)
    multi_t = 1000
    multi_y = 1000
    goodness_of_fit_test_type = "ks"  # possibilities: ks, es, cvm
    filter = []  # ["n_peaks", 17]
    clear_all_caches = False  # True is used for debugging
    savefig_path = "rallpack3/pics"
    with_title = False
    binwidth_y = 0.0005 * multi_y
    binwidth_t = 0.001 * multi_t
    binwidth_Hz = 0.1
    # for batched boxplots
    nbatches = 10

    # ##########################################

    sample_names, sample_0_DB, sample_1_DB = create_base_DBs(
        sample_1_raw_traces_folder,
        sample_0_raw_traces_folder,
        multi_t,
        multi_y,
        npeaks,
        clear_all_caches,
    )

    comp, comp_pvalues, pvalues_traces = create_comparators(
        sample_1_DB, sample_0_DB, goodness_of_fit_test_type, filter
    )

    batched_comp_pvalues = create_batched_comparator(
        comp, npeaks_focus, goodness_of_fit_test_type, nbatches, filter
    )

    Utils.pretty_print_goodness_of_fit(comp, goodness_of_fit_test_type, filter)

    logging.info("Plots")

    """Plots"""
    # plot_npeaks_and_freq(
    #     comp, multi_y, multi_t, binwidth_Hz, npeaks, savefig_path, filter, with_title
    # )

    plot_distributions(comp, binwidth_t, binwidth_y, savefig_path, filter, with_title)

    # plot_boxplot_summary(
    #     pvalues_traces, goodness_of_fit_test_type, savefig_path, with_title
    # )

    plot_avg_and_std(comp, binwidth_t, binwidth_y, npeaks, savefig_path, with_title)

    # plot_missing_spike(multi_t, multi_y, savefig_path)
    #
    plot_batched_p_values(batched_comp_pvalues, npeaks_focus, savefig_path, with_title)

    plot_missing_spike_and_p_values(
        multi_t, multi_y, batched_comp_pvalues, npeaks_focus, savefig_path, with_title
    )


def pvalues_reference(
    goodness_of_fit_test_type, npvalues=100, mean=1, sigma=0.1, size=1000
):
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
        if goodness_of_fit_test_type == "ks":
            pvalue = sp.stats.ks_2samp(Y["Y1"], Y["Y2"]).pvalue
        elif goodness_of_fit_test_type == "es":
            pvalue = sp.stats.epps_singleton_2samp(Y["Y1"], Y["Y2"]).pvalue
        elif goodness_of_fit_test_type == "cvm":
            pvalue = sp.stats.cramervonmises_2samp(Y["Y1"], Y["Y2"]).pvalue
        else:
            raise ValueError(
                f"Unknown goodness of fit test type: {goodness_of_fit_test_type}"
            )
        pvalues.append(pvalue)
    return pvalues


def create_base_DBs(
    sample_1_raw_traces_folder,
    sample_0_raw_traces_folder,
    multi_t,
    multi_y,
    npeaks,
    clear_all_caches,
):

    sample_names = Utils.autonaming_after_folders(
        sample_0_raw_traces_folder, sample_1_raw_traces_folder
    )

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
        save_refined_traces_cache=True,
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
        save_refined_traces_cache=True,
    )

    return sample_names, sample_0_DB, sample_1_DB


def create_missing_spike_DB(multi_t, multi_y):
    """Create the benchmark traces. How do you want to refine the data? Usually exactly like the sample traces"""
    traces = []
    traces.append(Trace("t", "ms", multi=multi_t))

    traces.append(
        Trace(
            "V zmin",
            "mV",
            reduce_ops={},
            multi=multi_y,
        )
    )
    traces.append(copy.deepcopy(traces[-1]))
    traces[-1].name = "V zmax"

    """Create the benchmark database"""
    DB_STEPS4 = TraceDB(
        "STEPS4",
        traces,
        "rallpack3/raw_traces/STEPS4/missing_spike",
        clear_refined_traces_cache=True,
        save_refined_traces_cache=False,
        keep_raw_traces=True,
    )
    return DB_STEPS4


def get_neuron_results():
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
    return neuron_results


def add_ref_to_comp_pvalues(pvalues_traces, goodness_of_fit_test_type):

    """Add pvalue reference and produce the boxplot"""
    pvalues_traces.append(
        Trace(
            "ref",
            "",
            reduce_ops={
                "": pvalues_reference(
                    goodness_of_fit_test_type=goodness_of_fit_test_type
                )
            },
        )
    )
    pvalues_traceDB = TraceDB(
        "p values",
        pvalues_traces,
        clear_refined_traces_cache=True,
        save_refined_traces_cache=False,
    )
    return Comparator(traceDBs=[pvalues_traceDB])


def create_comparators(sample_1_DB, sample_0_DB, goodness_of_fit_test_type, filter):
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
    for tDBnames, tests in comp.test_goodness_of_fit(
        test_type=goodness_of_fit_test_type, filter=filter
    ).items():
        for t, d in tests.items():
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
    return comp, comp_pvalues, pvalues_traces


def create_batched_comparator(
    comp, npeaks_focus, goodness_of_fit_test_type, nbatches, filter
):
    """Batched pvalues"""

    batched_pvalues = {
        "V zmin": {
            **{f"['i_peak_y', {i}]": [] for i in range(npeaks_focus)},
            **{f"['i_peak_t', {i}]": [] for i in range(npeaks_focus)},
        },
        "V zmax": {
            **{f"['i_peak_y', {i}]": [] for i in range(npeaks_focus)},
            **{f"['i_peak_t', {i}]": [] for i in range(npeaks_focus)},
        },
    }

    # the ks tests are our new raw data
    for tDBnames, ks_tests in comp.test_goodness_of_fit(
        test_type=goodness_of_fit_test_type, filter=filter, nbatches=nbatches
    ).items():
        for t, d in ks_tests.items():
            for k, v in d.items():
                if t in batched_pvalues:
                    for k_slim in batched_pvalues[t]:
                        if k_slim in k:
                            batched_pvalues[t][k_slim] = [
                                *batched_pvalues[t][k_slim],
                                *[pp.pvalue for pp in v],
                            ]

    batched_pvalues_traces = [
        Trace(k, "mV", reduce_ops=v) for k, v in batched_pvalues.items()
    ]

    """Create a database"""
    batched_pvalues_traceDB = TraceDB(
        "p values",
        batched_pvalues_traces,
        clear_refined_traces_cache=False,
        save_refined_traces_cache=False,
    )
    batched_comp_pvalues = Comparator(traceDBs=[batched_pvalues_traceDB])
    return batched_comp_pvalues


def plot_npeaks_and_freq(
    comp, multi_y, multi_t, binwidth_Hz, npeaks, savefig_path, filter, with_title
):

    fig, axtot = plt.subplots(2, 2, figsize=(8, 6))
    for i, tracename in enumerate(["V zmin", "V zmax"]):
        ax = axtot[0][i]
        comp.distplot(
            tracename,
            f"['freq', {1/multi_y}, {1/multi_t}]",
            filter=filter,
            ax=ax,
            common_norm=False,
            binwidth=binwidth_Hz,
        )
        ax.set_xlabel("Hz")
        Utils.set_subplot_title(0, i, 2, ax, f"freq {tracename}")

        ax = axtot[1][i]
        comp.distplot(
            tracename,
            f"n_peaks",
            filter=filter,
            ax=ax,
            binwidth=1,
            binrange=[npeaks - 2.5, npeaks + 2.5],
        )
        ax.set_xlabel("n peaks")
        Utils.set_subplot_title(
            1, i, 2, ax, f"n peaks {tracename}" if with_title else None
        )
    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="npeaks_and_freq", fig=fig)
    fig.show()


def plot_distributions(comp, binwidth_t, binwidth_y, savefig_path, filter, with_title):

    fig, axtot = plt.subplots(2, 2, figsize=(8, 6))
    for j, op_tuple in enumerate(
        [("peaks_t", "ms", binwidth_t), ("peaks_y", "mV", binwidth_y)]
    ):
        op, label, binwidth = op_tuple
        for i, tracename in enumerate(["V zmin", "V zmax"]):
            ax = axtot[j][i]
            comp.distplot(
                tracename,
                op,
                filter=filter,
                binwidth=binwidth,
                ax=ax,
            )
            ax.set_xlabel(label)
            if i == 1 and j == 0:
                ax.legend(["STEPS3", "STEPS4"])
            else:
                ax.get_legend().remove()
            Utils.set_subplot_title(
                j, i, 2, ax, f"distribution, {tracename}, {op}" if with_title else None
            )

    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="distributions", fig=fig)
    fig.show()


def plot_boxplot_summary(
    pvalues_traces, goodness_of_fit_test_type, savefig_path, with_title
):
    comp_pvalues = add_ref_to_comp_pvalues(pvalues_traces, goodness_of_fit_test_type)

    fig, ax = plt.subplots(1, 1)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    comp_pvalues.boxplot_refined_traces(ax=ax, color=default_colors[0])
    ax.set_xticklabels(
        [
            "V zmin\npeaks y",
            "V zmin\npeaks t",
            "V zmax\npeaks y",
            "V zmax\npeaks t",
            "ref",
        ]
    )
    ax.set_ylabel("p values")
    if with_title:
        ax.set_title("boxplot p values")
    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="boxplot_p_values", fig=fig)
    fig.show()


def plot_avg_and_std(comp, binwidth_t, binwidth_y, npeaks, savefig_path, with_title):
    neuron_results = get_neuron_results()
    fig, axtot = plt.subplots(2, 2, figsize=(8, 6))

    for j, op_tuple in enumerate(
        [
            ("peaks_t", "ms", binwidth_t, "time stamp"),
            ("peaks_y", "mV", binwidth_y, "height"),
        ]
    ):
        op, label, binwidth, label2 = op_tuple
        for i, tracename in enumerate(["V zmin", "V zmax"]):
            ax = axtot[j][i]
            comp.avgplot_refined_traces(
                tracename,
                [f"['i_{op.replace('s', '')}', {q}]" for q in range(npeaks)],
                mean_offset=neuron_results[j][i],
                ax=ax,
                fmt=["o", "s", "v", "^"],
                capsize=2,
                elinewidth=1,
            )
            ax.set_xlabel("peak n")
            ax.set_ylabel(
                f"Peak {label2} differences at {tracename[2:]}\navg. and std. ({label})"
            )
            if i == 1 and j == 0:
                ax.legend()
            Utils.set_subplot_title(
                j, i, 2, ax, f"avg and std, {tracename}, {op}" if with_title else None
            )
    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="avg_std", fig=fig)
    fig.show()


def plot_batched_p_values(batched_comp_pvalues, npeaks_focus, savefig_path, with_title):

    fig, axtot = plt.subplots(2, 2, figsize=(8, 6))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ylabels_what = [
        "heights",
        "time stamps",
    ]
    ylabels_pos = ["zmin", "zmax"]
    for j, op in enumerate(["peak_y", "peak_t"]):
        for i, tracename in enumerate(["V zmin", "V zmax"]):
            ax = axtot[j][i]
            batched_comp_pvalues.boxplot_refined_traces(
                DB_trace_reduce_ops=[
                    ("p values", tracename, f"['i_{op}', {ip}]")
                    for ip in range(npeaks_focus)
                ],
                ax=ax,
                color=default_colors[0],
            )
            ax.set_xlabel("peak n")

            ax.set_ylabel(f"peak {ylabels_what[j]} p values at {ylabels_pos[i]}")
            Utils.set_subplot_title(
                j,
                i,
                2,
                ax,
                f"batched p values, {tracename}, {op}" if with_title else None,
            )

    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="batched_boxplots", fig=fig)
    fig.show()


def plot_missing_spike(multi_t, multi_y, savefig_path):
    """Create the benchmark database"""
    DB_STEPS4 = create_missing_spike_DB(multi_t, multi_y)
    fig, ax = plt.subplots(1, 1)
    DB_STEPS4.plot(ax=ax, fmt=[[{"linestyle": "-"}], [{"linestyle": "--"}]])
    ax.set_xlabel("ms")
    ax.set_ylabel("mV")
    ax.legend(["V zmin", "V zmax"])
    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="missing_spike", fig=fig)
    fig.show()


def plot_missing_spike_and_p_values(
    multi_t, multi_y, batched_comp_pvalues, npeaks_focus, savefig_path, with_title
):
    """plot missing spike and summary of batched boxplots"""
    fig, axtot = plt.subplots(1, 3, figsize=(10, 3))

    ax = axtot[0]
    DB_missing_spike_STEPS4 = create_missing_spike_DB(multi_t, multi_y)
    DB_missing_spike_STEPS4.plot(
        ax=ax, fmt=[[{"linestyle": "-"}], [{"linestyle": "--"}]]
    )
    ax.set_xlabel("ms")
    ax.set_ylabel("mV")
    ax.set_ylim([-80, 110])
    ax.legend(["V zmin", "V zmax"])
    Utils.set_subplot_title(
        0,
        0,
        3,
        ax,
        f"missing spike" if with_title else None,
    )

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    tracename = "V zmax"

    op = "peak_y"
    ax = axtot[1]
    batched_comp_pvalues.boxplot_refined_traces(
        DB_trace_reduce_ops=[
            ("p values", tracename, f"['i_{op}', {ip}]") for ip in range(npeaks_focus)
        ],
        ax=ax,
        color=default_colors[0],
    )
    ax.set_xlabel("peak n")
    ax.set_ylabel(f"peak heights p values at zmax")
    Utils.set_subplot_title(
        0,
        1,
        3,
        ax,
        f"batched p values, {tracename}, {op}" if with_title else None,
    )

    op = "peak_t"
    ax = axtot[2]
    batched_comp_pvalues.boxplot_refined_traces(
        DB_trace_reduce_ops=[
            ("p values", tracename, f"['i_{op}', {ip}]") for ip in range(npeaks_focus)
        ],
        ax=ax,
        color=default_colors[0],
    )
    ax.set_xlabel("peak n")
    ax.set_ylabel(f"peak time stamps p values at zmax")
    Utils.set_subplot_title(
        0,
        2,
        3,
        ax,
        f"batched p values, {tracename}, {op}" if with_title else None,
    )

    fig.tight_layout()
    Utils.savefig(path=savefig_path, name="missing_spike_and_batched_boxplots", fig=fig)
    fig.show()


if __name__ == "__main__":
    check(*sys.argv[1:])
