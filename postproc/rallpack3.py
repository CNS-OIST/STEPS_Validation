from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
from postproc.utils import Utils
import copy
import os

npeaks = 17
savefig_path = "rallpack3/pics"
multi = 1000

# ##########################################

"""Create the benchmark traces. How do you want to refine the data? Usually exactly like the sample traces"""
traces_benchmark = []
traces_benchmark.append(Trace("t", "ms", multi=multi))

traces_benchmark.append(
    Trace(
        "V zmin",
        "mV",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
            **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
            "freq": [],
            "n_peaks": [],
            "peaks_y": [],
            "peaks_t": [],
        },
    )
)
traces_benchmark.append(copy.deepcopy(traces_benchmark[-1]))
traces_benchmark[-1].name = "V zmax"

"""Create the benchmark database"""
benchmarkDB = TraceDB(
    "STEPS3",
    traces_benchmark,
    "rallpack3/benchmark_STEPS3/results/pr708_oldspack_20220131",
    clear_raw_traces_cache=False,
    clear_refined_traces_cache=False,
)

"""Create the sample traces. How do you want to refine the data?"""
trace_sample = []
trace_sample.append(Trace("t", "ms", multi=multi))
trace_sample.append(
    Trace(
        "V zmin",
        "mV",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
            **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
            "freq": [],
            "n_peaks": [],
            "peaks_y": [],
            "peaks_t": [],
        },
    )
)
trace_sample.append(copy.deepcopy(trace_sample[-1]))
trace_sample[-1].name = "V zmax"

"""Create the sample database"""
sampleDB = TraceDB(
    "STEPS4",
    trace_sample,
    "rallpack3/sample_STEPS4/results/pr709_oldspack_20220215",
    clear_raw_traces_cache=False,
    clear_refined_traces_cache=False,
)

"""Create the comparator for advanced studies

Note: anywhere is relevant, the first traceDB is considered the benchmark. The others are samples
"""
comp = Comparator(traceDBs=[benchmarkDB, sampleDB])


# filter data out
filter = []  # ["n_peaks", 17]

"""Perform the ks test"""
for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
    print(tDBnames)
    for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
        for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
            print(t, k, v)

# this can take some time. Commented for now
# """Compute the mse"""
# for tDBnames, mse_tests in comp.mse_refactored().items():
#     print(tDBnames)
#     for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
#         print(k, *v.items())

"""Plots"""

bindwidth_y = 0.0005 * multi
bindwidth_t = 0.001 * multi
bindwidth_Hz = 1


for tracename in ["V zmin", "V zmax"]:
    for op_tuple in [("peaks_t", "ms", bindwidth_t), ("peaks_y", "mV", bindwidth_y)]:
        op, xlabel, bindwidth = op_tuple
        comp.distplot(
            tracename,
            op,
            binwidth=bindwidth,
            savefig_path=savefig_path,
            filter=filter,
            xlabel=xlabel,
        )
        ### this works only if we use standard units: s, V
        # comp.distplot(
        #     tracename,
        #     f"freq",
        #     binwidth=bindwidth_Hz,
        #     savefig_path=savefig_path,
        #     xlabel="Hz",
        #     filter=filter,
        # )
        comp.distplot(
            tracename,
            f"n_peaks",
            binwidth=1,
            binrange=[12.5, 19.5],
            savefig_path=savefig_path,
            filter=filter,
            xlabel="n peaks",
        )


########################

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


for tname in ["V zmax", "V zmin"]:
    for op in ["i_peak_y", "i_peak_t"]:
        comp_pvalues.distplot(
            tname,
            op,
            binwidth=0.1,
            binrange=[0, 1],
            xlabel="p values",
            savefig_path=savefig_path,
        )

comp.avgplot_refined_traces(
    "V zmin",
    [f"['i_peak_t', {i}]" for i in range(npeaks)],
    xlabel="peak_{n}",
    ylabel="ms",
    savefig_path=savefig_path,
    title=f"Peaks t avg. and std. deviation",
)
comp.avgplot_refined_traces(
    "V zmax",
    [f"['i_peak_t', {i}]" for i in range(npeaks)],
    xlabel="peak n",
    ylabel="ms",
    savefig_path=savefig_path,
    title=f"Peaks t avg. and std. deviation",
)
comp.avgplot_refined_traces(
    "V zmin",
    [f"['i_peak_y', {i}]" for i in range(npeaks)],
    xlabel="peak n",
    ylabel="mV",
    savefig_path=savefig_path,
    title=f"Peaks y avg. and std. deviation",
)
comp.avgplot_refined_traces(
    "V zmax",
    [f"['i_peak_y', {i}]" for i in range(npeaks)],
    xlabel="peak n",
    ylabel="mV",
    savefig_path=savefig_path,
    title=f"Peaks y avg. and std. deviation",
)
