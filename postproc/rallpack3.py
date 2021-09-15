from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
from postproc.utils import Utils
import copy
import numpy


npeaks = 17

"""Create the sample traces

Tell the program what is inside the raw data and what aliases you want to perform for each trace
"""
multi = 1
traces_sample_STEPS4 = []
traces_sample_STEPS4.append(Trace("t", "s", multi=multi))

traces_sample_STEPS4.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
            **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
            "freq": [],
            "n_peaks": [],
        },
    )
)
traces_sample_STEPS4.append(copy.deepcopy(traces_sample_STEPS4[-1]))
traces_sample_STEPS4[-1].name = "V_z_max"

"""Create the sample database"""
sample_STEPS4 = TraceDB(
    "STEPS4",
    traces_sample_STEPS4,
    # "rallpack3/sample_STEPS4/results",
    "rallpack3/sample_STEPS4/results/ef_dt_1e-6_rtol_1e-8_1000",
    clear_raw_traces_cache=False,
    clear_refined_traces_cache=False,
)

# ##########################################

"""Create the benchmark STEPS 3 traces """
multi = 1
traces_benchmark_STEPS3 = []
traces_benchmark_STEPS3.append(Trace("t", "s", multi=1))

traces_benchmark_STEPS3.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
            **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
            "freq": [],
            "n_peaks": [],
        },
    )
)
traces_benchmark_STEPS3.append(copy.deepcopy(traces_benchmark_STEPS3[-1]))
traces_benchmark_STEPS3[-1].name = "V_z_max"

"""Create the benchmark database"""
benchmark_STEPS3 = TraceDB(
    "STEPS3",
    traces_benchmark_STEPS3,
    # "rallpack3/benchmark_STEPS3/results",
    "rallpack3/benchmark_STEPS3/results/1000",
    clear_raw_traces_cache=False,
    clear_refined_traces_cache=False,
)


##########################################

# """Create the benchmark NEURON traces """
# multi = 1e-3
# traces_benchmark_NEURON = []
# traces_benchmark_NEURON.append(Trace("t", "s", multi=multi))
#
# traces_benchmark_NEURON.append(
#     Trace(
#         "V_z_min",
#         "V",
#         multi=multi,
#         reduce_ops={
#             "amin": [],
#             "amax": [],
#             **{f"['i_peak_y', {i}]": [] for i in range(npeaks)},
#             **{f"['i_peak_t', {i}]": [] for i in range(npeaks)},
#             "freq": [],
#             "n_peaks": [],
#         },
#     )
# )
# traces_benchmark_NEURON.append(copy.deepcopy(traces_benchmark_NEURON[-1]))
# traces_benchmark_NEURON[-1].name = "V_z_max"
#
# """Create the benchmark database"""
# benchmark_NEURON = TraceDB(
#     "NEURON",
#     traces_benchmark_NEURON,
#     "rallpack3/benchmark_NEURON/results",
#     clear_raw_traces_cache=True,
#     clear_refined_traces_cache=True,
# )

##########################################


"""Create the comparator for advanced studies"""
comp = Comparator(traceDBs=[benchmark_STEPS3, sample_STEPS4])


"""Perform the ks test"""
for tDBnames, ks_tests in comp.test_ks(filter=["n_peaks", 17]).items():
    print(tDBnames)
    for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
        for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
            print(t, k, v)

for tDBnames, ks_tests in comp.test_ks(filter=["n_peaks", 17]).items():
    print(f"{tDBnames}, not passing the KS test")
    for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
        for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
            if v.pvalue < 0.05:
                print(t, k, v)


# this can take some time. Commented for now
# """Compute the mse"""
# for tDBnames, mse_tests in comp.mse_refactored().items():
#     print(tDBnames)
#     for k, v in sorted(mse_tests.items(), key=lambda k: k[0]):
#         print(k, *v.items())


"""Plots"""

bindwidth_y = 0.0005
bindwidth_t = 0.001
# bindwidth_t = 0.00001
bindwidth_Hz = 1

suffix = "STEPS3_vs_STEPS4_ef_dt_1e-6_rtol_1e-8"
savefig_path = "rallpack3/pics"
comp.distplot(
    "V_z_min",
    f"freq",
    binwidth=bindwidth_Hz,
    savefig_path=savefig_path,
    suffix=suffix,
    filter=["n_peaks", 17],
)


comp.distplot(
    "V_z_max",
    f"freq",
    binwidth=bindwidth_Hz,
    savefig_path=savefig_path,
    suffix=suffix,
    filter=["n_peaks", 17],
)

comp.distplot(
    "V_z_min",
    f"n_peaks",
    binwidth=1,
    savefig_path=savefig_path,
    suffix=suffix,
)

comp.distplot(
    "V_z_max",
    f"n_peaks",
    binwidth=1,
    savefig_path=savefig_path,
    suffix=suffix,
)


for i in range(npeaks):
    comp.distplot(
        "V_z_min",
        f"['i_peak_y', {i}]",
        binwidth=bindwidth_y,
        savefig_path=savefig_path,
        suffix=suffix,
        # traceDB_names=["STEPS3", "STEPS4"],
        filter=["n_peaks", 17],
    )
    comp.distplot(
        "V_z_min",
        f"['i_peak_t', {i}]",
        binwidth=bindwidth_t,
        savefig_path=savefig_path,
        suffix=suffix,
        # traceDB_names=["STEPS3", "STEPS4"],
        filter=["n_peaks", 17],
    )
    comp.distplot(
        "V_z_max",
        f"['i_peak_y', {i}]",
        binwidth=bindwidth_y,
        savefig_path=savefig_path,
        suffix=suffix,
        # traceDB_names=["STEPS3", "STEPS4"],
        filter=["n_peaks", 17],
    )
    comp.distplot(
        "V_z_max",
        f"['i_peak_t', {i}]",
        binwidth=bindwidth_t,
        savefig_path=savefig_path,
        suffix=suffix,
        # traceDB_names=["STEPS3", "STEPS4"],
        filter=["n_peaks", 17],
    )

########################
"""p value statistics and graphs"""
pvalues = {
    "V_z_min": {
        "i_peak_y": [],
        "i_peak_t": [],
    },
    "V_z_max": {
        "i_peak_y": [],
        "i_peak_t": [],
    },
}
for tDBnames, ks_tests in comp.test_ks(filter=["n_peaks", 17]).items():
    for t, d in ks_tests.items():
        for k, v in d.items():
            if t in pvalues:
                for k_slim in pvalues[t]:
                    if k_slim in k:
                        pvalues[t][k_slim].append(v.pvalue)

pvalues_traces = [Trace(k, "V", reduce_ops=v) for k, v in pvalues.items()]

"""Create a database"""
pvalues_traceDB = TraceDB("pvalues", pvalues_traces, is_refine=False)

comp_pvalues = Comparator(traceDBs=[pvalues_traceDB])

for tname in ["V_z_max", "V_z_min"]:
    for op in ["i_peak_y", "i_peak_t"]:
        comp_pvalues.distplot(
            tname,
            op,
            binwidth=0.1,
            savefig_path=savefig_path,
            suffix=suffix,
        )
