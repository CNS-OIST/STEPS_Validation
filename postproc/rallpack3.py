from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy

"""Create the sample traces
                
Tell the program what is inside the raw data and what alaiyses you want to perform for each trace
"""
multi = 1
traces_s = [
    Trace("t", "s", multi=multi),
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            "mean": [],
            "max_prominence": [],
            "['i_prominence', 1]": [],
            "freq2": [],
            "freq3": [],
            "['i_prominence_timeshift', 0]": [],
            "['i_prominence_timeshift', 1]": [],
            "['i_prominence_timeshift', 2]": [],
            "['i_prominence_timeshift', 3]": [],
            "['i_prominence_timeshift', 4]": [],
            "['i_prominence_timeshift', 5]": [],
            "['i_prominence_timeshift', 6]": [],
            "['i_prominence_timeshift', 7]": [],
            "['i_prominence_timeshift', 8]": [],
            "['i_prominence_timeshift', 9]": [],
            "['i_prominence_timeshift', 10]": [],
            "['i_prominence_timeshift', 11]": [],
            "['i_prominence_timeshift', 12]": [],
            "['i_prominence_timeshift', 13]": [],
            "['i_prominence_timeshift', 14]": [],
        },
    ),
    Trace(
        "V_z_max",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            "mean": [],
            "max_prominence": [],
            "['i_prominence', 1]": [],
            "freq2": [],
            "freq3": [],
            "['i_prominence_timeshift', 0]": [],
            "['i_prominence_timeshift', 1]": [],
            "['i_prominence_timeshift', 2]": [],
            "['i_prominence_timeshift', 3]": [],
            "['i_prominence_timeshift', 4]": [],
            "['i_prominence_timeshift', 5]": [],
            "['i_prominence_timeshift', 6]": [],
            "['i_prominence_timeshift', 7]": [],
            "['i_prominence_timeshift', 8]": [],
            "['i_prominence_timeshift', 9]": [],
            "['i_prominence_timeshift', 10]": [],
            "['i_prominence_timeshift', 11]": [],
            "['i_prominence_timeshift', 12]": [],
            "['i_prominence_timeshift', 13]": [],
            "['i_prominence_timeshift', 14]": [],
        },
    ),
    Trace(
        "dVdt_z_min",
        "V/s",
        reduce_ops={
            "amin": [],
            "amax": [],
            "mean": [],
        },
        derivation_params=["gradient", ["V_z_min", "t"]],
    ),
    Trace(
        "dVdt_z_max",
        "V/s",
        reduce_ops={
            "amin": [],
            "amax": [],
            "mean": [],
        },
        derivation_params=["gradient", ["V_z_max", "t"]],
    ),
]

"""Create the benchmark traces

We need to deepcopy the traces and change the muti parameter becausse in neuron things are measured in mV and ms
"""
traces_b = copy.deepcopy(traces_s)
for idx, _ in enumerate([i for i in traces_b if not i.derivation_params]):
    traces_b[idx].multi = 1e-3

"""Create the sample database"""
sample = TraceDB(traces_s, "rallpack3/raw_data/sample/sampling5e-6")
sample.plot(savefig_path="rallpack3/pics")
"""Create the benchmark database"""
benchmark = TraceDB(traces_b, "rallpack3/raw_data/benchmark")
benchmark.plot(savefig_path="rallpack3/pics")

"""Create the comparator for advanced studies"""
comp = Comparator(benchmark=benchmark, sample=sample)

"""Perform the ks test"""
comp.test_ks()
"""Compute the mse"""
comp.mse_refactored()

"""Plots"""
comp.distplot("V_z_min", "['i_prominence_timeshift', 0]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 1]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 2]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 3]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 4]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 5]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 6]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 7]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 8]", savefig_path="rallpack3/pics")
comp.distplot("V_z_min", "['i_prominence_timeshift', 9]", savefig_path="rallpack3/pics")
comp.distplot(
    "V_z_min", "['i_prominence_timeshift', 10]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_min", "['i_prominence_timeshift', 11]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_min", "['i_prominence_timeshift', 12]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_min", "['i_prominence_timeshift', 13]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_min", "['i_prominence_timeshift', 14]", savefig_path="rallpack3/pics"
)

comp.distplot("V_z_max", "['i_prominence_timeshift', 0]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 1]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 2]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 3]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 4]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 5]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 6]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 7]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 8]", savefig_path="rallpack3/pics")
comp.distplot("V_z_max", "['i_prominence_timeshift', 9]", savefig_path="rallpack3/pics")
comp.distplot(
    "V_z_max", "['i_prominence_timeshift', 10]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_max", "['i_prominence_timeshift', 11]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_max", "['i_prominence_timeshift', 12]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_max", "['i_prominence_timeshift', 13]", savefig_path="rallpack3/pics"
)
comp.distplot(
    "V_z_max", "['i_prominence_timeshift', 14]", savefig_path="rallpack3/pics"
)


sample_raw_trace_idx = 0

comp.plot(
    trace_name_b="V_z_min",
    raw_trace_idx_s=sample_raw_trace_idx,
    savefig_path="rallpack3/pics",
)
comp.plot(
    trace_name_b="V_z_max",
    raw_trace_idx_s=sample_raw_trace_idx,
    savefig_path="rallpack3/pics",
)

comp.plot(
    trace_name_b="dVdt_z_min",
    raw_trace_idx_s=sample_raw_trace_idx,
    time_trace_name_b="V_z_min",
    time_trace_name_s="V_z_min",
    savefig_path="rallpack3/pics",
)
comp.plot(
    trace_name_b="dVdt_z_max",
    raw_trace_idx_s=sample_raw_trace_idx,
    time_trace_name_b="V_z_max",
    time_trace_name_s="V_z_max",
    savefig_path="rallpack3/pics",
)
