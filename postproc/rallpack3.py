from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy

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
            "n_peaks": [],
            "max_prominence": [],
            "['i_prominence', 1]": [],
            "freq2": [],
            "freq3": [],
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
            "n_peaks": [],
            "max_prominence": [],
            "['i_prominence', 1]": [],
            "freq2": [],
            "freq3": [],
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

traces_b = copy.deepcopy(traces_s)
for idx, _ in enumerate(traces_b[0:2]):
    traces_b[idx].multi = 1e-3

sample = TraceDB(traces_s, "rallpack3/raw_data/sample")
# print(sample)
benchmark = TraceDB(traces_b, "rallpack3/raw_data/benchmark")
# print(benchmark)


comp = Comparator(benchmark=benchmark, sample=sample)

sample_raw_trace_idx = 5
# comp.plot(
#     trace_name_b="dVdt_z_min",
#     raw_trace_idx_s=sample_raw_trace_idx,
#     time_trace_name_b="V_z_min",
#     time_trace_name_s="V_z_min",
#     savefig_path="rallpack3/pics",
# )
comp.plot(
    trace_name_b="V_z_min",
    raw_trace_idx_s=sample_raw_trace_idx,
    # time_trace_name_b="V_z_max",
    # time_trace_name_s="V_z_max",
    savefig_path="rallpack3/pics",
)

#

# comp.plot(
#     trace_name_b="dVdt_z_min",
#     raw_trace_name_s=sample_raw_trace_idx,
#     savefig_path="rallpack3/pics",
# )
# comp.plot(
#     trace_name_b="dVdt_z_max",
#     raw_trace_name_s=sample_raw_trace_idx,
#     savefig_path="rallpack3/pics",
# )
#
# comp.test_ks()
# comp.mse()
