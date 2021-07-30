from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy
import numpy

"""Create the sample traces
                
Tell the program what is inside the raw data and what alaiyses you want to perform for each trace
"""
multi = 1
n_expected_peaks = 12
traces_s = []
traces_s.append(Trace("t", "s", multi=multi))

traces_s.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_prominence', {i}]": [] for i in range(n_expected_peaks)},
            **{f"['i_prominence_t', {i}]": [] for i in range(n_expected_peaks)},
            "freq": [],
        },
    )
)


traces_s.append(copy.deepcopy(traces_s[-1]))
traces_s[-1].name = "V_z_max"


# Trace(
#     "dVdt_z_min",
#     "V/s",
#     reduce_ops={
#         "amin": [],
#         "amax": [],
#         "mean": [],
#     },
#     derivation_params=["gradient", ["V_max_on_vertices_z_min", "t"]],
# ),
# Trace(
#     "dVdt_z_max",
#     "V/s",
#     reduce_ops={
#         "amin": [],
#         "amax": [],
#         "mean": [],
#     },
#     derivation_params=["gradient", ["V_max_on_vertices_z_max", "t"]],
# ),
"""Create the sample database"""
sample = TraceDB(
    "STEPS 4",
    traces_s,
    "rallpack3/sample/sampling5e-6",
    clear_raw_traces_cache=False,
    clear_refined_traces_cache=False,
)


"""Create the benchmark traces """
multi = 1e-3
traces_b = []
traces_b.append(Trace("t", "s", multi=multi))

traces_b.append(
    Trace(
        "V_z_min",
        "V",
        multi=multi,
        reduce_ops={
            "amin": [],
            "amax": [],
            **{f"['i_prominence', {i}]": [] for i in range(n_expected_peaks)},
            **{f"['i_prominence_t', {i}]": [] for i in range(n_expected_peaks)},
            "freq": [],
        },
    )
)
traces_b.append(copy.deepcopy(traces_b[-1]))
traces_b[-1].name = "V_z_max"

# sample.plot(savefig_path="rallpack3/pics")
"""Create the benchmark database"""
benchmark = TraceDB(
    "neuron",
    traces_b,
    "rallpack3/benchmark",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Create the comparator for advanced studies"""
comp = Comparator(benchmark=benchmark, sample=sample)

"""Perform the ks test"""
# for k, v in comp.test_ks().items():
#     print(k,v)

# """Compute the mse"""
# for k, v in comp.mse_refactored().items():
#     print(k, *v.items())

"""Plots"""

bindwidth_y = 0.0005
bindwidth_t = 0.0005
bindwidth_Hz = 1


comp.distplot(
    "V_z_min",
    f"freq",
    binwidth=bindwidth_Hz,
    savefig_path="rallpack3/pics",
    suffix="rallpack3",
)
comp.distplot(
    "V_z_max",
    f"freq",
    binwidth=bindwidth_Hz,
    savefig_path="rallpack3/pics",
    suffix="rallpack3",
)

# for i in [0, 1, 9]:
#     comp.distplot(
#         "V_z_min",
#         f"['i_prominence', {i}]",
#         binwidth=bindwidth_y,
#         savefig_path="rallpack3/pics",
#         suffix="rallpack3"
#     )
# for i in [0, 1, 9]:
#     comp.distplot(
#         "V_z_min",
#         f"['i_prominence_t', {i}]",
#         binwidth=bindwidth_t,
#         savefig_path="rallpack3/pics",
#         suffix="rallpack3"
#     )
# for i in [0, 1, 9]:
#     comp.distplot(
#         "V_z_max",
#         f"['i_prominence', {i}]",
#         binwidth=bindwidth_y,
#         savefig_path="rallpack3/pics",
#         suffix="rallpack3"
#     )
# for i in [0, 1, 9]:
#     comp.distplot(
#         "V_z_max",
#         f"['i_prominence_t', {i}]",
#         binwidth=bindwidth_t,
#         savefig_path="rallpack3/pics",
#         suffix="rallpack3"
#     )


#
# sample_raw_trace_idx = 0
#
# comp.plot(
#     trace_name_b="V_max_on_vertices_z_min",
#     raw_trace_idx_s=sample_raw_trace_idx,
#     savefig_path="rallpack3/pics",
# )
# comp.plot(
#     trace_name_b="V_max_on_vertices_z_max",
#     raw_trace_idx_s=sample_raw_trace_idx,
#     savefig_path="rallpack3/pics",
# )
#
# comp.plot(
#     trace_name_b="dVdt_z_min",
#     raw_trace_idx_s=sample_raw_trace_idx,
#     time_trace_name_b="V_max_on_vertices_z_min",
#     time_trace_name_s="V_max_on_vertices_z_min",
#     savefig_path="rallpack3/pics",
# )
# comp.plot(
#     trace_name_b="dVdt_z_max",
#     raw_trace_idx_s=sample_raw_trace_idx,
#     time_trace_name_b="V_max_on_vertices_z_max",
#     time_trace_name_s="V_max_on_vertices_z_max",
#     savefig_path="rallpack3/pics",
# )
