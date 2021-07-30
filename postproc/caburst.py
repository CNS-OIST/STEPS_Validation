from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator


import logging

logging.basicConfig(level=logging.WARNING)


"""Create the benchmark traces"""
multi = 1
traces_b = []
traces_b.append(Trace("t", "s", multi=multi))
for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        traces_b.append(
            Trace(
                f"{membrane}_{op}_V",
                "V",
                multi=multi,
                reduce_ops={
                    "amin": [],
                    "amax": [],
                    "max_prominence_t": [],
                    "['i_prominence_y', 1]": [],
                    "['val', 0.05]": [],
                    "n_peaks": [],
                },
            )
        )

    # traces_b[1].refined_traces["['i_prominence_t', 1, 0.01]"] = []


""" create the benchmark database"""
benchmark = TraceDB(
    "STEPS3",
    traces_b,
    "caburst/benchmark/test_100",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Create the sample traces"""
multi = 1
traces_s = []
traces_s.append(Trace("it", "", multi=multi))
traces_s.append(Trace("t", "s", multi=multi))
for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        traces_s.append(Trace(f"{membrane}_{op}_V_on_verts", "V", multi=multi))
for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        traces_s.append(
            Trace(
                f"{membrane}_{op}_V",
                "V",
                multi=multi,
                reduce_ops={
                    "amin": [],
                    "amax": [],
                    "max_prominence_t": [],
                    "['i_prominence_y', 1]": [],
                    "['val', 0.05]": [],
                    "n_peaks": [],
                },
            )
        )


"""Create the sample database"""
sample = TraceDB(
    "STEPS4",
    traces_s,
    "caburst/sample/benchmark_32nodes",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


# """Create the comparator for advanced studies"""
comp = Comparator(benchmark=benchmark, sample=sample)


# """Refined traces diff"""
# for trace_name, res in comp.refined_traces_diff().items():
#     for op, v in res.items():
#         print(trace_name, op, *v.items())

"""Perform the ks test"""

for k, v in sorted(comp.test_ks().items(), key=lambda k: k[0]):
    print(k, v)
"""Compute the mse"""
for k, v in sorted(comp.mse_refactored().items(), key=lambda k: k[0]):
    print(k, *v.items())
# """Plots"""
#
# binwidth_y = 0.0001
# binwidth_t = 0.0001
# for membrane in ["smooth", "spiny"]:
#     for op in ["max", "min"]:
#         comp.distplot(f"{membrane}_{op}_V", "['i_prominence_y', 1]", binwidth=binwidth_y, savefig_path="caburst/pics",suffix="")
# for t in [0.01, 0.02, 0.03, 0.04, 0.05]:
#     comp.distplot(f"{membrane}_{op}_V", f"['val', {t}]", binwidth=binwidth_y, savefig_path="caburst/pics",
#                   suffix="no_GHK")
# comp.distplot(f"{membrane}_{op}_V", "amin", binwidth=binwidth_y, savefig_path="caburst/pics",
#               suffix="no_GHK")
# comp.distplot(f"{membrane}_{op}_V", "amax", binwidth=binwidth_y, savefig_path="caburst/pics",
#               suffix="no_GHK")
# comp.distplot(f"{membrane}_{op}_V", "max_prominence_t", binwidth=binwidth_t, savefig_path="caburst/pics",
#               suffix="no_GHK")


# for membrane in ["smooth", "spiny"]:
#     for op in ["max", "min"]:
#         comp.diffplot(
#             trace_name=f"{membrane}_{op}_V", savefig_path="caburst/pics")

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        comp.avgplot(
            trace_name=f"{membrane}_{op}_V",
            conf_lvl=0,
            savefig_path="caburst/pics",
            suffix="",
        )
for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        comp.avgplot(
            trace_name=f"{membrane}_{op}_V",
            std=False,
            savefig_path="caburst/pics",
            suffix="",
        )


# sample_raw_trace_idx =0
#
# for membrane in ["smooth", "spiny"]:
#     for op in ["max", "min"]:
#         comp.plot(
#             trace_name_b=f"{membrane}_{op}_V",
#             raw_trace_idx_b=0,
#             raw_trace_idx_s=sample_raw_trace_idx,
#             savefig_path="caburst/pics",
#         )
