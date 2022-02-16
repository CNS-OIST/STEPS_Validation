from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
from postproc.utils import Utils


import logging

logging.basicConfig(level=logging.WARNING)

savefig_path = "caburst/pics"


"""Create the benchmark traces"""
multi = 1000
traces_b = []

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        traces_b.append(
            Trace(
                f"{membrane} {op} V",
                "mV",
                multi=multi,
                reduce_ops={
                    "amin": [],
                    "amax": [],
                    "['i_peak_t', 0]": [],
                    "['i_peak_y', 0]": [],
                    # "['i_peak_y', 1]": [],
                    "['val', 0.038]": [],
                    "n_peaks": [],
                },
            )
        )
traces_b.append(Trace("t", "ms", multi=multi))


""" create the benchmark database"""
benchmark_STEPS3 = TraceDB(
    "STEPS3",
    traces_b,
    "caburst/benchmark_STEPS3/results/spack_downgrade_20220124",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Create the sample traces"""
"""Create the benchmark traces"""
multi = 1000
traces_s = []

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        traces_s.append(
            Trace(
                f"{membrane} {op} V",
                "mV",
                multi=multi,
                reduce_ops={
                    "amin": [],
                    "amax": [],
                    "['i_peak_t', 0]": [],
                    "['i_peak_y', 0]": [],
                    # "['i_peak_y', 1]": [],
                    "['val', 0.038]": [],
                    "n_peaks": [],
                },
            )
        )
traces_s.append(Trace("t", "ms", multi=multi))


"""Create the sample database"""
sample_STEPS4 = TraceDB(
    "STEPS4",
    traces_s,
    "caburst/sample_STEPS4/results/pr709_oldspack_20220215",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


# """Create the comparator for advanced studies"""
comp = Comparator(traceDBs=[sample_STEPS4, benchmark_STEPS3])


"""Perform the ks test"""
filter = []
for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
    print(tDBnames)
    for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
        for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
            print(t, k, v)
# """Compute the mse"""
# for tDBnames, mse_refactored in comp.mse_refactored(normalized=False).items():
#     print(tDBnames)
#     for k, v in sorted(mse_refactored.items(), key=lambda k: k[0]):
#         print(k, *v.items())

"""Plots"""

# for membrane in ["smooth", "spiny"]:
#     for op in ["max", "min"]:
#         comp.avgplot_raw_traces(
#             trace_name=f"{membrane} {op} V",
#             conf_lvl=0,
#             savefig_path="caburst/pics",
#             suffix="",
#         )

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        comp.avgplot_raw_traces(
            trace_name=f"{membrane} {op} V",
            std=False,
            savefig_path="caburst/pics",
            suffix="",
            # xlim=[24, 26],
            # ylim=[-30, -20],
            # title=title
        )

# membrane = "smooth"
# op = "max"
# comp.distplot(
#     f"{membrane} {op} V",
#     f"['val', 0.038]",
#     binwidth=0.0002,
#     savefig_path=savefig_path,
#     # xlabel="V",
#     filter=filter,
# )
