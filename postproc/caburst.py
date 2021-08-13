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
                    "['i_peak_y', 1]": [],
                    "['val', 0.05]": [],
                    "n_peaks": [],
                },
            )
        )


""" create the benchmark database"""
benchmark_STEPS3 = TraceDB(
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
                    "['i_peak_y', 1]": [],
                    "['val', 0.05]": [],
                    "n_peaks": [],
                },
            )
        )


"""Create the sample database"""
sample_STEPS4 = TraceDB(
    "STEPS4",
    traces_s,
    "caburst/sample/benchmark_32nodes",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


# """Create the comparator for advanced studies"""
comp = Comparator(traceDBs=[sample_STEPS4, benchmark_STEPS3])


"""Perform the ks test"""

for tDBnames, ks_tests in comp.test_ks().items():
    print(tDBnames)
    for k, v in sorted(ks_tests.items(), key=lambda k: k[0]):
        print(k, v)
"""Compute the mse"""
for tDBnames, mse_refactored in comp.mse_refactored().items():
    print(tDBnames)
    for k, v in sorted(mse_refactored.items(), key=lambda k: k[0]):
        print(k, *v.items())

"""Plots"""

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
