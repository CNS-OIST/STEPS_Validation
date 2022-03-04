import logging

from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
from postproc.utils import Utils


logging.basicConfig(level=logging.WARNING)

savefig_path = "caburst/pics"


"""Create the benchmark traces"""
multi = 1000
traces_STEPS3 = []

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        traces_STEPS3.append(
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
traces_STEPS3.append(Trace("t", "ms", multi=multi))


""" create the benchmark database"""
STEPS3_DB = TraceDB(
    "STEPS3",
    traces_STEPS3,
    "caburst/raw_traces/STEPS3",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Create the sample traces"""
multi = 1000
traces_STEPS4 = []

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        traces_STEPS4.append(
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
traces_STEPS4.append(Trace("t", "ms", multi=multi))


"""Create the sample database"""
STEPS4_DB = TraceDB(
    "STEPS4",
    traces_STEPS4,
    "caburst/raw_traces/STEPS4",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Create the comparator for advanced studies"""
comp = Comparator(traceDBs=[STEPS3_DB, STEPS4_DB])


"""Perform the ks test"""
filter = []
for tDBnames, ks_tests in comp.test_ks(filter=filter).items():
    print(tDBnames)
    for t, d in sorted(ks_tests.items(), key=lambda t: Utils.natural_keys(t[0])):
        for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
            print(t, k, v)


"""Plots"""

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        comp.avgplot_raw_traces(
            trace_name=f"{membrane} {op} V",
            conf_lvl=0,
            savefig_path="caburst/pics",
            suffix="",
        )

for membrane in ["smooth", "spiny"]:
    for op in ["max", "min"]:
        comp.avgplot_raw_traces(
            trace_name=f"{membrane} {op} V",
            std=False,
            savefig_path="caburst/pics",
            suffix="",
        )
