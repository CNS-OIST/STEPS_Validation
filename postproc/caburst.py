from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator

traces = [
    Trace(
        "V",
        "V",
        reduce_ops={
            "2ndspike_timeshift": "caburst/francesco/refined_data/benchmark/tetexact_total_2ndspike_timeshift.txt",
            "amax_timeshift": "caburst/francesco/refined_data/benchmark/tetexact_total_maxvoltage_timeshift.txt",
            "RMS": "caburst/francesco/refined_data/benchmark/tetexact_total_RMS.txt",
            "max_V_shift": "caburst/francesco/refined_data/benchmark/tetexact_total_maxvoltage_shift.txt",
        },
    ),
    Trace(
        "dVdt",
        "V",
        reduce_ops={
            "RMS": "caburst/francesco/refined_data/benchmark/tetexact_total_dVdt_RMS.txt",
        },
    ),
]
benchmark = TraceDB(traces)

traces = [
    Trace(
        "V",
        "V",
        reduce_ops={
            "2ndspike_timeshift": "caburst/francesco/refined_data/sample/tetopsplit_total_2ndspike_timeshift.txt",
            "amax_timeshift": "caburst/francesco/refined_data/sample/tetopsplit_total_maxvoltage_timeshift.txt",
            "RMS": "caburst/francesco/refined_data/sample/tetopsplit_total_RMS.txt",
            "max_V_shift": "caburst/francesco/refined_data/sample/tetopsplit_total_maxvoltage_shift.txt",
        },
    ),
    Trace(
        "dVdt",
        "V",
        reduce_ops={
            "RMS": "caburst/francesco/refined_data/sample/tetopsplit_total_dVdt_RMS.txt",
        },
    ),
]
sample = TraceDB(traces)

# print(benchmark)
#
# traces = [
#     Trace("t", "s"),
#     Trace("smooth_max_V", "V", reduce_ops={"amin":[], "amax":[], "mean":[], "amax_timeshift":[],
#                                            "['i_prominence_timeshift', 1, 0.03]": []}),
#     Trace("smooth_min_V", "V", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("spiny_max_V", "V", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("spiny_min_V", "V", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("smooth_CaP_m0", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("smooth_CaP_m1", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("smooth_CaP_m2", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("smooth_CaP_m3", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("spiny_CaP_m0", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("spiny_CaP_m1", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("spiny_CaP_m2", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("spiny_CaP_m3", "n_mols", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("tot_GHK_curr", "A", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("ohm_curr_smooth_memb_BKchan", "A", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("ohm_curr_spiny_memb_BKchan", "A", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("ohm_curr_smooth_memb_SKchan", "A", reduce_ops={"amin":[], "amax":[], "mean":[]}),
#     Trace("ohm_curr_spiny_memb_SKchan", "A", reduce_ops={"amin":[], "amax":[], "mean":[]}),
# ]

# sample = TraceDB(traces, "caburst/raw_data/sample")
print(sample)
comp = Comparator(benchmark=benchmark, sample=sample)
comp.test_ks()
