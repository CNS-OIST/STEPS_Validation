import logging

import matplotlib.pyplot as plt

from postproc.comparator import Comparator
from postproc.figure import Figure
from postproc.traceDB import TraceDB, Trace
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

# fig, ax = plt.subplots(2, 2)
# subplot_label = "A"
# for i, membrane in enumerate(["smooth", "spiny"]):
#     for j, op in enumerate(["max", "min"]):
#         comp.avgplot_raw_traces(
#             trace_name=f"{membrane} {op} V",
#             conf_lvl=0,
#             savefig_path="caburst/pics",
#             suffix="",
#             pplot=ax[i][j],
#             legendfontsize=5,
#         )
#         ax[i][j].set_title(subplot_label + "\n", loc="left", fontweight="bold")
#         subplot_label = chr(ord(subplot_label) + 1)
# fig.tight_layout()
# Figure.savefig(savefig_path=savefig_path, file_name="avg_and_std", fig=fig)
# fig.show()

fig, ax = plt.subplots(3, 2, figsize=(9, 10))
subplot_label = "A"
legendfontsize = 8
for i, membrane in enumerate(["smooth", "spiny"]):
    for j, op in enumerate(["min", "max"]):
        comp.avgplot_raw_traces(
            trace_name=f"{membrane} {op} V",
            std=False,
            savefig_path="caburst/pics",
            suffix="",
            pplot=ax[i][j],
            legendfontsize=legendfontsize,
        )
        ax[i][j].set_title(subplot_label + "\n", loc="left", fontweight="bold")
        subplot_label = chr(ord(subplot_label) + 1)

comp.avgplot_raw_traces(
    trace_name=f"spiny min V",
    std=False,
    savefig_path="caburst/pics",
    suffix="",
    title=r"Focus of panel C",
    pplot=ax[2][0],
    legendfontsize=legendfontsize,
    xlim=[35, 37],
    ylim=[-39, -41],
)
ax[2][0].set_title(subplot_label + "\n", loc="left", fontweight="bold")
fig.delaxes(ax[2][1])
subplot_label = chr(ord(subplot_label) + 1)
fig.tight_layout()
Figure.savefig(savefig_path=savefig_path, file_name="avg_and_conf_int", fig=fig)
fig.show()
