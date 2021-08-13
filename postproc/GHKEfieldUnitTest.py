#########
# Still a WIP: this test needs to be updated for the new API and the name changed. Please use the other tests
#########

from postproc.traceDB import TraceDB, Trace
from postproc.comparator import Comparator
import copy
import numpy

"""Create the benchmark traces

Tell the program what is inside the raw data and what aliases you want to perform for each trace
"""
traces_3 = []
traces_3.append(Trace("t", "s"))
for i in ["Na", "D", "E", "Na_surf"]:
    traces_3.append(
        Trace(
            f"{i}_count",
            "n_molecules",
            reduce_ops={
                "amin": [],
                "amax": [],
                "['val', 0.002]": [],
                "['val', 0.005]": [],
                "['val', 0.008]": [],
            },
        )
    )
traces_3.append(
    Trace(
        "V_min",
        "V",
        reduce_ops={
            "amin": [],
            "amax": [],
            "['val', 0.002]": [],
            "['val', 0.005]": [],
            "['val', 0.008]": [],
        },
    )
)
traces_3.append(
    Trace(
        "V_max",
        "V",
        reduce_ops={
            "amin": [],
            "amax": [],
            "['val', 0.002]": [],
            "['val', 0.005]": [],
            "['val', 0.008]": [],
        },
    )
)

"""Create the sample database"""
steps_3 = TraceDB(
    "STEPS 3",
    traces_3,
    "/home/katta/projects/STEPS4Models/GHKEfieldUnitTest/benchmark_STEPS3/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


"""Create the sample traces

Tell the program what is inside the raw data and what aliases you want to perform for each trace
"""
traces_4 = []
traces_4.append(Trace("it", ""))
traces_4.append(Trace("t", "s"))
for i in ["Na", "D", "E", "Na_surf"]:
    traces_4.append(
        Trace(
            f"{i}_count",
            "n_molecules",
            reduce_ops={
                "amin": [],
                "amax": [],
                "['val', 0.002]": [],
                "['val', 0.005]": [],
                "['val', 0.008]": [],
            },
        )
    )
traces_4.append(
    Trace(
        "V_min",
        "V",
        reduce_ops={
            "amin": [],
            "amax": [],
            "['val', 0.002]": [],
            "['val', 0.005]": [],
            "['val', 0.008]": [],
        },
    )
)
traces_4.append(
    Trace(
        "V_max",
        "V",
        reduce_ops={
            "amin": [],
            "amax": [],
            "['val', 0.002]": [],
            "['val', 0.005]": [],
            "['val', 0.008]": [],
        },
    )
)


"""Create the sample database"""
steps_4 = TraceDB(
    "STEPS 4",
    traces_4,
    "/home/katta/projects/STEPS4Models/GHKEfieldUnitTest/sample_STEPS4/results",
    clear_raw_traces_cache=True,
    clear_refined_traces_cache=True,
)


comp = Comparator(benchmark=steps_3, sample=steps_4)

for k, v in sorted(comp.test_ks().items(), key=lambda k: k[0]):
    print(k, v)


for i in ["Na", "D", "E", "Na_surf"]:
    comp.avgplot(
        trace_name=f"{i}_count", std=False, savefig_path="GHKEfieldUnitTest/pics"
    )

for i in ["min", "max"]:
    comp.avgplot(trace_name=f"V_{i}", std=False, savefig_path="GHKEfieldUnitTest/pics")
