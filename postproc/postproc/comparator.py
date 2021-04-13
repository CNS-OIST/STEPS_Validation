from scipy import stats
import numpy
import matplotlib.pyplot as plt
import os

from .traceDB import TraceDB
from .utils import Utils


class ComparatorError(Exception):
    pass


class Comparator:
    """Comparator

    This class does high-level statistics to determine if 2 samples (benchmark and sample) can come from the same
    population
    """

    def __init__(self, benchmark: TraceDB, sample: TraceDB):
        self.sample = sample
        self.benchmark = benchmark

    def test_ks(self):
        out = []

        for trace_name, trace_b in self.benchmark.traces.items():
            try:
                trace_s = self.sample.traces[trace_name]
                for op, refined_trace_b in trace_b.refined_traces.items():
                    try:
                        refined_trace_s = trace_s.refined_traces[op]

                        out.append(
                            [
                                trace_name,
                                op,
                                stats.ks_2samp(refined_trace_s, refined_trace_b),
                            ]
                        )
                        print(*out[-1])
                    except KeyError:
                        pass
            except KeyError:
                pass
        return out

    def mse(self, normalized=True):
        time_trace_b = self.benchmark.get_time_trace()
        time_trace_s = self.sample.get_time_trace()

        mse_lists = {}
        for trace_name, trace_b in self.benchmark.traces.items():
            if trace_name == time_trace_b.name:
                continue
            trace_s = self.sample.traces[trace_name]

            mse_lists[trace_name] = []
            for file_b, raw_trace_b in trace_b.raw_traces.items():
                raw_time_trace_b = time_trace_b.raw_traces[file_b]
                for file_s, raw_trace_s in trace_s.raw_traces.items():
                    raw_time_trace_s = time_trace_s.raw_traces[file_s]
                    mse, _, _ = Utils.sqrtmse(
                        sample_time_trace=raw_time_trace_s,
                        sample=raw_trace_s,
                        benchmark_time_trace=raw_time_trace_b,
                        benchmark=raw_trace_b,
                    )
                    mse_lists[trace_name].append(mse)

            norm_factor = 1.0
            if normalized:
                try:
                    norm_factor = max(trace_b.refined_traces["amax"]) - min(
                        trace_b.refined_traces["amin"]
                    )
                except KeyError:
                    raise ComparatorError(
                        f"Comparator needs 'amax' and 'amin' for the trace {trace_name} to compute "
                        f"the normalized mse."
                    )
            if norm_factor <= 0:
                raise ComparatorError(
                    f"The norm factor {norm_factor} is <= 0. This should not happen."
                )
            mse_lists[trace_name] = (mse_lists[trace_name], norm_factor)

        out = {}
        for trace_name, (mse_list, norm_factor) in mse_lists.items():
            mse_list = numpy.array(mse_list) / norm_factor

            mean = mse_list.mean()
            amax = numpy.amax(mse_list)
            amin = numpy.amin(mse_list)
            out[trace_name] = {
                "mean": mean,
                "amax": amax,
                "amin": amin,
                "norm_factor": norm_factor,
            }
            print(trace_name, *out[trace_name].items())

        return out

    def plot(
        self,
        trace_name_b,
        trace_name_s=None,
        raw_trace_idx_b=0,
        raw_trace_idx_s=0,
        time_trace_name_s=None,
        time_trace_name_b=None,
        savefig_path=None,
    ):
        if not trace_name_s:
            trace_name_s = trace_name_b
        if not time_trace_name_s:
            time_trace_name_s = time_trace_name_b

        raw_trace_name_b = next(
            iter(self.benchmark.traces[trace_name_b].raw_traces.keys()), raw_trace_idx_b
        )
        raw_trace_name_s = next(
            iter(self.sample.traces[trace_name_s].raw_traces.keys()), raw_trace_idx_s
        )

        time_trace_b = (
            self.benchmark.get_time_trace()
            if not time_trace_name_b
            else self.benchmark.traces[time_trace_name_b]
        )
        time_trace_s = (
            self.sample.get_time_trace()
            if not time_trace_name_s
            else self.sample.traces[time_trace_name_s]
        )

        trace_b = self.benchmark.traces[trace_name_b]
        trace_s = self.sample.traces[trace_name_s]
        print("benchmark:")
        print(trace_b.__str__([raw_trace_idx_b]))
        if time_trace_name_b:
            print(time_trace_b.__str__([raw_trace_idx_b]))
        print("sample:")
        print(trace_s.__str__([raw_trace_idx_s]))
        if time_trace_name_s:
            print(time_trace_s.__str__([raw_trace_idx_s]))

        title = f"{trace_b.name}_vs_{time_trace_b.name}"

        plt.clf()

        plt.plot(
            time_trace_b.raw_traces[raw_trace_name_b],
            trace_b.raw_traces[raw_trace_name_b],
            label="benchmark",
        )

        plt.plot(
            time_trace_s.raw_traces[raw_trace_name_s],
            trace_s.raw_traces[raw_trace_name_s],
            label="sample",
        )
        if not time_trace_name_s and not time_trace_name_b:
            _, interp_time, interp_diff = Utils.sqrtmse(
                sample_time_trace=time_trace_s.raw_traces[raw_trace_name_s],
                sample=trace_s.raw_traces[raw_trace_name_s],
                benchmark_time_trace=time_trace_b.raw_traces[raw_trace_name_b],
                benchmark=trace_b.raw_traces[raw_trace_name_b],
            )
            plt.plot(interp_time, interp_diff, label="diff")
        plt.title(title)
        plt.xlabel(f"{time_trace_b.name} [{time_trace_b.unit}]")
        plt.ylabel(f"{trace_b.name} [{trace_b.unit}]")
        plt.legend()
        if savefig_path:
            plt.savefig(os.path.join(savefig_path, title))
        plt.show()
