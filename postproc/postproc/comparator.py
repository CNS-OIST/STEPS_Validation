from scipy import stats
import numpy
import matplotlib.pyplot as plt
import os
import pandas
import seaborn
import re
import logging


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

    def test_ks(self, traces: set = {}, ops: set = {}):
        """Kolmogorov–Smirnov test """
        out = {}

        if len(traces) == 0:
            traces = self.benchmark.traces.keys() & self.sample.traces.keys()
        else:
            traces = (
                self.benchmark.traces.keys() & self.sample.traces.keys() & set(traces)
            )

        # for trace_name, trace_b in self.benchmark.traces.items():
        for trace_name in traces:
            trace_s = self.sample.traces[trace_name]
            trace_b = self.benchmark.traces[trace_name]

            if len(ops) == 0:
                ops = set(trace_b.refined_traces.keys()) & set(
                    trace_s.refined_traces.keys()
                )
            else:
                ops = (
                    set(trace_b.refined_traces.keys())
                    & set(trace_s.refined_traces.keys())
                    & set(ops)
                )

            for op in ops:
                refined_trace_b = trace_b.refined_traces[op]
                refined_trace_s = trace_s.refined_traces[op]
                out[trace_name + "_" + op] = stats.ks_2samp(
                    refined_trace_s, refined_trace_b
                )

        return out

    def refined_traces_diff(self, normalized=True):

        out = {}
        for trace_name, trace_b in self.benchmark.traces.items():
            for op, refined_trace_b in trace_b.refined_traces.items():
                try:
                    trace_s = self.sample.traces[trace_name]
                    refined_trace_s = trace_s.refined_traces[op]
                except KeyError:
                    continue

                refined_trace_s = numpy.array(refined_trace_s)
                refined_trace_b = numpy.array(refined_trace_b)

                mean_b = numpy.sqrt(numpy.square(refined_trace_b).mean())
                norm_factor = mean_b if mean_b != 0 and normalized else 1

                diff = numpy.array(
                    [
                        abs(i - j) / norm_factor
                        for i in refined_trace_s
                        for j in refined_trace_b
                    ]
                )
                mean = diff.mean()
                amax = numpy.amax(diff)
                amin = numpy.amin(diff)

                if trace_name not in out:
                    out[trace_name] = {}

                out[trace_name][op] = {
                    "mean": mean,
                    "amax": amax,
                    "amin": amin,
                    "norm_factor": norm_factor,
                }

        return out

    def mse_refactored(self, normalized=True):
        """Refactored mean square error

        This is the square root of the mean square error refactored based on a parameter that should be
        representative of the signal as a whole (it is 0 only if the signal is flat)
        """
        time_trace_b = self.benchmark.get_time_trace()
        time_trace_s = self.sample.get_time_trace()

        mse_lists = {}
        for trace_name, trace_b in self.benchmark.traces.items():
            if trace_name == time_trace_b.name:
                continue
            if any(["amax", "amin"]) not in trace_b.refined_traces and normalized:
                logging.warning(
                    f"Comparator needs 'amax' and 'amin' for the trace {trace_name} in benchmark to "
                    f"compute the normalized mse. Trace skipped."
                )
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
                norm_factor = max(trace_b.refined_traces["amax"]) - min(
                    trace_b.refined_traces["amin"]
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

        return out

    def plot(
        self,
        trace_name_b,
        trace_name_s=None,
        raw_trace_idx_b=0,
        raw_trace_idx_s=0,
        time_trace_name_b=None,
        time_trace_name_s=None,
        savefig_path=None,
        suffix="",
    ):
        """Compare traces:

        particular realization of sample vs particular realization of benchmark

        Args:
              - trace_name_b (str): name of the benchmark trace
              - trace_name_s (str, optional): name of the sample trace
              - raw_trace_idx_b (int, optional): idx of the particular benchmark realization
              - raw_trace_idx_s (int, optional): idx of the particular sample realization
              - time_trace_name_b (str, optional): name of the time trace of the benchmark
              - time_trace_name_s (str, optional): name of the time trace of the sample
              - savefig_path (str, optional): path to save the file
        """
        if not trace_name_s:
            trace_name_s = trace_name_b
        if not time_trace_name_s:
            time_trace_name_s = time_trace_name_b

        raw_trace_name_b = list(self.benchmark.traces[trace_name_b].raw_traces.keys())[
            raw_trace_idx_b
        ]
        raw_trace_name_s = list(self.sample.traces[trace_name_s].raw_traces.keys())[
            raw_trace_idx_s
        ]

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
        print(trace_b.__str__(raw_trace_idx_b))
        if time_trace_name_b:
            print(time_trace_b.__str__(raw_trace_idx_b))
        print("sample:")
        print(trace_s.__str__(raw_trace_idx_s))
        if time_trace_name_s:
            print(time_trace_s.__str__(raw_trace_idx_s))

        short_name_b = trace_b.short_name(raw_trace_name_b)
        short_name_s = trace_s.short_name(raw_trace_name_s)

        title = f"{trace_b.name}_b_{short_name_b}_vs_s_{short_name_s}"

        plt.clf()

        plt.plot(
            time_trace_b.raw_traces[raw_trace_name_b],
            trace_b.raw_traces[raw_trace_name_b],
            label=f"benchmark {short_name_b}",
        )

        plt.plot(
            time_trace_s.raw_traces[raw_trace_name_s],
            trace_s.raw_traces[raw_trace_name_s],
            label=f"sample {short_name_s}",
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
            file_name = re.sub(" ", "_", title)
            if suffix:
                file_name += "_" + suffix
            file_name += ".png"
            plt.savefig(os.path.join(savefig_path, file_name))
        plt.show()

    def avgplot(
        self,
        trace_name,
        std=True,
        conf_lvl=0.95,
        savefig_path=None,
        suffix="",
        xlim=None,
        ylim=None,
    ):
        time_trace_b = self.benchmark.get_time_trace()

        time_trace_s = self.sample.get_time_trace()

        trace_b = self.benchmark.traces[trace_name]
        trace_s = self.sample.traces[trace_name]

        avg_b = trace_b.raw_traces.mean(axis=1)
        nt_b = len(trace_b.raw_traces.columns)
        avg_s = trace_s.raw_traces.mean(axis=1)
        nt_s = len(trace_s.raw_traces.columns)

        plt.clf()

        plt.plot(
            time_trace_b.raw_traces.iloc[:, 0],
            avg_b,
            label=f"avg_{self.benchmark.name}_(nt: {nt_b})",
        )
        plt.plot(
            time_trace_s.raw_traces.iloc[:, 0],
            avg_s,
            label=f"avg_{self.sample.name}_(nt: {nt_s})",
        )

        title = f"{trace_name} avg"

        if std:
            title += " std"
            std_b = trace_b.raw_traces.std(axis=1)
            std_s = trace_s.raw_traces.std(axis=1)
            plt.fill_between(
                time_trace_b.raw_traces.iloc[:, 0],
                avg_b + std_b,
                avg_b - std_b,
                label=f"std_{self.benchmark.name}",
                alpha=0.5,
            )
            plt.fill_between(
                time_trace_s.raw_traces.iloc[:, 0],
                avg_s + std_s,
                avg_s - std_s,
                label=f"std_{self.sample.name}",
                alpha=0.5,
            )

        if conf_lvl > 0:
            title += f" conf_int {conf_lvl}"
            conf_int_b = numpy.array(
                list(
                    zip(
                        *trace_b.raw_traces.apply(
                            lambda x: Utils.conf_int(x, conf_lvl), axis=1
                        )
                    )
                )
            )
            conf_int_s = numpy.array(
                list(
                    zip(
                        *trace_s.raw_traces.apply(
                            lambda x: Utils.conf_int(x, conf_lvl), axis=1
                        )
                    )
                )
            )

            plt.fill_between(
                time_trace_b.raw_traces.iloc[:, 0],
                conf_int_b[0],
                conf_int_b[1],
                label=f"conf_int_{self.benchmark.name}",
                alpha=0.3,
            )
            plt.fill_between(
                time_trace_s.raw_traces.iloc[:, 0],
                conf_int_s[0],
                conf_int_s[1],
                label=f"conf_int_{self.sample.name}",
                alpha=0.3,
            )

        plt.legend()
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)

        if savefig_path:
            file_name = re.sub(" ", "_", title)
            if suffix:
                file_name += "_" + suffix
            file_name += ".png"
            plt.savefig(os.path.join(savefig_path, file_name))

        plt.show()

    def diffplot(self, trace_name, savefig_path=None, percent=False, suffix=""):
        """Compare traces:

        plot differences between all the realizations in benchmark vs the ones in sample

        Args:
              - trace_name_b (str): name of the benchmark trace
              - trace_name_s (str, optional): name of the sample trace
              - savefig_path (str, optional): path to save the file
        """

        time_trace_b = self.benchmark.get_time_trace()
        time_trace_s = self.sample.get_time_trace()

        trace_b = self.benchmark.traces[trace_name]
        trace_s = self.sample.traces[trace_name]

        title = f"{trace_name} (b-s)"

        plt.clf()
        plt.title(title)
        plt.xlabel(f"{time_trace_b.name} [{time_trace_b.unit}]")
        plt.ylabel(f"{trace_b.name} [{trace_b.unit}]")
        for raw_trace_name_b in trace_b.raw_traces.keys():
            for raw_trace_name_s in trace_s.raw_traces.keys():
                _, interp_time, interp_diff = Utils.sqrtmse(
                    sample_time_trace=time_trace_s.raw_traces[raw_trace_name_s],
                    sample=trace_s.raw_traces[raw_trace_name_s],
                    benchmark_time_trace=time_trace_b.raw_traces[raw_trace_name_b],
                    benchmark=trace_b.raw_traces[raw_trace_name_b],
                    percent=percent,
                )
                plt.plot(interp_time, interp_diff)

        if savefig_path:
            file_name = re.sub(" ", "_", title)
            if suffix:
                file_name += "_" + suffix
            file_name += ".png"
            plt.savefig(os.path.join(savefig_path, file_name))
        plt.show()

    def distplot(self, trace, op, binwidth=0.005, savefig_path=None, suffix=""):
        """Distribution plot

        Distribution plot comparison of a refined trace between benchmark and sample

        Args:
              - trace (str): trace name
              - op (str): operation performed to refine the raw traces
              - bins=(int, optional): bin number for the distplot
              - savefig_path (str): path to save the figure
        """
        trace_b = self.benchmark.traces[trace].refined_traces[op]
        trace_s = self.sample.traces[trace].refined_traces[op]
        newdf = pandas.DataFrame(
            {
                f"{self.benchmark.name} ({len(trace_b)})": trace_b,
                f"{self.sample.name} ({len(trace_s)})": trace_s,
            }
        )
        p = seaborn.histplot(
            data=newdf, binwidth=binwidth, stat="probability", common_norm=False
        )  # , palette=["grey", "black"]
        title = f"{trace}_{op}"
        p.set_title(title)
        if savefig_path:
            file_name = re.sub(" ", "_", title)
            if suffix:
                file_name += "_" + suffix
            file_name += ".png"
            plt.savefig(os.path.join(savefig_path, file_name))
        plt.show()
