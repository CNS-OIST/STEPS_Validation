from scipy import stats
import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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

    def __init__(self, traceDBs):
        # swich to a dict in case we get a list
        self.traceDBs = {v.name: v for v in traceDBs}
        if len(self.traceDBs) != len(traceDBs):
            raise ComparatorError("Different traces have the same name!")

    def _combinatory_map(self, fff, *argv, **kwargs):
        """Apply the function func to all the possible pair combinations in self.traceDBs

        We suppose that A vs B == B vs A or that we are interested only in one of the two comparisons.
        If the order of
        traceDBs
        matters, the most benchmark-like tracesDBs must be put in front
        """
        out = {}
        traceDB_names = list(self.traceDBs.keys())
        for idx_b, benchmarkDB_name in enumerate(traceDB_names):
            for sampleDB_name in traceDB_names[idx_b + 1 :]:
                out[f"{benchmarkDB_name}_vs_{sampleDB_name}"] = fff(
                    benchmarkDB_name, sampleDB_name, *argv, **kwargs
                )
        return out


    def _auto_pic_suffix(self, suffix=""):
        """ Suggest prefix for picture """
        if len(self.traceDBs) == 0 or len(suffix) != 0:
            return suffix

        folders = [os.path.basename(db.folder_path) for db in self.traceDBs.values()]
        names = self.traceDBs.keys()
        if all(x==folders[0] for x in folders):
            return f"{folders[0]}_{'_vs_'.join(names)}"
            return f"{folders[0]}_{'_vs_'.join(names)}"
        else:
            tags = [f"{os.path.basename(v.folder_path)}_{k}" for k, v in self.traceDBs.items()]
            return "_vs_".join(tags)

    def test_ks(self, *argv, **kwargs):
        """ combinatory wrapper"""
        return self._combinatory_map(self._test_ks, *argv, **kwargs)

    def _test_ks(self, benchmarkDB_name, sampleDB_name, filter=None):
        """Kolmogorov–Smirnov test """
        benchmark, sample = (
            self.traceDBs[benchmarkDB_name],
            self.traceDBs[sampleDB_name],
        )

        out = {}

        # we compare only the traces that are shared among the 2 databases
        traces = benchmark.traces.keys() & sample.traces.keys()

        # for trace_name, trace_b in self.benchmark.traces.items():
        for trace_name in traces:
            trace_s = sample.traces[trace_name]
            trace_b = benchmark.traces[trace_name]

            ops = set(trace_b.refined_traces.keys()) & set(
                trace_s.refined_traces.keys()
            )

            if len(ops):
                out[trace_name] = {}

            for op in ops:
                out[trace_name][op] = "no data"

                refined_trace_b = trace_b.filter_refined_trace(op, filter).explode()
                refined_trace_s = trace_s.filter_refined_trace(op, filter).explode()


                if len(refined_trace_b) and len(refined_trace_s):
                    out[trace_name][op] = stats.ks_2samp(
                        refined_trace_s, refined_trace_b
                    )
        return out

    def refined_traces_diff(self, *argv, **kwargs):
        """ combinatory wrapper"""
        return self._combinatory_map(self._refined_traces_diff, *argv, **kwargs)

    def _refined_traces_diff(self, benchmarkDB_name, sampleDB_name, normalized=True):
        """ Compute difference of refined traces """
        benchmark, sample = (
            self.traceDBs[benchmarkDB_name],
            self.traceDBs[sampleDB_name],
        )
        out = {}
        for trace_name, trace_b in benchmark.traces.items():
            for op, refined_trace_b in trace_b.refined_traces.items():
                try:
                    trace_s = sample.traces[trace_name]
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

    def mse_refactored(self, *argv, **kwargs):
        """ combinatory wrapper"""
        return self._combinatory_map(self._mse_refactored, *argv, **kwargs)

    def _mse_refactored(self, benchmarkDB_name, sampleDB_name, normalized=True):
        """Refactored mean square error

        This is the mean square error refactored based on a parameter that should be
        representative of the signal as a whole (it is 0 only if the signal is flat)
        """
        benchmark, sample = (
            self.traceDBs[benchmarkDB_name],
            self.traceDBs[sampleDB_name],
        )

        time_trace_b = benchmark.get_time_trace()
        time_trace_s = sample.get_time_trace()

        mse_lists = {}
        for trace_name, trace_b in benchmark.traces.items():
            if trace_name == time_trace_b.name:
                continue

            if (
                any(mm not in trace_b.refined_traces for mm in ["amax", "amin"])
                and normalized
            ):

                logging.warning(
                    f"Comparator needs 'amax' and 'amin' for the trace {trace_name} in {benchmarkDB_name} to "
                    f"compute the normalized mse. Trace skipped."
                )
                continue

            trace_s = sample.traces[trace_name]

            mse_lists[trace_name] = []
            for file_b, raw_trace_b in trace_b.raw_traces.items():
                raw_time_trace_b = time_trace_b.raw_traces[file_b]
                for file_s, raw_trace_s in trace_s.raw_traces.items():
                    raw_time_trace_s = time_trace_s.raw_traces[file_s]
                    mse, _, _ = Utils.mse(
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

    def plot(self, *argv, **kwargs):
        """ combinatory wrapper"""
        return self._combinatory_map(self._plot, *argv, **kwargs)

    def _plot(
        self,
        benchmarkDB_name,
        sampleDB_name,
        trace_name_b,
        trace_name_s=None,
        raw_trace_idx_b=0,
        raw_trace_idx_s=0,
        time_trace_name_b=None,
        time_trace_name_s=None,
        savefig_path=None,
        suffix="",
        isdiff=True,
        istitle=True,
    ):
        """Compare traces:

        particular realization of sample vs particular realization of benchmark

        Args:
              - traceDB_b (str): trace database name for the benchmark
              - traceDB_s (str): trace database name for the sample
              - trace_name_b (str): name of the benchmark trace
              - trace_name_s (str, optional): name of the sample trace
              - raw_trace_idx_b (int, optional): idx of the particular benchmark realization
              - raw_trace_idx_s (int, optional): idx of the particular sample realization
              - time_trace_name_b (str, optional): name of the time trace of the benchmark
              - time_trace_name_s (str, optional): name of the time trace of the sample
              - savefig_path (str, optional): path to save the file
        """

        suffix = self._auto_pic_suffix(suffix)

        benchmark, sample = (
            self.traceDBs[benchmarkDB_name],
            self.traceDBs[sampleDB_name],
        )

        if not trace_name_s:
            trace_name_s = trace_name_b
        if not time_trace_name_s:
            time_trace_name_s = time_trace_name_b

        raw_trace_name_b = list(benchmark.traces[trace_name_b].raw_traces.keys())[
            raw_trace_idx_b
        ]
        raw_trace_name_s = list(sample.traces[trace_name_s].raw_traces.keys())[
            raw_trace_idx_s
        ]

        time_trace_b = (
            benchmark.get_time_trace()
            if not time_trace_name_b
            else benchmark.traces[time_trace_name_b]
        )
        time_trace_s = (
            sample.get_time_trace()
            if not time_trace_name_s
            else sample.traces[time_trace_name_s]
        )

        trace_b = benchmark.traces[trace_name_b]
        trace_s = sample.traces[trace_name_s]
        print(f"{benchmarkDB_name}:")
        print(trace_b.__str__(raw_trace_idx_b))
        if time_trace_name_b:
            print(time_trace_b.__str__(raw_trace_idx_b))
        print(f"{sampleDB_name}:")
        print(trace_s.__str__(raw_trace_idx_s))
        if time_trace_name_s:
            print(time_trace_s.__str__(raw_trace_idx_s))

        short_name_b = trace_b.short_name(raw_trace_name_b)
        if short_name_b:
            short_name_b = "_" + short_name_b
        short_name_s = trace_s.short_name(raw_trace_name_s)
        if short_name_s:
            short_name_s = "_" + short_name_s

        title = f"{trace_b.name}_{benchmarkDB_name}{short_name_b}_vs_{sampleDB_name}{short_name_s}"

        plt.clf()

        plt.plot(
            time_trace_b.raw_traces[raw_trace_name_b],
            trace_b.raw_traces[raw_trace_name_b],
            label=f"{benchmarkDB_name}{short_name_b}",
        )

        plt.plot(
            time_trace_s.raw_traces[raw_trace_name_s],
            trace_s.raw_traces[raw_trace_name_s],
            "--",
            label=f"{sampleDB_name} {short_name_s}",
        )
        if not time_trace_name_s and not time_trace_name_b and isdiff:
            _, interp_time, interp_diff = Utils.mse(
                sample_time_trace=time_trace_s.raw_traces[raw_trace_name_s],
                sample=trace_s.raw_traces[raw_trace_name_s],
                benchmark_time_trace=time_trace_b.raw_traces[raw_trace_name_b],
                benchmark=trace_b.raw_traces[raw_trace_name_b],
            )
            plt.plot(interp_time, interp_diff, label="diff")
        if istitle:
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

    def diffplot(self, *argv, **kwargs):
        """ combinatory wrapper"""
        return self._combinatory_map(self._diffplot, *argv, **kwargs)

    def _diffplot(
        self,
        benchmarkDB_name,
        sampleDB_name,
        trace_name,
        savefig_path=None,
        percent=False,
        suffix="",
    ):
        """Compare traces:

        plot differences between all the realizations in benchmark vs the ones in sample

        Args:
              - trace_name_b (str): name of the benchmark trace
              - trace_name_s (str, optional): name of the sample trace
              - savefig_path (str, optional): path to save the file
        """

        benchmark, sample = (
            self.traceDBs[benchmarkDB_name],
            self.traceDBs[sampleDB_name],
        )

        time_trace_b = benchmark.get_time_trace()
        time_trace_s = sample.get_time_trace()

        trace_b = benchmark.traces[trace_name]
        trace_s = sample.traces[trace_name]

        title = f"{trace_name} ({benchmarkDB_name}-{sampleDB_name})"

        plt.clf()
        plt.title(title)
        plt.xlabel(f"{time_trace_b.name} [{time_trace_b.unit}]")
        plt.ylabel(f"{trace_b.name} [{trace_b.unit}]")
        for raw_trace_name_b in trace_b.raw_traces.keys():
            for raw_trace_name_s in trace_s.raw_traces.keys():
                _, interp_time, interp_diff = Utils.mse(
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

    def distplot(
        self,
        trace,
        op,
        binwidth=None,
        savefig_path=None,
        suffix="",
        traceDB_names=[],
        filter=None,
    ):
        """Distribution plot

        Distribution plot comparison of a refined trace between benchmark and sample

        Args:
              - trace (str): trace name
              - op (str): operation performed to refine the raw traces
              - bins=(int, optional): bin number for the distplot
              - savefig_path (str): path to save the figure
        """

        suffix = self._auto_pic_suffix(suffix)

        if len(traceDB_names) == 0:
            traceDB_names = self.traceDBs.keys()

        if not any(x in traceDB_names for x in self.traceDBs.keys()):
            logging.warning(
                f"{traceDB_names} are not present in {self.traceDBs.keys()}. Results are empty"
            )

        newdf = pandas.DataFrame(
            {
                f"{k} ({len(v.traces[trace].filter_refined_trace(op, filter))})": v.traces[
                    trace
                ].filter_refined_trace(
                    op, filter
                )
                for k, v in self.traceDBs.items()
                if k in traceDB_names
            }
        )

        newdf = Utils.flatten_data_frame_if_necessary(newdf)


        p = seaborn.histplot(
            data=newdf, binwidth=binwidth, stat="probability", common_norm=False
        )  # , palette=["grey", "black"]

        title = f"{trace}_{op}"
        p.set_title(title)
        plt.xticks(rotation=-30)

        if savefig_path:
            file_name = re.sub(" ", "_", title)
            if suffix:
                file_name += "_" + suffix
            file_name += ".png"
            plt.savefig(os.path.join(savefig_path, file_name))

        plt.show()

    def avgplot_raw_traces(
        self,
        trace_name,
        std=True,
        conf_lvl=0.95,
        savefig_path=None,
        suffix="",
        xlim=None,
        ylim=None,
    ):
        """Average plot with std deviations and confidence bands of the raw traces"""

        suffix = self._auto_pic_suffix(suffix)

        plt.clf()
        for traceDB_name, traceDB in self.traceDBs.items():

            time_trace = traceDB.get_time_trace()
            trace = traceDB.traces[trace_name]

            avg0 = trace.raw_traces.mean(axis=1)
            nt = len(trace.raw_traces.columns)

            plt.plot(
                time_trace.raw_traces.iloc[:, 0],
                avg0,
                label=f"avg_{traceDB_name}_(nt: {nt})",
            )

            if std:
                std0 = trace.raw_traces.std(axis=1)
                plt.fill_between(
                    time_trace.raw_traces.iloc[:, 0],
                    avg0 + std0,
                    avg0 - std0,
                    label=f"std_{traceDB_name}",
                    alpha=0.5,
                )

            if conf_lvl > 0:
                conf_int = numpy.array(
                    list(
                        zip(
                            *trace.raw_traces.apply(
                                lambda x: Utils.conf_int(x, conf_lvl), axis=1
                            )
                        )
                    )
                )

                plt.fill_between(
                    time_trace.raw_traces.iloc[:, 0],
                    conf_int[0],
                    conf_int[1],
                    label=f"conf_int_{traceDB_name}",
                    alpha=0.3,
                )

        title = f"{trace_name} avg"
        if std:
            title += " std"
        if conf_lvl:
            title += f" conf_int {conf_lvl}"

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

    def avgplot_refined_traces(
        self,
        trace_name,
        reduce_ops=[],
        savefig_path = "",
        suffix = ""
    ):
        suffix = self._auto_pic_suffix(suffix)
        plt.clf()
        plt.figure()
        xx = [*range(len(reduce_ops))]
        shift = 0.2*numpy.array([*range(len(self.traceDBs))])
        shift -= shift.mean()

        common_prefix = os.path.commonprefix(reduce_ops)

        ticknames = [i[len(common_prefix):] for i in reduce_ops]
        for idx, (db_name, db) in enumerate(self.traceDBs.items()):
            if trace_name not in db.traces:
                continue

            rt = db.traces[trace_name].refined_traces
            avgs = numpy.array([rt[op].mean() if op in rt else numpy.NaN for op in reduce_ops])
            std = numpy.array([rt[op].std() if op in rt else numpy.NaN for op in reduce_ops])
            plt.errorbar(xx-shift[idx], avgs, yerr=std, fmt='o', capsize=2, elinewidth=1, label=db_name)

        plt.xticks(xx, ticknames, rotation='30')
        title = f"{trace_name} avg std {common_prefix}"

        plt.legend()
        plt.title(title)

        if savefig_path:
            file_name = re.sub(" ", "_", title)
            if suffix:
                file_name += "_" + suffix
            file_name += ".png"
            plt.savefig(os.path.join(savefig_path, file_name))


        plt.show()