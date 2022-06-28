import logging
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy
import pandas
import pandas as pd
import seaborn
from scipy import stats

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
            raise ComparatorError("Different trace DBs have the same name!")

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

    def _DB_trace_ops_combinations(self):
        """Returns a list of all the combinations of DB, trace, and reduce operation that are possible with the
        current data"""
        out = []
        for db_name, db in self.traceDBs.items():
            for trace_name, trace in db.traces.items():
                for reduce_ops in trace.refined_traces:
                    out.append((db_name, trace_name, reduce_ops))

        return out

    def _auto_pic_suffix(self, suffix):
        """Suggest prefix for picture"""
        if len(self.traceDBs) == 0 or suffix is not None:
            return suffix

        folders = [os.path.basename(db.folder_path) for db in self.traceDBs.values()]
        names = self.traceDBs.keys()
        if all(x == folders[0] for x in folders):
            return f"{folders[0]}_{'_vs_'.join(names)}"
            return f"{folders[0]}_{'_vs_'.join(names)}"
        else:
            tags = [
                f"{os.path.basename(v.folder_path)}_{k}"
                for k, v in self.traceDBs.items()
            ]
            return "_vs_".join(tags)

    def test_goodness_of_fit(self, *argv, **kwargs):
        """combinatory wrapper"""
        return self._combinatory_map(self._test_goodness_of_fit, *argv, **kwargs)

    def _test_goodness_of_fit(
        self, benchmarkDB_name, sampleDB_name, test_type, filter=None, nbatches=1
    ):
        """Kolmogorov–Smirnov test

        Args:
            - benchmarkDB_name: first sample name
            - sampleDB_name: second sample name
            - test type: the supported test types are:
                - "ks": Kolmogorov-Smirnov test for goodness of fit
                - "es": Epps-Singleton test for goodness of fit
                - "cvm": Cramér-von Mises test for goodness of fit
            - filter: to remove data before computation
            - nbatches: if you want to batch the data in subgroups and compute the pvalues for all the possible
            combinations. For example, if you have 100 runs for sample 1 and 100 for sample 2 and you divide with 10
            sections you will get 10*10 = 100 p_values
        """

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

                for ib0 in numpy.array_split(refined_trace_b, nbatches):
                    for is0 in numpy.array_split(refined_trace_s, nbatches):
                        if len(ib0) and len(is0):

                            val = -1
                            if test_type == "ks":
                                val = stats.ks_2samp(is0, ib0)
                            elif test_type == "es":
                                try:
                                    val = stats.epps_singleton_2samp(
                                        is0.tolist(), ib0.tolist()
                                    )  # for testing
                                except numpy.linalg.LinAlgError:
                                    logging.warning(
                                        f"The goodness of fit test Epps-Singleton failed for {trace_name} {op}. Probably "
                                        f"the "
                                        f"SVD did not converge."
                                    )
                            elif test_type == "cvm":
                                val = stats.cramervonmises_2samp(is0, ib0)
                            else:
                                raise ValueError(
                                    f"Unknown goodness of fit test type: {test_type}"
                                )

                            if type(out[trace_name][op]) is str:
                                out[trace_name][op] = [val]
                            else:
                                out[trace_name][op].append(val)

        return out

    def refined_traces_diff(self, *argv, **kwargs):
        """combinatory wrapper"""
        return self._combinatory_map(self._refined_traces_diff, *argv, **kwargs)

    def _refined_traces_diff(self, benchmarkDB_name, sampleDB_name, normalized=True):
        """Compute difference of refined traces"""
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
        """combinatory wrapper"""
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
        """combinatory wrapper"""
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
        isdiff=True,
        ax=plt,
        *argv,
        **kwargs,
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
        """

        benchmark, sample = (
            self.traceDBs[benchmarkDB_name],
            self.traceDBs[sampleDB_name],
        )

        if not trace_name_s:
            trace_name_s = trace_name_b
        if not time_trace_name_s:
            time_trace_name_s = time_trace_name_b

        raw_trace_names_b = list(benchmark.traces[trace_name_b].raw_traces.keys())
        raw_trace_name_b = raw_trace_names_b[raw_trace_idx_b]
        raw_trace_names_s = list(sample.traces[trace_name_s].raw_traces.keys())
        raw_trace_name_s = raw_trace_names_s[raw_trace_idx_s]

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

        short_raw_trace_names_b = Utils.pretty_print_combinations(raw_trace_names_b)
        short_name_b = short_raw_trace_names_b[raw_trace_idx_b]
        if short_name_b:
            short_name_b = "_" + short_name_b
        short_raw_trace_names_s = Utils.pretty_print_combinations(raw_trace_names_s)
        short_name_s = short_raw_trace_names_s[raw_trace_idx_s]
        if short_name_s:
            short_name_s = "_" + short_name_s

        ax.plot(
            time_trace_b.raw_traces[raw_trace_name_b],
            trace_b.raw_traces[raw_trace_name_b],
            label=f"{benchmarkDB_name}",
            *argv,
            **kwargs,
        )

        ax.plot(
            time_trace_s.raw_traces[raw_trace_name_s],
            trace_s.raw_traces[raw_trace_name_s],
            "--",
            label=f"{sampleDB_name}",
            *argv,
            **kwargs,
        )
        if not time_trace_name_s and not time_trace_name_b and isdiff:
            _, interp_time, interp_diff = Utils.mse(
                sample_time_trace=time_trace_s.raw_traces[raw_trace_name_s],
                sample=trace_s.raw_traces[raw_trace_name_s],
                benchmark_time_trace=time_trace_b.raw_traces[raw_trace_name_b],
                benchmark=trace_b.raw_traces[raw_trace_name_b],
            )
            ax.plot(interp_time, interp_diff, label="diff")

    def diffplot(self, *argv, **kwargs):
        """combinatory wrapper"""
        return self._combinatory_map(self._diffplot, *argv, **kwargs)

    def _diffplot(
        self,
        benchmarkDB_name,
        sampleDB_name,
        trace_name,
        percent=False,
        *argv,
        **kwargs,
    ):
        """Compare traces:

        plot differences between all the realizations in benchmark vs the ones in sample

        Args:
              - trace_name_b (str): name of the benchmark trace
              - trace_name_s (str, optional): name of the sample trace
              - savefig_path (str, optional): path to save the file
        """
        ff = Figure(*argv, **kwargs)

        benchmark, sample = (
            self.traceDBs[benchmarkDB_name],
            self.traceDBs[sampleDB_name],
        )

        time_trace_b = benchmark.get_time_trace()
        time_trace_s = sample.get_time_trace()

        trace_b = benchmark.traces[trace_name]
        trace_s = sample.traces[trace_name]

        for raw_trace_name_b in trace_b.raw_traces.keys():
            for raw_trace_name_s in trace_s.raw_traces.keys():
                _, interp_time, interp_diff = Utils.mse(
                    sample_time_trace=time_trace_s.raw_traces[raw_trace_name_s],
                    sample=trace_s.raw_traces[raw_trace_name_s],
                    benchmark_time_trace=time_trace_b.raw_traces[raw_trace_name_b],
                    benchmark=trace_b.raw_traces[raw_trace_name_b],
                    percent=percent,
                )
                ff.pplot.plot(interp_time, interp_diff)

        title = ff.set_title(f"{trace_name} ({benchmarkDB_name}-{sampleDB_name})")

        xlabel = ff.set_xlabel(f"{time_trace_b.name} [{time_trace_b.unit}]")
        ylabel = ff.set_ylabel(f"{trace_b.name} [{trace_b.unit}]")
        ff.finalize()

    def distplot(
        self,
        trace,
        op,
        traceDB_names=[],
        filter=[],
        ax=plt,
        *argv,
        **kwargs,
    ):
        """Distribution plot

        Distribution plot comparison of a refined trace between benchmark and sample

        Args:
              - trace (str): trace name
              - op (str): operation performed to refine the raw traces
        """

        if len(traceDB_names) == 0:
            traceDB_names = list(self.traceDBs.keys())

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

        seaborn.histplot(
            data=newdf,
            ax=ax,
            *argv,
            **kwargs,
        )

    def avgplot_raw_traces(
        self,
        trace_name,
        std=True,
        conf_lvl=0.95,
        ax=plt,
        plot_kwargs={},
        std_fill_between_kwargs={},
        conf_int_fill_between_kwargs={},
    ):
        """Average plot with std deviations and confidence bands of the raw traces"""

        for traceDB_name, traceDB in self.traceDBs.items():

            if not traceDB.keep_raw_traces:
                raise ComparatorError(
                    f"The database {traceDB.name} has `keep_raw_traces` set to False. This is "
                    f"incompatible with `avgplot_raw_traces`. Change that to plot."
                )

            time_trace = traceDB.get_time_trace()
            trace = traceDB.traces[trace_name]

            avg0 = trace.raw_traces.mean(axis=1)
            nt = len(trace.raw_traces.columns)

            ax.plot(
                time_trace.raw_traces.iloc[:, 0],
                avg0,
                label=f"avg. {traceDB_name} (nt: {nt})",
                **plot_kwargs,
            )

            if std:
                std0 = trace.raw_traces.std(axis=1)
                ax.fill_between(
                    time_trace.raw_traces.iloc[:, 0],
                    avg0 + std0,
                    avg0 - std0,
                    label=f"std. {traceDB_name}",
                    **std_fill_between_kwargs,
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

                ax.fill_between(
                    time_trace.raw_traces.iloc[:, 0],
                    conf_int[0],
                    conf_int[1],
                    label=f"conf. int. {traceDB_name}",
                    **conf_int_fill_between_kwargs,
                )

    def avgplot_refined_traces(
        self,
        trace_name,
        reduce_ops=[],
        mean_offset=None,
        fmt=["-"],
        ax=plt.gca(),
        *argv,
        **kwargs,
    ):
        """Average plot with std deviations and confidence bands of the refined traces"""

        ndata = len(reduce_ops)
        if type(mean_offset) is list:
            ndata = min(ndata, len(mean_offset))

        xx = [*range(ndata)]
        shift = 0.2 * numpy.array([*range(len(self.traceDBs))])
        shift -= shift.mean()

        if type(fmt) == list:
            fmt = cycle(fmt)

        common_prefix = Utils.common_prefix(reduce_ops)
        common_suffix = Utils.common_suffix(reduce_ops)

        ticknames = [
            i[len(common_prefix) : -len(common_suffix)] for i in reduce_ops[:ndata]
        ]
        for idx, (db_name, db) in enumerate(self.traceDBs.items()):
            if trace_name not in db.traces:
                continue

            rt = db.traces[trace_name].refined_traces

            if type(mean_offset) is int or type(mean_offset) is float:
                avgs = numpy.array([mean_offset] * ndata)
            else:
                avgs = numpy.array(
                    [
                        rt[op].mean() if op in rt else numpy.NaN
                        for op in reduce_ops[:ndata]
                    ]
                )

                if type(mean_offset) is list:
                    avgs -= numpy.array(mean_offset[: len(reduce_ops)])

            std = numpy.array(
                [rt[op].std() if op in rt else numpy.NaN for op in reduce_ops[:ndata]]
            )

            ax.errorbar(
                xx - shift[idx],
                avgs,
                yerr=std,
                fmt=next(fmt),
                label=db_name,
                *argv,
                **kwargs,
            )
            ax.set_xticks(range(len(ticknames)))
            ax.set_xticklabels(ticknames)

    def boxplot_refined_traces(
        self, DB_trace_reduce_ops=None, ax=plt.gca(), *argv, **kwargs
    ):
        """Box plot (scientific candlestick plot)"""

        if DB_trace_reduce_ops is None:
            DB_trace_reduce_ops = self._DB_trace_ops_combinations()

        ticknames = Utils.pretty_print_combinations(DB_trace_reduce_ops)

        df = {}
        for idx, (db_name, trace_name, op) in enumerate(DB_trace_reduce_ops):
            try:
                trace = self.traceDBs[db_name].traces[trace_name].refined_traces[op]
            except KeyError:
                logging.warning(
                    f"The trace in {(db_name, trace_name, op)} was not found. Skipped"
                )
                continue

            df[ticknames[idx]] = trace

        seaborn.boxplot(data=pd.DataFrame(df), ax=ax, *argv, **kwargs)
