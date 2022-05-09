import inspect
import logging
import os

import numpy
import pandas
import scipy
import seaborn

from .figure import Figure
from .utils import Utils


class TraceError(Exception):
    pass


class Trace:
    """Trace class

    This class keeps track of all the realizations of a trace (i.e. the voltage trace) over many simulations. It is
    meant to be built progressively:

    - We initialize the trace telling the program what refinements and operations we want to do on the class.
    - We import the trace from files or we derive the trace from other traces in TraceDB.
    - We refine the trace to get the statistics that are then available for further processing.

    The main parts of this class are 2:
    - raw_traces: is an ordered_dict that sorts the traces extracted from the files by file name
    - refined_traces: statistics performed over the many realizations of the simulation

    """

    def __init__(self, name, unit, multi=1.0, reduce_ops={}, derivation_params=[]):
        """Init

        Args:
            - name (str): trace name
            - unit (str): measurement unit
            - multi (float): when the trace is extracted from raw data (or derived from other strings) it is
            multiplied by this
            - reduce_ops (dict): statistics computed on all the raw traces (i.e. max_prominence etc.). It can be any
            function in Utils, numpy or scipy provided that the keys can be called as arguments for these functions.
            If the value is a string it is interpreted as a filepath and this refined statistic is extracted from a
            file directly without raw_data refining (using still multi)
            - derivation_params (list): if this list is not empty the trace is considered derived from other
            raw_traces. It is not extracted with the other root traces and it is derived from the traces using the
            functions in Utils. The list provides the arguments to this function
        """
        self.name = name
        self.unit = unit
        self.multi = multi
        self.raw_traces = pandas.DataFrame()
        self.derivation_params = derivation_params
        self.refined_traces = pandas.DataFrame(reduce_ops)
        self.raw_traces_common_prefix_len = 0
        self.raw_traces_common_suffix_len = 0

    def _update_raw_traces_common_prefix_and_suffix(self):
        """Service function to compute trimmmable raw_data file names"""
        self.raw_traces_common_prefix_len = len(
            os.path.commonprefix(list(self.raw_traces.keys()))
        )
        self.raw_traces_common_suffix_len = len(
            os.path.commonprefix([i[::-1] for i in self.raw_traces.keys()])
        )

    def short_name(self, name):
        """Compute trimmed (without shared parts) raw_data file name"""
        return name[
            self.raw_traces_common_prefix_len : len(name)
            - self.raw_traces_common_suffix_len
        ]

    def __str__(self, raw_trace_idx=None):
        """Print

        Args:
              - raw_trace_idxs (list): since self.raw_traces are many; we let the possibility to print only a
              selection of them indicated by
        """

        ss = f"name: {self.name}\n"
        ss += f"unit: {self.unit}\n"
        ss += f"multi: {self.multi}\n"
        if raw_trace_idx:
            ss += f"raw_traces:\n{next(iter(self.raw_traces), raw_trace_idx)}\n"
        else:
            ss += f"raw_traces:\n{self.raw_traces}\n"
        ss += f"refined_traces:\n{self.refined_traces}"

        return ss

    def _check_for_nans(self):
        """ Check for nans in the trace """
        for k, v in self.raw_traces.items():
            if v.isnull().values.any():
                raise TraceError(
                    f"The file {k} presents nan values for the trace {self.name}"
                )

    def filter_refined_trace(self, op, filter=None):
        """ Apply filter to refined traces """
        if not filter:
            return self.refined_traces[op]
        else:
            t = self.refined_traces[op]
            return t.loc[self.refined_traces[filter[0]] == filter[1]].dropna()

    def raw_traces_to_parquet(self, path):
        """ Save raw traces to parquet """
        self.raw_traces.to_parquet(os.path.join(path, f"{self.name}.parquet"))

    def raw_traces_from_parquet(self, path):
        """ Load raw traces from parquet """
        self.raw_traces = pandas.read_parquet(
            os.path.join(path, f"{self.name}.parquet")
        )

    def refined_traces_to_parquet(self, path):
        """ Save refined traces to parquet """
        self.refined_traces.to_parquet(os.path.join(path, f"{self.name}.parquet"))

    def refined_traces_from_parquet(self, path):
        """ Load refined traces from parquet """
        if not len(self.refined_traces.columns):
            return

        data = pandas.read_parquet(os.path.join(path, f"{self.name}.parquet"))

        if not self.raw_traces.empty and len(self.raw_traces.columns) != len(data):
            raise TraceError(
                f"The number of raw_traces ({len(self.raw_traces.columns)}) does not match the number "
                f"of loaded refined trace raws ({len(data.index)})"
            )

        if set(self.refined_traces.columns) != set(data.columns):
            raise TraceError(
                "The refined traces requested have changed, the cache is invalid"
            )

        self.refined_traces = data

    def append_to_raw_trace(self, file_path, val):
        """Service function to append to a raw_trace"""
        val = val * self.multi
        if file_path in self.raw_traces:
            self.raw_traces[file_path].append(val)
        else:
            self.raw_traces[file_path] = [val]

    def add_raw_trace(self, file_path, trace):
        self.raw_traces[file_path] = self.multi * trace

    @staticmethod
    def add_to_raw_trace_list(df, trace_list):
        """Add pandas DataFrame to a list of traces

        Useful when you want to pass data from a simulation directly to postprocessing without saving to file
        """
        for i in trace_list:
            try:
                i.add_raw_trace("", df[i.name])
            except KeyError:
                logging.warning(
                    f"The Trace {i.name} is not present in the DataFrame keys {df.keys()}. Skipped."
                )
                pass

    def derive_raw_data(self, traces):
        """Derive the current self.raw_traces from other traces

        Using the other traces provided we evaluate the expression provided in derivation_params.
        derivation_params has the following structure:

        [operation, [trace_names], other_arguments]

        and the derivation is computed so that:

        operation(*[i for i in traces[trace_names]], *other_arguments)

        Args:
            - traces (dict): Complete dictionary of traces from TraceDB
        """
        if not self.derivation_params:
            return

        [op, root_traces, *args] = self.derivation_params

        for file in traces[root_traces[0]].raw_traces:
            input_traces = [traces[i].raw_traces[file] for i in root_traces]

            if hasattr(Utils, op) and callable(func := getattr(Utils, op)):
                derived_trace = func(*input_traces, *args)
            elif hasattr(numpy, op) and callable(func := getattr(numpy, op)):
                derived_trace = func(*input_traces, *args)
            elif hasattr(scipy, op) and callable(func := getattr(scipy, op)):
                derived_trace = func(*input_traces, *args)
            else:
                raise TraceError(
                    f"The derived trace {self.name} presents an unknown operation {op}."
                )

            self.raw_traces[file] = self.multi * numpy.array(derived_trace)

    def refine(self, time_trace={}):
        """Fill refined_traces using evaluating the keys as member functions in Utils, numpy or scipy

        The refine_traces key can be a str or a list of arguments. In case of a str, it is evaluated as a member
        function of Utils, numpy or scipy. In case of a list, the first term is the member function while the others
        are arguments to be fed after the traces.
        """

        logging.info(f"Refining {self.name}")

        for op_key in self.refined_traces.columns:
            tr = []
            if len(self.refined_traces[op_key]) == 1 and isinstance(
                self.refined_traces[op_key][0], str
            ):
                f = open(self.refined_traces[op_key][0])
                for l in f.readlines():
                    try:
                        tr.append(self.multi * float(l))
                    except:
                        pass

            else:

                for file, trace in self.raw_traces.items():

                    logging.info(f"Processing {file}")

                    op_args = []
                    op = op_key
                    try:
                        l = eval(op_key)
                        op = l[0]
                        op_args = l[1:]
                    except NameError:
                        pass

                    if hasattr(Utils, op) and callable(func := getattr(Utils, op)):
                        if (
                            "time_trace"
                            not in inspect.getfullargspec(getattr(Utils, op)).args
                        ):
                            val = func(trace, *op_args)
                        else:
                            val = func(
                                trace, time_trace.raw_traces[file], *op_args
                            )
                    elif hasattr(numpy, op) and callable(func := getattr(numpy, op)):
                        val = func(trace, *op_args)
                    elif hasattr(scipy, op) and callable(func := getattr(scipy, op)):
                        val = func(trace, *op_args)
                    else:
                        raise TraceError(
                            f"The derived trace {self.name} presents an unknown operation {op}."
                        )

                    tr.append(val)

            self.refined_traces[op_key] = tr

    def plot(
        self, time_trace, trace_files=[], title_prefix="", legend=False, *argv, **kwargs
    ):
        """Plot all raw_traces"""

        ff = Figure(*argv, **kwargs)

        if not trace_files:
            trace_files = self.raw_traces.keys()
        else:
            trace_files = numpy.array(self.raw_traces.keys())[trace_files]

        title = title_prefix + " " + self.name

        for file in trace_files:
            trace = self.raw_traces[file]
            t = time_trace.raw_traces[file]
            ff.pplot.plot(t, trace, label=f"trace {self.short_name(file)}")

        ff.set_title(title)
        ff.set_xlabel(time_trace.unit)
        ff.set_ylabel(self.unit)

        ff.finalize(with_legend=legend)

    def distplot(self, op, *argv, **kwargs):
        """ Plot refined traces in a histogram """
        ff = Figure(*argv, **kwargs)
        seaborn.histplot(data=self.refined_traces, x=op, ax=ff.pplot)
        ff.finalize()
