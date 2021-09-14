import os
import numpy
import scipy
import seaborn
import matplotlib.pyplot as plt
import pandas
import shutil
import inspect
import logging

import re

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
        for k, v in self.raw_traces.items():
            if v.isnull().values.any():
                raise TraceError(
                    f"The file {k} presents nan values for the trace {self.name}"
                )

    def filter_refined_trace(self, op, filter=None):
        if not filter:
            return self.refined_traces[op]
        else:
            t = self.refined_traces[op]
            return t.loc[self.refined_traces[filter[0]] == filter[1]].dropna()

    def raw_traces_to_parquet(self, path):
        self.raw_traces.to_parquet(os.path.join(path, f"{self.name}.parquet"))

    def raw_traces_from_parquet(self, path):
        self.raw_traces = pandas.read_parquet(
            os.path.join(path, f"{self.name}.parquet")
        )

    def refined_traces_to_parquet(self, path):
        self.refined_traces.to_parquet(os.path.join(path, f"{self.name}.parquet"))

    def refined_traces_from_parquet(self, path):

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
                f"The refined traces requested have changed, the cache is invalid"
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
                raise TraceDBError(
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
                            val = getattr(Utils, op)(trace, *op_args)
                        else:
                            val = getattr(Utils, op)(
                                trace, time_trace.raw_traces[file], *op_args
                            )
                    elif hasattr(numpy, op) and callable(func := getattr(numpy, op)):
                        val = getattr(numpy, op)(trace, *op_args)
                    elif hasattr(scipy, op) and callable(func := getattr(scipy, op)):
                        val = getattr(scipy, op)(trace, *op_args)
                    else:
                        raise TraceDBError(
                            f"The derived trace {self.name} presents an unknown operation {op}."
                        )
                    tr.append(val)

            self.refined_traces[op_key] = tr

    def plot(
        self,
        time_trace,
        trace_files=[],
        savefig_path=None,
        title_prefix="",
        legend=False,
        suffix="",
    ):
        """Plot all raw_traces"""

        if not trace_files:
            trace_files = self.raw_traces.keys()
        else:
            trace_files = numpy.array(self.raw_traces.keys())[trace_files]

        title = title_prefix + "_" + self.name

        plt.clf()
        for file in trace_files:
            trace = self.raw_traces[file]
            t = time_trace.raw_traces[file]
            plt.plot(t, trace, label=f"trace_{self.short_name(file)}")

        plt.title(title)
        plt.xlabel(time_trace.unit)
        plt.ylabel(self.unit)

        if legend:
            plt.legend()

        if savefig_path:
            file_name = "".join([c for c in title if re.match(r"\w", c)])
            if suffix:
                file_name += "_" + suffix
            plt.savefig(os.path.join(savefig_path, file_name))
        plt.show()

    def distplot(self, op):
        seaborn.histplot(data=self.refined_traces, x=op)
        plt.show()


class TraceDBError(Exception):
    pass


class TraceDB:
    """Trace DataBase

    This class maintains all the traces of a particular simulation type (sample or benchmark)
    """

    def __init__(
        self,
        name,
        traces,
        folder_path="",
        time_trace="t",
        clear_raw_traces_cache=False,
        clear_refined_traces_cache=False,
    ):
        """Init

        Args:
              - traces: traces that must be fed with the raw data found in folder_path
              - folder_path (str): path to where the raw data are. The raw data must be in the form of
              a matrix where each column is a trace (in the order specified in traces). More information in
              _extract_raw_data
              - time_trace specified a particular trace as the time trace
        """
        self.name = name
        self.time_trace = time_trace
        self.root_traces = [i.name for i in traces if not i.derivation_params]
        self.traces = {i.name: i for i in traces}
        if len(traces) != len(self.traces):
            raise TraceDBError(
                "Some traces have the same name. Please make them unique"
            )

        self.folder_path = folder_path

        self._extract_all_raw_data(clear_raw_traces_cache)

        self._check_for_nans()

        self._refine(clear_refined_traces_cache)

        for i in self.traces:
            self.traces[i]._update_raw_traces_common_prefix_and_suffix()

    def get_time_trace(self):
        return self.traces.get(self.time_trace, None)

    def __str__(self, interesting_traces=[], is_full_display=False):
        if is_full_display:
            pandas.set_option("display.max_rows", None, "display.max_columns", None)

        ss = f"folder_path: {self.folder_path}\n"
        if not interesting_traces:
            interesting_traces = self.traces.keys()
        elif isinstance(interesting_traces, str):
            interesting_traces = [interesting_traces]
        for k in interesting_traces:
            ss += str(self.traces[k]) + "\n"
        return ss

    def __len__(self):
        return len(self.traces)

    def rewrite_raw_traces(self, suffix="_1"):
        """Rewrite raw trace files after they were loaded

        Useful to convert dimensions
        """

        if len(suffix) == 0:
            cont = input(
                "I am going to re-write the raw traces. Are you sure you want to proceed? [y/n] (This "
                "operation cannot "
                "be reversed)"
            )

            if cont != "y":
                logging.warning("Aborted")
                return

            logging.warning("I am going to rewrite the raw traces")

        for file_path in next(iter(self.traces.values())).raw_traces.keys():
            df = pandas.DataFrame(
                {k: v.raw_traces[file_path] for k, v in self.traces.items()}
            )
            base_file_path, ext = os.path.splitext(file_path)
            final_path = f"{base_file_path}{suffix}{ext}"
            logging.warning(f"rewriting {final_path} ...")
            df.to_csv(final_path, sep=" ", index=False)

    def _derive_raw_data(self):
        """Service function to call derive on all traces"""
        for trace_name in self.traces:
            self.traces[trace_name].derive_raw_data(self.traces)

    def _check_for_nans(self):
        """Check for nans in the raw traces"""
        for i in self.traces.values():
            i._check_for_nans()

    def _refine(self, clear_cache=False):
        """Service function that calls refine on all the traces"""

        cache_path = os.path.join(self.folder_path, "refined_traces_cache/")

        if clear_cache:
            try:
                shutil.rmtree(cache_path)
            except OSError:
                pass

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        for trace_name in self.traces:
            try:
                self.traces[trace_name].refined_traces_from_parquet(cache_path)
            except:
                self.traces[trace_name].refine(self.get_time_trace())
                self.traces[trace_name].refined_traces_to_parquet(cache_path)

    def _extract_all_raw_data(self, clear_cache=False):
        """Extract all raw data from the files in self.folder_path (not recursive) """

        if not self.folder_path:
            return

        cache_path = os.path.join(self.folder_path, "raw_traces_cache/")

        if clear_cache:
            try:
                shutil.rmtree(cache_path)
            except OSError:
                pass

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        try:
            for k in self.traces.keys():
                self.traces[k].raw_traces_from_parquet(cache_path)
            logging.info("Successfully loaded from cache")
        except:
            root, _, files = next(os.walk(self.folder_path))
            files.sort()
            for f in files:
                file_path = os.path.join(root, f)
                self._extract_raw_data(file_path)

            self._derive_raw_data()

            for v in self.traces.values():
                v.raw_traces_to_parquet(cache_path)

    def _extract_raw_data(self, file_path: str):
        """Extract raw data from a file

        This function reads the file and tries to extract the raw data supposing we have a matrix where each column
        is a trace ordered as self.traces (without the derived traces).

        Note: we assume that every row that presents that number of columns with
        floats is valid.

        Args:
            file_path (str): file from which data must be extracted
        """

        logging.info(f"Extract {file_path}")

        header = -1
        with open(file_path) as f:
            for ls in f.readlines():
                try:
                    spl = [float(x) for x in ls.split()]

                    if len(spl) < len(self.root_traces):
                        raise ValueError

                    if len(spl) > len(self.root_traces):
                        raise TraceDBError(
                            f"The file {file_path} contains more traces ({len(spl)}) than the root "
                            f"traces "
                            f"provided ({len(self.root_traces)}): {', '.join(self.root_traces)}"
                        )
                    break
                except ValueError:
                    header += 1

        data = pandas.read_csv(
            file_path,
            delim_whitespace=True,
            names=self.root_traces,
            header=None,
            skiprows=range(header + 1),
        )

        data = data.apply(pandas.to_numeric, errors="coerce").dropna()

        for k, v in data.items():
            self.traces[k].add_raw_trace(file_path, v)

    def raw_rows(self):
        """Maximum number of rows per raw trace"""
        return numpy.array([len(trace.raw_traces) for trace in self.traces.values()])

    def plot(
        self, trace_names=[], trace_files=[], savefig_path=None, legend=False, suffix=""
    ):
        """Plot

        Args:
              - trace_names (list, optional): if specified it plots all the raw_traces of the traces specified.
              Otherwise we plot everything
              - savefig_path (str, optional): if speficied we also save the file in the specified folder.
        """
        time_trace = self.get_time_trace()

        if not trace_names:
            trace_names = self.traces

        for trace_name in trace_names:
            trace = self.traces[trace_name]
            if trace.name == time_trace.name:
                continue

            trace.plot(
                trace_files=trace_files,
                time_trace=time_trace,
                savefig_path=savefig_path,
                title_prefix=self.name,
                legend=legend,
                suffix=suffix,
            )
