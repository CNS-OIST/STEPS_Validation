import os
import numpy
import scipy
import matplotlib.pyplot as plt
import collections

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
        self.raw_traces = collections.OrderedDict()
        self.derivation_params = derivation_params
        self.refined_traces = reduce_ops
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

    def _trim_raw_trace_name(self, name):
        """Compute trimmed (without shared parts) raw_data file name"""
        return name[
            self.raw_traces_common_prefix_len : len(name)
            - self.raw_traces_common_suffix_len
        ]

    def __str__(self, raw_trace_idxs=[]):
        """Print

        Args:
              - raw_trace_idxs (list): since self.raw_traces are many; we let the possibility to print only a
              selection of them indicated by
        """
        ss = f"name: {self.name}\n"
        ss += f"unit: {self.unit}\n"
        ss += f"multi: {self.multi}\n"

        ss += f"raw_traces ({len(self.raw_traces)}):\n"

        if not raw_trace_idxs:
            raw_trace_keys = list(self.raw_traces.keys())
        else:
            raw_trace_keys = [
                next(iter(self.raw_traces.keys()), i) for i in raw_trace_idxs
            ]

        for k in raw_trace_keys:
            v = self.raw_traces[k]
            ss += f"{self._trim_raw_trace_name(k)} ({len(v)}): {numpy.array(v)}\n"

        ss += f"refined_traces ({len(self.refined_traces)}):\n"
        for k, v in self.refined_traces.items():
            ss += f"{k}: {numpy.array(v)[raw_trace_idxs]}\n"

        return ss

    def append_to_raw_trace(self, file_path, val):
        """Service function to append to a raw_trace"""
        val = val * self.multi
        if file_path in self.raw_traces:
            self.raw_traces[file_path].append(val)
        else:
            self.raw_traces[file_path] = [val]

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

    def reduce_ops(self):
        """Service function to get the operations required to compute the refined_traces"""
        return self.refined_traces.keys()

    def refine(self, time_trace={}):
        """Fill refined_traces using evaluating the keys as member functions in Utils, numpy or scipy

        The refine_traces key can be a str or a list of arguments. In case of a str, it is evaluated as a member
        function of Utils, numpy or scipy. In case of a list, the first term is the member function while the others
        are arguments to be fed after the traces.
        """
        for op_key, file_path in self.refined_traces.items():
            if isinstance(file_path, str):
                f = open(file_path)

                trace = []
                for l in f.readlines():
                    try:
                        trace.append(self.multi * float(l))
                    except:
                        pass
                self.refined_traces[op_key] = trace
            else:
                for file, trace in self.raw_traces.items():

                    op_args = []
                    op = op_key
                    try:
                        l = eval(op_key)
                        op = l[0]
                        op_args = l[1:]
                    except NameError:
                        pass

                    if hasattr(Utils, op) and callable(func := getattr(Utils, op)):
                        try:
                            val = getattr(Utils, op)(*op_args, trace=trace)
                            # In case we need also the time_trace, we try it now
                        except TypeError:
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

                    self.refined_traces[op_key].append(val)

    def plot(self, time_trace, savefig_path=None):
        """Plot all the raw_traces"""
        title = self.name

        plt.clf()
        for file, trace in self.raw_traces.items():
            t = time_trace.raw_traces[file]
            plt.plot(t, trace, label=f"trace_{self._trim_raw_trace_name(file)}")

        plt.title(title)
        plt.xlabel(time_trace.unit)
        plt.ylabel(self.unit)

        if savefig_path:
            plt.savefig(os.path.join(savefig_path, title))
        plt.show()


class TraceDBError(Exception):
    pass


class TraceDB:
    """Trace DataBase

    This class maintains all the traces of a particular simulation type (sample or benchmark)
    """

    def __init__(self, traces, folder_path="", time_trace="t"):
        """Init

        Args:
              - traces: traces that must be fed with the raw data found in folder_path
              - folder_path (str): path to where the raw data are. The raw data must be in the form of
              a matrix where each column is a trace (in the order specified in traces). More information in
              _extract_raw_data
              - time_trace specified a particular trace as the time trace
        """
        self.time_trace = time_trace
        self.root_traces = [i.name for i in traces if not i.derivation_params]
        self.traces = {i.name: i for i in traces}
        if len(traces) != len(self.traces):
            raise TraceDBError(
                "Some traces have the same name. Please make them unique"
            )

        self.folder_path = folder_path

        self._extract_all_raw_data()

        self._derive_raw_data()

        self._refine()

        for i in self.traces:
            self.traces[i]._update_raw_traces_common_prefix_and_suffix()

    def get_time_trace(self):
        return self.traces.get(self.time_trace, None)

    def __str__(self):
        ss = f"folder_path: {self.folder_path}\n"
        for i in self.traces.values():
            ss += str(i) + "\n"
        return ss

    def _derive_raw_data(self):
        """Service function to call derive on all traces"""
        for trace_name in self.traces:
            self.traces[trace_name].derive_raw_data(self.traces)

    def _extract_all_raw_data(self):
        """Extract all raw data from the files in self.folder_path (not recursive) """

        if self.folder_path:
            root, _, files = next(os.walk(self.folder_path))
            files.sort()
            for f in files:
                file_path = os.path.join(root, f)
                self._extract_raw_data(file_path)

    def _extract_raw_data(self, file_path: str):
        """Extract raw data from a file

        This function reads the file and tries to extract the raw data supposing we have a matrix where each column
        is a trace ordered as self.traces (without the derived traces).

        Note: we assume that every row that presents that number of columns with
        floats is valid.

        Args:
            file_path (str): file from which data must be extracted
        """

        f = open(file_path)

        for l in f.readlines():
            try:
                ls = l.split()
                if len(ls) == len(self.root_traces):
                    for idx, trace_name in enumerate(self.root_traces):
                        self.traces[trace_name].append_to_raw_trace(
                            file_path, float(ls[idx])
                        )
            except:
                pass

    def _refine(self):
        """Service function that calls refine on all the traces"""
        for trace_name in self.traces:
            self.traces[trace_name].refine(self.get_time_trace())

    def plot(self, trace_names=[], savefig_path=None):
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

            trace.plot(time_trace=time_trace, savefig_path=savefig_path)
