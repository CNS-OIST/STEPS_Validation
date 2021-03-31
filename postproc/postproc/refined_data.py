import os
import numpy
import scipy
from scipy.signal import find_peaks
from .raw_data import RawData


class RefinedDataError(Exception):
    pass


class RefinedData:
    """RefinedData

    This class takes the data collected and sorts it a standardized data structure.
    If the data is raw, it also reduces it based on the reduce_ops.

    For example:

    Imagine that as raw_data we have:
    {file_path : {'t', 'x', 'y'}}

    and we want the max and min of the traces (reduce_ops = ['amax', 'amin']).

    The refined data is:

    {'t', {'max' : [max_t_file_1, max_t_file_2, ..., max_t_file_n],
           'min' : [min_t_file_1, min_t_file_2, ..., min_t_file_n]},
     'x', {'max' : [max_x_file_1, max_x_file_2, ..., max_x_file_n],
           'min' : [min_x_file_1, min_x_file_2, ..., min_x_file_n]},
     't', {'max' : [max_y_file_1, max_y_file_2, ..., max_y_file_n],
           'min' : [min_y_file_1, min_y_file_2, ..., min_y_file_n]}}

    and is stored in self.data
    """

    def __init__(self, arg0, reduce_ops: list = [], time_trace: str = "t"):
        """Init

        Args:
            folder (str): in which the various raw data are
            traces (list): data ordering. Check _extract_raw_data for more info
        """
        self.reduce_ops = []
        self.time_trace = time_trace
        self.reduce_ops = reduce_ops
        self.folder = ""
        self.data = {}

        if isinstance(arg0, str):
            if not reduce_ops:
                raise RefinedDataError(
                    "you provided a folder for refined data but you asked for reduce operations"
                )
            self.folder = arg0
            self.data = self._extract_all_refined_data_sfst()
        elif isinstance(arg0, RawData):
            if reduce_ops is None:
                raise RefinedDataError(
                    "you provided raw data without reduce operations"
                )
            self.data = self.refine(raw_data=arg0)
        elif isinstance(arg0, dict):
            self.data = arg0
        else:
            raise RefinedDataError(f"unknown arg0 of type {type(arg0).__name__}")

    def __str__(self):
        ss = f"reduce_ops: {str(self.reduce_ops)}"
        ss += f"folder: {self.folder}"
        ss += f"time_trace: {self.time_trace}"
        ss += "data:"
        ss += str(self.data)
        return ss

    def _extract_refined_data_sfst(self, file: str):
        """Extract refined data sfst: single file, single trace

        sfst means that one file contains only one trace and we extract it
        """
        out = []

        f = open(file)
        for l in f.readlines():
            try:
                if len(l.split()) == 1:
                    out.append(float(l))
            except:
                pass

        if len(out) == 0:
            raise RefinedDataError(
                "No trace was extracted. Probably the file structure is not single file, single trace"
            )

        return out

    def _extract_all_refined_data_sfst(self, trace_name: str = "V"):
        """Extract all the refined data in folder

        We suppose that all the files in the folder are sfst (single file, single trace) type. The reduce operation
        done on the Refined data is inside their name
        """
        out = {trace_name: {}}
        folder = self.folder

        for root, _, files in os.walk(folder):
            if root == folder:
                common_prefix = os.path.commonprefix(files)
                for f in files:
                    path = os.path.join(root, f)
                    op = os.path.splitext(f[len(common_prefix) :])[0]
                    out[trace_name][op] = self._extract_refined_data_sfst(path)

        return out

    def refine(self, raw_data):
        """Data refiner

        you can use all the static functions of this class plus standard numpy ans scipy functions
        """
        ops = self.reduce_ops
        time_trace = self.time_trace

        out = {
            i: {j: [] for j in ops} for i in next(iter(raw_data.data.values())).keys()
        }

        for file, traces in raw_data.data.items():
            for trace_name, trace in traces.items():
                if trace_name == time_trace:
                    continue

                for op in ops:
                    try:
                        out[trace_name][op].append(getattr(self, op)(trace))
                    except TypeError:
                        out[trace_name][op].append(
                            getattr(self, op)(trace, traces[time_trace])
                        )
                    except AttributeError:
                        try:
                            out[trace_name][op].append(getattr(numpy, op)(trace))
                        except AttributeError:
                            out[trace_name][op].append(getattr(scipy, op)(trace))

        return out

    @staticmethod
    def peaks(trace: list, prominence_multi: float = 0.5):
        max = numpy.amax(trace)
        min = numpy.amin(trace)
        return find_peaks(trace, prominence=prominence_multi * (max - min))

    @staticmethod
    def n_peaks(trace: list):
        return len(RefinedData.peaks(trace)[0])

    @staticmethod
    def max_prominence(trace: list):
        trace = numpy.array(trace)
        peaks = RefinedData.peaks(trace)
        prominences = peaks[1]["prominences"]

        if len(prominences):
            return numpy.amax(prominences)
        else:
            return 0

    @staticmethod
    def freq(trace: list, time_trace: list):
        return RefinedData.n_peaks(trace) / (time_trace[-1] - time_trace[0])

    @staticmethod
    def freq2(trace: list, time_trace: list):
        time_trace = numpy.array(time_trace)
        peaks = RefinedData.peaks(trace)
        periods = numpy.array(
            [
                (val - time_trace[peaks[0][idx]])
                for idx, val in enumerate(time_trace[peaks[0][1:]])
            ]
        )

        return numpy.average(1 / numpy.array(periods))
