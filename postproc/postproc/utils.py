import logging
import os
import re

import numpy
import pandas
from scipy import interpolate, stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


class UtilsError(Exception):
    pass


class Utils:
    """Class that collects general methods"""

    @staticmethod
    def autonaming_after_folders(path1, path2):
        """ Extrapolate sample name from folders """
        res = [os.path.basename(path1), os.path.basename(path2)]
        while res[0] == res[1] and path1 and path2:

            path1 = os.path.dirname(path1)
            path2 = os.path.dirname(path2)
            res = [os.path.basename(path1), os.path.basename(path2)]

        if res[0] == res[1]:
            res = ["sample_0", "sample_1"]

        return res

    @staticmethod
    def val(trace: list, time_trace: list, t: float):
        """Get value from trace corresponding a to a certain time t"""
        idx = numpy.argmin(numpy.abs(numpy.array(time_trace) - t))
        return trace[idx]

    @staticmethod
    def peaks(trace: list, height=(0.0, None), distance=100):
        """Extract the peaks from a trace based on the prominence

        Args:
              -trace (list): signal that we want to analyze
              - prominence_multi (float): prominence threshold as fraction of the span (max -min) of the trace
        """
        trace = numpy.array(trace)

        # max = numpy.amax(trace)
        # min = numpy.amin(trace)
        # prominence = prominence_multi * (max - min)

        return find_peaks(trace, height=height, distance=distance)

    @staticmethod
    def n_peaks(trace: list):
        """Get number of peaks"""
        trace = numpy.array(trace)
        return len(Utils.peaks(trace)[0])

    @staticmethod
    def max_prominence(trace: list):
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace)
        prominences = peaks[1]["prominences"]

        return numpy.amax(prominences)

    @staticmethod
    def max_prominence_t(trace: list, time_trace: list):
        trace = numpy.array(trace)
        time_trace = numpy.array(time_trace)
        peaks = Utils.peaks(trace)
        prominences = peaks[1]["prominences"]
        i_peak = numpy.argmax(prominences)

        return time_trace[peaks[0][i_peak]]

    @staticmethod
    def max_prominence_y(trace: list):
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace)
        prominences = peaks[1]["prominences"]
        i_peak = numpy.argmax(prominences)

        return trace[peaks[0][i_peak]]

    @staticmethod
    def amax_t(trace: list, time_trace: list):
        """Time stamp of the max"""
        return time_trace[numpy.argmax(trace)]

    @staticmethod
    def i_prominence(trace: list, i_peak: int):
        """Prominence of the ith peak"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace)
        prominences = peaks[1]["prominences"]
        if len(prominences) > i_peak:
            return prominences[i_peak]
        else:
            return float("NaN")

    @staticmethod
    def peaks_t(trace: list, time_trace: list, npeaks=None):
        """Time stamp of the peaks"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace)
        if npeaks:
            return [time_trace[i] for i in peaks[0][:npeaks]]
        else:
            return [time_trace[i] for i in peaks[0]]

    @staticmethod
    def peaks_y(trace: list, npeaks=None):
        """Height of the peaks"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace)
        if npeaks:
            return [trace[i] for i in peaks[0][:npeaks]]
        else:
            return [trace[i] for i in peaks[0]]

    @staticmethod
    def i_peak_t(trace: list, time_trace: list, i_peak: int):
        """Time stamp of the ith peak"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace)

        if len(peaks[0]) > i_peak:
            return time_trace[peaks[0][i_peak]]
        else:
            return float("NaN")

    @staticmethod
    def i_peak_y(trace: list, i_peak: int):
        """Height of the ith peak"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace)
        if len(peaks[0]) > i_peak:
            return trace[peaks[0][i_peak]]
        else:
            return float("NaN")

    @staticmethod
    def freq(trace, time_trace, multi_y=1, multi_t=1):
        """Frequency computed using the fft

        multi_y and multi_t are required when the traces are not in SI and you still want Hz in return. They are the
        correct multipliers for restoring the traces to SI units
        """

        trace = numpy.array(trace) * multi_y
        time_trace = numpy.array(time_trace) * multi_t

        xf, yf = Utils.fft(trace, time_trace)
        xf = xf[1:]
        yf = yf[1:]

        fpeaks = Utils.peaks(yf, height=0.005, distance=None)

        f = xf[fpeaks[0][0]]

        return f

    @staticmethod
    def freq2(trace: list, time_trace: list):
        """Frequency computed as n_peaks/simulation_time"""
        trace = numpy.array(trace)
        time_trace = numpy.array(time_trace)
        return Utils.n_peaks(trace) / (time_trace[-1] - time_trace[0])

    @staticmethod
    def freq3(trace: list, time_trace: list):
        """Frequency computed as avg(1/T) where T is the period between peaks"""
        trace = numpy.array(trace)
        time_trace = numpy.array(time_trace)
        peaks = Utils.peaks(trace)
        periods = numpy.array(
            [
                (val - time_trace[peaks[0][idx]])
                for idx, val in enumerate(time_trace[peaks[0][1:]])
            ]
        )
        return numpy.average(1 / numpy.array(periods))

    @staticmethod
    def fft(trace, time_trace):
        """Fast Fourier Transform (only positive side)"""
        trace = numpy.array(trace)
        time_trace = numpy.array(time_trace)

        T = time_trace[1] - time_trace[0]
        N = len(trace)

        yf = numpy.abs(fft(trace)[0 : N // 2]) * 2 / N

        xf = fftfreq(N, T)[: N // 2]

        return xf, yf

    @staticmethod
    def _format_traces(x0, y0, x1, y1):
        """Format 2 traces so that present same sampling

        We interpolate the longer on the shorter. If the simulation times do not overlap we throw an error.
        """

        start = max(x0[0], x1[0])
        stop = min(x0[-1], x1[-1])
        npoints = max(len(x0), len(x1))

        if start >= stop:
            raise UtilsError("Trace times do not overlap")

        x_intersection = numpy.linspace(start, stop, npoints)
        f_y0 = interpolate.interp1d(x0, y0)
        f_y1 = interpolate.interp1d(x1, y1)

        interp_y0 = f_y0(x_intersection)
        interp_y1 = f_y1(x_intersection)

        return interp_y0, interp_y1, x_intersection

    @staticmethod
    def mse(sample_time_trace, sample, benchmark_time_trace, benchmark, percent=False):
        """mean square error"""

        sample_time_trace = numpy.array(sample_time_trace)
        sample = numpy.array(sample)
        benchmark_time_trace = numpy.array(benchmark_time_trace)
        benchmark = numpy.array(benchmark)
        [interp_sample, interp_benchmark, interp_time] = Utils._format_traces(
            sample_time_trace, sample, benchmark_time_trace, benchmark
        )

        interp_benchmark = numpy.array(interp_benchmark)
        interp_sample = numpy.array(interp_sample)
        diff = interp_benchmark - interp_sample
        if percent:
            diff /= numpy.maximum(abs(interp_benchmark), abs(interp_sample))

        return numpy.square(diff).mean(), interp_time, diff

    @staticmethod
    def conf_int(a, confidence=0.95):
        """Confidence interval"""

        # in case all elements are equal the interval is (NaN, NaN) instead of (a[0], a[0])
        if all(a[0] == i for i in a):
            return (a[0], a[0])

        r = stats.t.interval(
            confidence,
            len(a) - 1,
            loc=numpy.mean(a),
            scale=stats.sem(a),  # scale=numpy.std(a)
        )

        return r

    @staticmethod
    def atoi(text):
        return int(text) if text.isdigit() else text

    @staticmethod
    def natural_keys(text):
        """
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        """
        return [Utils.atoi(c) for c in re.split(r"(\d+)", text)]

    @staticmethod
    def flatten_data_frame_if_necessary(df):
        return pandas.DataFrame(
            {k: pandas.Series(v.explode().to_numpy()) for k, v in df.items()}
        )

    @staticmethod
    def common_prefix(v):
        """Return common prefix for a list of strings"""
        return os.path.commonprefix(v)

    @staticmethod
    def common_suffix(v):
        """Return common suffix for a list of strings"""
        return Utils.common_prefix([i[::-1] for i in v])[::-1]

    @staticmethod
    def pretty_print_combinations(comb):
        if len(comb) == 0:
            return []

        p = zip(*comb) if type(comb[0]) == tuple else [comb]

        ans = [""] * len(comb)
        for i in p:
            cp = Utils.common_prefix(i)
            cs = Utils.common_suffix(i)

            if len(cp) == len(i[0]):
                continue

            ans = [
                token[len(cp) : len(token) - len(cs)]
                if not ans[idx]
                else ans[idx] + "\n" + token[len(cp) : len(token) - len(cs)]
                for idx, token in enumerate(i)
            ]

        return ans

    @staticmethod
    def sanitize_path(path):
        """Useful for using path as path for files"""
        path = re.sub(" ", "_", path)
        path = re.sub("\.", "", path)
        return path

    @staticmethod
    def savefig(path, name, fig):
        """A few checks and default saving configuration"""
        path = Utils.sanitize_path(path)
        name = Utils.sanitize_path(name)
        if not name.endswith(".jpg"):
            name += ".jpg"

        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, name)

        fig.savefig(full_path, dpi=300)

    @staticmethod
    def set_subplot_title(i, j, ncols, ax, title=None):
        """Set default titles for subplots"""
        ic = i * ncols + j
        left_title = chr(ord("A") + ic)
        if title:
            left_title += "\n"
            ax.set_title(title)

        ax.set_title(left_title, loc="left", fontweight="bold")

    @staticmethod
    def pretty_print_goodness_of_fit(comp, goodness_of_fit_test_type, filter):
        """Pretty print of the the goodness of fit test"""

        logging.info("Goodness of fit tests")
        for tDBnames, tests in comp.test_goodness_of_fit(
            test_type=goodness_of_fit_test_type, filter=filter
        ).items():
            print(tDBnames)
            for t, d in sorted(tests.items(), key=lambda t: Utils.natural_keys(t[0])):
                for k, v in sorted(d.items(), key=lambda k: Utils.natural_keys(k[0])):
                    print(t, k, v)
