import numpy
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


class UtilsError(Exception):
    pass


class Utils:
    """Class that collects general mathematic methods"""

    @staticmethod
    def peaks(trace: list, prominence_multi: float = 0.01):
        """Extract the peaks from a trace based on the prominence

        Args:
              -trace (list): signal that we want to analyze
              - prominence_multi (float): prominence threshold as fraction of the span (max -min) of the trace
        """
        trace = numpy.array(trace)
        max = numpy.amax(trace)
        min = numpy.amin(trace)
        return find_peaks(trace, prominence=prominence_multi * (max - min))

    @staticmethod
    def n_peaks(trace: list):
        """Get number of peaks"""
        trace = numpy.array(trace)
        return len(Utils.peaks(trace)[0])

    @staticmethod
    def max_prominence(trace: list, prominence_multi: float = 0.01):
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace, prominence_multi)
        prominences = peaks[1]["prominences"]

        return numpy.amax(prominences)

    @staticmethod
    def max_prominence_t(trace: list, time_trace: list, prominence_multi: float = 0.01):
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace, prominence_multi)
        prominences = peaks[1]["prominences"]
        i_peak = numpy.argmax(prominences)

        return time_trace[peaks[0][i_peak]]

    @staticmethod
    def amax_t(trace: list, time_trace: list):
        """Time stamp of the max"""
        return time_trace[numpy.argmax(trace)]

    @staticmethod
    def i_prominence(i_peak: int, trace: list, prominence_multi: float = 0.01):
        """Prominence of the ith peak"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace, prominence_multi)
        prominences = peaks[1]["prominences"]

        return prominences[i_peak]

    @staticmethod
    def i_prominence_t(
        trace: list, time_trace: list, i_peak: int, prominence_multi: float = 0.01
    ):
        """Time stamp of the ith peak"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace, prominence_multi)
        return time_trace[peaks[0][i_peak]]

    @staticmethod
    def i_prominence_y(
        trace: list, time_trace: list, i_peak: int, prominence_multi: float = 0.01
    ):
        """Height of the ith peak"""
        trace = numpy.array(trace)
        peaks = Utils.peaks(trace, prominence_multi)
        return trace[peaks[0][i_peak]]

    @staticmethod
    def freq(trace: list, time_trace: list):
        """Frequency computed as n_peaks/simulation_time"""
        trace = numpy.array(trace)
        time_trace = numpy.array(time_trace)
        return Utils.n_peaks(trace) / (time_trace[-1] - time_trace[0])

    @staticmethod
    def freq2(trace: list, time_trace: list):
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
    def freq3(trace, time_trace, prominence_multi: float = 0.5):
        """Frequency computed using the fft"""
        trace = numpy.array(trace)
        time_trace = numpy.array(time_trace)

        xf, yf = Utils.fft(trace, time_trace)
        xf = xf[1:]
        yf = yf[1:]
        fpeaks = Utils.peaks(yf, prominence_multi)

        f = xf[fpeaks[0][0]]

        return f

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
    def sqrtmse(sample_time_trace, sample, benchmark_time_trace, benchmark):
        """Square root of the mean square error"""

        sample_time_trace = numpy.array(sample_time_trace)
        sample = numpy.array(sample)
        benchmark_time_trace = numpy.array(benchmark_time_trace)
        benchmark = numpy.array(benchmark)
        [interp_sample, interp_benchmark, iterp_time] = Utils._format_traces(
            sample_time_trace, sample, benchmark_time_trace, benchmark
        )

        diff = numpy.array(interp_sample) - numpy.array(interp_benchmark)

        return numpy.sqrt(numpy.square(diff).mean()), iterp_time, diff
