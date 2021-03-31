from scipy import stats
from .refined_data import RefinedData


class Comparator:
    def __init__(self, sample: RefinedData, benchmark: RefinedData):
        self.sample = sample
        self.benchmark = benchmark

    def test_ks(self, time_trace: str = "t"):
        for trace_name, op in self.benchmark.data.items():
            if trace_name == time_trace:
                continue
            for op, op_trace in op.items():
                print(
                    trace_name,
                    op,
                    stats.ks_2samp(self.sample.data[trace_name][op], op_trace),
                )
