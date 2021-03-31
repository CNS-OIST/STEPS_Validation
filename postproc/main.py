from postproc.raw_data import RawData
from postproc.refined_data import RefinedData
from postproc.comparator import Comparator

import matplotlib.pyplot as plt

# In case of raw data

# raw_data = RawData('raw_data/sample', ['t','z_min', 'z_max'])
# ref_data = RefinedData(arg0=raw_data, reduce_ops=["amax", "n_peaks", "max_prominence", "freq", "freq2"])
# print(Comparator(ref_data, ref_data).test_ks())

# In case of refined data

# sample_ref_data = RefinedData('refined_data/sample')
# benchmark_ref_data = RefinedData('refined_data/benchmark')
#
# print(Comparator(sample_ref_data, benchmark_ref_data).test_ks())

s_raw_data = RawData("rallpack3/raw_data/sample", ["t", "V"])
b_raw_data = RawData("rallpack3/raw_data/benchmark", ["t", "V"])

s_ref_data = RefinedData(
    arg0=s_raw_data, reduce_ops=["amax", "n_peaks", "max_prominence", "freq", "freq2"]
)
b_ref_data = RefinedData(
    arg0=b_raw_data, reduce_ops=["amax", "n_peaks", "max_prominence", "freq", "freq2"]
)

print(s_ref_data)
print(b_ref_data)
