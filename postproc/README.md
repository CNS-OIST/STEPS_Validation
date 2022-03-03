# STEPS post-processing

This small Python module contains some routines to compare simulation results. Usually comparisons are between STEPS 
3 and 4. Additional datasets for triple comparisons are possible. The central object is the `Comparator`. This accepts
2 or more `TraceDB` for comparisons. The first one provided is considered the baseline.

Generally speaking, the module performs the following actions:

1. extract simulation results
1. refine the raw data
1. compare results to assess if simulators produce "same results" (in a statistical sense)

The standard use case is when we have a new simulator/simulator feature and we want to test on various traces if 
results are statistically equal. In other words, if we cannot exclude using a goodness of fit test the null hypothesis.

## How to use the Python module

The module requires Python 3.8 or higher.
There are 2 ways to use the code: using raw or refined data. Raw data must be extracted and refined before they can be 
compared. Refined data provides already meaningful traces that can be used for comparison. In both cases 
sample/benchmark data go in their respective folders (or use other folders where the data are stored). For raw data we 
also need to provide what refined data must be extracted. 

Do not clear cache if you only have refined data! 

## Examples

The best way to illustrate post-processing functionalities is through examples:

- rallpack 1
- rallpack 3
- caburst

Incidentally, they are also the routines used to generate the results published in the STEPS 4 paper.
