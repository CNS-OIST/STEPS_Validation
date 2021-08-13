# STEPS postprocessing

This small python module contains some routines to compare simulation results. The data sets are usually 
called:

- sample_STEPS4
- benchmark_STEPS3
- benchmark_NEURON

Additional trace databases are possible.

Generally speaking, the module performs the following actions:

- extract simulation results
- refine the raw data
- compare results to assess if simulators produce "same results" (in a statistical sense)

In case the analysis requires a "benchmark" and a "sample" and a comparison between the 2 the code will try to use 
the first (of each of the possible couples) as benchmark and the second as sample. 

The standard use case is when we have a new simulator/simulator feature and we want to test on various traces if 
results are statistically equal. In other words, if we cannot exclude using a goodness of fit test the null hypothesis

## How to use the module

There are 2 ways to use the code: using raw or refined data. Raw data must be extracted and refined before it can be 
compared. Refined data provides already meaningful traces that can be used for comparison. In both cases 
sample/benchmark data go in their respective folders (or use other folders where the data are stored). For raw data we 
also need to provide what refined data must be extracted. 

TL;DR: Check rallpack3.py and caburst.py for hints on how to use the module. If folders are missing that is the 
place to add the raw traces