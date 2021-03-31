# STEPS postprocessing

This small python module contains some routines to:

- extract simulation results
- refine the raw data
- compare results with a benchmark to assess if simulators produce "same results" (in a statistical sense)

The standard use case is when we have a new simulator/simulator feature and we want to test on various traces if 
results are statistically equal. In other words, if we cannot exclude using a goodness of fit test the null hypothesis

## How to use the module

There are 2 ways to use the code: using raw or refined data. Raw data must be extracted and refined before it can be 
compared. Refined data provides already meaningful traces that can be used for comparison. In both cases 
sample/benchmark data go in their respective folders. For raw data we also need to provide what refined data must be 
extracted. A small example in main.py walks throw this process

  

