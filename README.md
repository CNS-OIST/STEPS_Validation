# STEPS_Validation

This repository contains long validation tests for STEPS. The short validation
tests are found in the main STEPS repository under test/validation.

Target STEPS version: 5.0.3

To run all validations, clone this repository and discover tests by running cmake:
 
 ```
git clone https://github.com/CNS-OIST/STEPS_Validation.git
cd STEPS_Validation
cmake .
```

`cmake` can be run with the `-DUSE_MPI=OFF` option to avoid considering the validation tests that use MPI. It can also be run with the `-DSTEPS_USE_DIST_MESH=OFF` option to avoid considering the validation tests that involve the distributed mesh solver.

## Automatic validation runs

To run all discovered tests, using e.g. 12 cores, use:

```
ctest -j 12
```
    
## Manual validation runs

The validations inside the `validation` folder can be run manually:

1. To run all serial validations (validation_rd, validation_cp, validation_efield)
    
    ```
    python3 run_validation_tests.py
    ```

2. To run all parallel validations (validation_rd_mpi, validation_efield_mpi)
    
    ```
    mpirun -n 4 python3 run_validation_mpi_tests.py 
    ```
    
3. To run all distributed validations (validation_rd_dist)
    
    ```
    mpirun -n 4 python3 run_validation_dist_tests.py 
    ```
    
Any of these scripts can run specific validation suites with e.g.:

```
python3 run_validation_tests.py validation_rd
```

Single validation scripts can be run from the STEPS_Validation folder with e.g.:

```
python3 -m validation.validation_cp.test_csd_clamp
```
