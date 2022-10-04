# STEPS_Validation

This repository contains long validation tests for STEPS. The short validation
tests are found in the main STEPS repository under test/validation.

To run all validations, clone this repository and go to the validation directory
 
 ```
 git clone https://github.com/CNS-OIST/STEPS_Validation.git
 cd STEPS_Validation/validation
 ```
    
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
