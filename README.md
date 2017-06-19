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
    python run_validation_tests.py
    ```

2. To run all parallel validations (validation_rd_mpi, validation_efield_mpi)
    
    ```
    mpirun -n 4 python run_validation_mpi_tests.py 
    ```
    
    


