# STEPS_Validation

This repository contains long validation tests for STEPS. The sort validation
tests are found in the main STEPS repository under test/validation.

To run all validations, use the following command:

1. Serial validations (validation_rd, validation_cp, validation_efield)

    cd validation
    python run_validation_tests.py

2. parallel validations (validation_rd_mpi)

    cd validation
    mpirun -n 4 python run_validation_rd_mpi_tests.py
    
    


