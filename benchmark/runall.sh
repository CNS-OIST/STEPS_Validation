#!/bin/bash
python -c "import steps; steps._greet()"
echo "Benchmark: Serial Wmdirect Wmrssa"
pushd serial/well_mixed && python runsim.py && popd 
echo "Benchmark: Serial Tetexact TetODE with EField"
pushd serial/spatial && python runsim.py && popd 
echo "Benchmark: Parallel TetOPSplit with PETSc EField"
pushd parallel/spatial && mpirun -n 10 python runsim.py && popd 