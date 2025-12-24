"""
A simple MPI "Hello World" script in Python.
Each process identifies itself and the total number of processes.
"""
from mpi4py import MPI
import sys

def print_hello():
    # Get the global communicator
    comm = MPI.COMM_WORLD
    # Get the rank (ID) of the current process (0 to size-1)
    rank = comm.Get_rank()
    # Get the total number of processes
    size = comm.Get_size()
    # Get the name of the processor the process is running on
    name = MPI.Get_processor_name()

    msg = f"Hello, World! I am process {rank} of {size} running on {name}.\n"
    sys.stdout.write(msg)

if __name__ == "__main__":
    print_hello()
