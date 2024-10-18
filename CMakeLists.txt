cmake_minimum_required(VERSION 3.16.3)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

project(STEPS_Validations)

include(CTest)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

option(USE_MPI "Include MPI-based validations" ON)
option(STEPS_USE_DIST_MESH "Include validations for the distributed mesh solver" ON)

if(USE_MPI)
  find_package(MPI REQUIRED)
  if(MPI_FOUND)
    get_filename_component(MPI_BIN_PATH ${MPI_CXX_COMPILER} DIRECTORY)
    find_program(
      MPIRUN
      NAMES srun mpirun
      HINTS ${MPI_BIN_PATH} ${MPI_HOME} $ENV{MPI_HOME}
      PATHS ${_MPI_PREFIX_PATH}
      PATH_SUFFIXES bin
      DOC "Executable for running MPI programs.")
  endif()
endif()

include(AddPythonTests)

add_python_tests(serial-validations . ${SERIAL_PYTEST_PREFIX})
add_python_tests(parallel-validations . ${PARALLEL_PYTEST_PREFIX})
add_python_tests(distributed-validations . ${DISTRIBUTED_PYTEST_PREFIX})
