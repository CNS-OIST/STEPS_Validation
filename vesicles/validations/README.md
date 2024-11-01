
# VALIDATION MODELS FOR VESICLE MODELING IN STEPS 5.0. 
*Authors Iain Hepburn and Jules Lallouette*

---------------------------------------------------------------------

We provide only a brief description of the models here, but a full description can be found in:
Iain Hepburn, Jules Lallouette, Weiliang Chen, Andrew R. Gallimore, Sarah Y. Nagasawa-Soeda, Erik De Schutter: Vesicle and reaction-diffusion hybrid modeling with STEPS. Communications Biology (in press)

The output of these models is directly used in the figures, as indicated in the notes below. 

We also give an indication of runtime because this varies considerably- some models necessitate a long runtime, whilst others run very quickly. Of course this is just an indication, and different platforms will give different results.  

All models, unless stated, must be run on a minimum of two cores on a STEPS version 5.0 or above. For example, to run the 'vesreac' model on 4 cores:
 ```
 mpirun -n 4 python3 parallel_vesreac.py
 ```

Models whose script name starts with `parallel_` are part of the automatically discovered validation tests for the whole repository (see `README.md` at the root of the repository) but can also be run manually.

NOTE: sometimes due to stochastic effects, a reproduced figure may look slightly different from the published figure. 

---------------------------------------------------------------------


**vesicle_diff.py**
 - runtime ~90 minutes on 2 cores. 
 - produces plot plots/vesicle\_diff.pdf, shown in Fig. 3b
 - data is recorded to data/vesicle_diff.h5

**rothman.py**
 - this is a large model of over 100,000 vesicles in a mossy fibre terminal mesh of over 500,000 tetrahedrons
 - runtime ~2.5 hours on 2 cores.
 - data is recorded to data/rothman.h5

**rothman_plot.py**
 - does not need to be run in parallel
 - is used to plot the Rothman mossy fibre terminal model data to plots/rothman.pdf, shown in Fig. 3c 

**path_ind.py**
 - runtime ~5 seconds on 2 cores 
 - produces plot plots/path\_ind.pdf
 - data is recorded to data/path_ind.h5

**parallel_path.py**
 - runtime ~5 seconds on 2 cores 
 - produces plot plots/path.pdf, shown in Fig. 3d
 - data is recorded to data/path_test.h5

**ghosh.py**
 - This model compares the Ghosh algorithm to surface diffusion on a tetrahedral mesh
 - runtime ~10 minutes on 1 core (serial solver)
 - produces plots shown in Fig. 3e,f and Supplementary Fig. 2
 - data is recorded to data/ghosh.h5

**ghosh_vessurf.py**
 - This model validates the implementation of the Ghosh algorithm for surface diffusion on a vesicle
 - runtime ~20 minutes on 2 cores
 - produces in plots/ ghosh_vessurf_AD1.pdf, ghosh_vessurf_AD2.pdf, ghosh_vessurf_t1.pdf, ghosh_vessurf_t2.pdf, ghosh_vessurf_t3.pdf
 - data is recorded to data/ghosh_vessurf.h5

**parallel_exocytosis.py**
 - runtime ~1 minute on 2 cores 
 - produces plot plots/exocytosis.pdf, shown in Fig. 4a
 - data is recorded to data/exocytosis_test.h5

**parallel_kissandrun.py**
 - runtime ~4 minutes on 2 cores
 - produces plot plots/kissandrun.pdf
 - data is recorded to data/kissandrun_test.h5

**parallel_raftendocytosis.py**
 - runtime ~2 minutes on 2 cores 
 - produces plot plots/raftendocytosis.pdf, shown in Fig. 4b
 - data is recorded to data/raftendocytosis_test.h5

**parallel_endocytosis.py**
 - runtime ~1 minute on 2 cores 
 - produces plot plots/endocytosis.pdf, shown in Fig. 4c,d
 - data is recorded to data/endocytosis_test.h5

**parallel_vesreac.py**
 - runtime ~10 minutes on 4 cores
 - produces plot plots/vesreac.pdf, shown in Fig. 5a,b,c,d
 - data is recorded to data/vesreac_test.h5

**vesreac_error.py**
 - should not be run with mpirun, this script will itself spawn mpirun commands so that several simulations are run in parallel
 - should be launched with e.g. `python3 vesreac_error.py runGrid 4 128 32 333` with `4` being the number of mpi ranks per simulation, `128` being the total number of cores that can be used, `32` being the number of iterations per mpirun command, and `333` being the base seed for the random number generator.
 - data is recorded to files named data/vesreac_error/vesreac_error_xxx.h5

**vesreac_error_plot.py**
 - does not need to be run in parallel. Produces plots from the data generated by vesreac_error.py
 - produces plot plots/vesreac_error_size.pdf, shown in Fig. 8b
 - produces plot plots/vesreac_error_size_foi.pdf, used as Supplementary Fig. 3
 
**parallel_binding.py**
 - runtime ~1 minute on 2 cores
 - produces plot plots/binding.pdf, shown in Fig. 5e
 - data is recorded to data/binding_test.h5

**parallel_raft_diff.py**
 - runtime ~3 minutes on 2 cores
 - produces plot plots/raft\_diff.pdf, shown in Fig. 6a
 - data is recorded to data/raft_diff_test.h5

**parallel_raftsreac.py**
 - runtime ~10 minutes on 8 cores
 - produces plot plots/raftsreac.pdf, shown in Fig. 6b,c,e,f
 - data is recorded to data/raftsreac_test.h5

**parallel_raft_gendis.py**
 - runtime ~1 hour on 2 cores
 - produces plot plots/raft\_gendis.pdf, shown in Fig. 6d
 - data is recorded to data/raft_gendis_test.h5

**reducedvol.py**
 - runtime ~8 hours on 2 cores
 - produces plots in plots/reducedvol\_(date)/ used in Supplementary Fig. 4
 - data is recorded to data/reducedvol.h5

**vesreac_immobile_error.py**
 - should not be run with mpirun, this script will itself spawn mpirun commands so that several simulations are run in parallel
 - should be launched with e.g. `python3 vesreac_immobile_error.py runGrid 2 64 data/vesreac_immobile_reactants/vesreac_immobile_reactants` with 2 being the number of mpi ranks per simulation and 64 being the total number of cores that can be used. Data will be recorded to files named data/vesreac_immobile_reactants/vesreac_immobile_reactants_xxx.h5
 - cytosolic species diffusion coefficient can be changed with e.g. `python3 vesreac_immobile_error.py runGrid 2 64 data/vesreac_slow_diff_spec/vesreac_slow_diff_spec 1e-12` (to run the simulations associated to Supplementary figure S5). Data will be recorded to files named data/vesreac_slow_diff_spec/vesreac_slow_diff_spec_xxx.h5

**vesreac_immobile_error_plot.py**
 - does not need to be run in parallel
 - produces plots/vesreac_immobile_reactants_error_DCST_1e-13.pdf, shown in Fig 8c from the data generated by `vesreac_immobile_error.py`
 - can also be called with a different path to generate the plot from Supplementary Figure S5: `python3 vesreac_immobile_error_plot.py data/vesreac_slow_diff_spec/vesreac_slow_diff_spec`

