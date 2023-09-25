
# VALIDATION MODELS FOR VESICLE MODELING IN STEPS 5.0. 
*Authors Iain Hepburn and Jules Lallouette*

---------------------------------------------------------------------

We provide only a brief description of the models here, but a full description can be found at https://doi.org/10.1101/2023.05.08.539782. 
The output of these models is directly used in the preprint figures, as indicated in the notes below. 

We also give an indication of runtime because this varies considerably- some models necessitate a long runtime, whilst others run very quickly. Of course this is just an indication, and different platforms will give different results.  

All models, unless stated, must be run on a minimum of two cores on a STEPS version 5.0 or above. For example, to run the 'vesreac' model on 4 cores:
 ```
 mpirun -n 4 python3 vesreac.py
 ```

NOTE: due to stochastic effects, any reproduced figure may look slightly different from published figures. 

---------------------------------------------------------------------


**vesicle_diff.py**
 - runtime ~90 minutes on 2 cores. 
 - produces plot plots/vesicle\_diff.pdf, shown in Fig. 2a

**rothman.py**
 - this is a large model of over 100,000 vesicles in a mossy fibre terminal mesh of over 500,000 tetrahedrons
 - runtime ~2.5 hours on 2 cores.
 - data is outputted to data/rothman\_{MESHFILE}\_{scale}\_{DT}\_{T\_END}

**rothman_plot.py**
 - does not need to be run in parallel
 - is used to plot the Rothman mossy fibre terminal model data to plots/rothman.pdf, shown in Fig. 2b 

**path_ind.py**
 - runtime ~5 seconds on 2 cores 
 - produces plot plots/path\_ind.pdf, shown in Fig. 2c

**path.py**
 - runtime ~5 seconds on 2 cores 
 - produces plot plots/path.pdf, shown in Fig. 2d

**ghosh.py**
 - runtime ~10 minutes on 8 cores
 - produces plots shown in Fig. 2e,f and Supplementary Fig. 2

**exocytosis.py**
 - runtime ~1 minute on 2 cores 
 - produces plot plots/exocytosis.pdf, shown in Fig. 3a

**raftendocytosis.py**
 - runtime ~2 minutes on 2 cores 
 - produces plot plots/raftendocytosis.pdf, shown in Fig. 3b

**endocytosis.py**
 - runstime ~1 minute on 2 cores 
 - produces plot plots/endocytosis.pdf, shown in Fig. 3c,d

**vesreac.py**
 - runtime ~10 minutes on 4 cores
 - produces plot plots/vesreac.pdf, shown in Fig. 4a,b,c,d

**binding.py**
 - runtime ~1 minute on 2 cores
 - produces plot plots/binding.pdf, shown in Fig. 4e

**raft_diff.py**
 - runtime ~3 minutes on 2 cores
 - produces plot plots/raft\_diff.pdf, shown in Fig. 5a

**raftsreac.py**
 - runtime ~10 minutes on 8 cores
 - produces plot plots/raftsreac.pd, shown in Fig. 5b,c,e,f

**raft_gendis.py**
 - runtime ~1 hour on 2 cores
 - produces plot plots/raft\_gendis.pdf, shown in Fig. 5d

**reducedvol.py**
 - runtime ~4 hours on 2 cores
 - produces plots in plots/reducedvol\_(date)/ used in Extended Data Fig. 1



