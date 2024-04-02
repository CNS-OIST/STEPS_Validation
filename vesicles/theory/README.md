
# VESICLE MODELING IN STEPS 5.0. 
*Authors Iain Hepburn and Jules Lallouette*

---------------------------------------------------------------------

A full description of these modesl can be found in:
Iain Hepburn, Jules Lallouette, Weiliang Chen, Andrew R. Gallimore, Sarah Y. Nagasawa-Soeda, Erik De Schutter: Vesicle and reaction-diffusion hybrid modeling with STEPS. Communications Biology (in press)

The output of these models is directly used in the figures, as indicated in the notes below. 

These models do not directly run the STEPS simulator, they are theoretical or complementary models to the work, and so do not need to be run in parallel. They can be run in serial in python by commands e.g.:
 ```
 python3 vesdiff_rmsd.py
 ```

NOTE: sometimes due to stochastic effects, a reproduced figure may look slightly different from the published figure. 

---------------------------------------------------------------------


**meshplots.py**
 - produces plot sphere_2D_291tets_tetlengths_average.pdf, shown in Fig. 2a
 - produces plot sphere_2D_265307tets_tetlengths_average.pdf, shown in Fig. 2b
 - produces plot pyr_axon_2021_cyt_tetlengths_average.pdf, shown in Fig. 2c

**vesdiff_steps.py**
 - generates data data/steps_5000000.npy

**vesdiff_steps_plot.py**
 - reads data data/steps_5000000.npy
 - produces plot plots/vesiclediff_steps.pdf, shown in Fig. 3a

**vesdiff_rmsd.py**
 - generates data data/rmsd_1000000.npy

**vesdiff_rmsd_plot.py**
 - reads data data/rmsd_1000000.npy
 - produces plot plots/vesiclediff_rmsd.pdf, shown in Fig. 8a
  
**path_dwelltimes.py**
 - reads data data/path_dwelltimes_single.npy and data/path_dwelltimes_double.npy
 - produces plot plots/path_dwelltimes.pdf, which is Supplementary Figure 1

