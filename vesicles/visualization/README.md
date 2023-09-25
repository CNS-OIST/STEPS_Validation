Visualizations of vesicle models in STEPS with three examples.  

These models are shown in Supplementary Videos 1,2,3 of: Hybrid vesicle and reaction-diffusion modeling with STEPS
Iain Hepburn, Jules Lallouette, Weiliang Chen, Andrew R. Gallimore, Sarah Y. Nagasawa, Erik De Schutter
bioRxiv 2023.05.08.539782; doi: https://doi.org/10.1101/2023.05.08.539782



1. Active transport on virtual actin cytoskeleton 'Paths'
TODO


2. Exocytosis

This is an extract from the model of Gallimore et al (https://doi.org/10.1101/2023.08.03.551909) modeling SNARE complex priming and exocytosis, with glutamate release into the extracellular space. The data is compressed in exocytosis/exocytosis.h5.zip which must be extracted first to exocytosis/exocytosis.h5.

The data is visualized in Blender using the stepsblender extension module. The data should be loaded via stepsblender with the commands to show the docked vesicles, SNARE complexes, glutamate and calcium:

 ```
 python3 -m stepsblender.load exocytosis --blenderArgs exocytosis.blend --exclude ".*" --include ves,glu,Ca,SNARE_syt_CXN_Ca3_bCa2,SNARE_syt,SNARE_syt_CXN,SNARE_syt_CXN_Ca,SNARE_syt_CXN_Ca2,SNARE_syt_CXN_Ca3,SNARE_syt_CXN_bCa,SNARE_syt_CXN_bCa2,SNARE_syt_CXN_Ca_bCa,SNARE_syt_CXN_Ca_bCa2,SNARE_syt_CXN_Ca2_bCa,SNARE_syt_CXN_Ca2_bCa2,SNARE_syt_CXN_Ca3_bCa  --timeScale 10 --Species.radius 0.004 --Ca.radius 0.001 --glu.radius 0.001 --Vesicles.immobileSpecs ".*"
 ```
 

3. Clustering

This is an extract from the model of Gallimore et al (https://doi.org/10.1101/2023.08.03.551909) modeling vesicle cluster formation by synapsin dimer formation. The data is compressed in clustering/clustering.h5.zip which must be extracted first to clustering/clustering.h5. 

The data is visualized in Blender using the stepsblender extension module. The data should be loaded via stepsblender with the following commands to show the vesicles that can form synpsin dimers, and 10 'inert' vesicles that do not interact with the cluster. 

 ```
 python3 -m stepsblender.load clustering --blenderArgs clustering.blend --exclude ".*" --include ves --timeScale 5
 ```
