#Visualizations of vesicle models in STEPS with three examples.  

These models are shown in Supplementary Movies 1,2,3 of: 
Iain Hepburn, Jules Lallouette, Weiliang Chen, Andrew R. Gallimore, Sarah Y. Nagasawa, Erik De Schutter: Vesicle and reaction-diffusion hybrid modeling with STEPS. Communications Biology (in press)


1. Active transport on virtual cytoskeleton 'Paths'

This simple model demonstrates stochastic walks on 'Paths', representing motor protein transport on actin filaments or microtubules. 

The Python script activetransport/path.py runs the model and writes the data to activetransport/data/path.json. This model should be run in the usual way (from the activetransport directory):
 ```
 mpirun -n 2 python3 path.py
 ```
 
 To visualize the data in Coreform Cubit (https://coreform.com/products/coreform-cubit/) a custom script is provided, activetransport/path_cubit.py. To generate the images as for Supplementary Video 1, first Coreform Cubit should be launched, the working directory changed, and the mesh loaded so that some visualization settings can be made manually. The following two commmands should be run on the Coreform Cubit command line:
 
 ```
cd "{path_to_repo}/STEPS_Validation/vesicles/visualization/activetransport"
import abaqus mesh geometry "meshes/sphere_0.5D_2088tets.inp"
 ```

The "View in transparent mode" option should be selected so that vesicles can be seen inside the mesh. 
Then simply run the script activetransport/path_cubit.py from the "Play Journal File" option, and images will be output to activetransport/images/. 


2. Exocytosis

This is an extract from the model of Gallimore et al (https://doi.org/10.1101/2023.08.03.551909) modeling SNARE complex priming and exocytosis, with glutamate release into the extracellular space. The data is compressed in exocytosis/exocytosis.h5.zip which must be extracted first to exocytosis/exocytosis.h5.

The data is visualized in Blender using the stepsblender extension module. The data should be loaded via stepsblender with the commands to show the docked vesicles, SNARE complexes, glutamate and calcium:

 ```
 python3 -m stepsblender.load exocytosis --blenderArgs exocytosis.blend --exclude ".*" --include ves,glu,Ca,SNARE_syt_CXN_Ca3_bCa2,SNARE_syt,SNARE_syt_CXN,SNARE_syt_CXN_Ca,SNARE_syt_CXN_Ca2,SNARE_syt_CXN_Ca3,SNARE_syt_CXN_bCa,SNARE_syt_CXN_bCa2,SNARE_syt_CXN_Ca_bCa,SNARE_syt_CXN_Ca_bCa2,SNARE_syt_CXN_Ca2_bCa,SNARE_syt_CXN_Ca2_bCa2,SNARE_syt_CXN_Ca3_bCa  --timeScale 10 --Species.radius 0.004 --Ca.radius 0.001 --glu.radius 0.001 --Vesicles.immobileSpecs ".*"
 ```
 
 To directly render the movie in background mode, the `--render` option can be added. The frames will be rendered in the current working directory, a different directory can be specified with `--outputPath /path/to/dir`.
 

3. Clustering

This is an extract from the model of Gallimore et al (https://doi.org/10.1101/2023.08.03.551909) modeling vesicle clustering by synapsin dimer formation. The data is compressed in clustering/clustering.h5.zip which must be extracted first to clustering/clustering.h5. 

The data is visualized in Blender using the stepsblender extension module. The data should be loaded via stepsblender with the following commands to show the vesicles that can form synapsin dimers, and 10 'inert' vesicles that do not interact with the cluster. 

 ```
 python3 -m stepsblender.load clustering --blenderArgs clustering.blend --exclude ".*" --include ves --timeScale 5
 ```

To directly render the movie in background mode, the `--render` option can be added. The frames will be rendered in the current working directory, a different directory can be specified with `--outputPath /path/to/dir`.
