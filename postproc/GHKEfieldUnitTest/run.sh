#! /bin/bash

for (( i=$1; i<$2; i++ ))
do
	/home/katta/projects/HBP_STEPS/build/src/tetopsplit_dist --test 12 --scale 1 --rng-seed $i --efield-dt 1e-5 --logfile sample/res${i}.txt /home/katta/projects/HBP_STEPS/test/mesh/2tets_2patches_1comp.msh
	python benchmark/running_framework/GHK_unitTest_parallel.py $i
done


