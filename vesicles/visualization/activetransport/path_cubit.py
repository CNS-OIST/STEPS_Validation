#!python
import pickle

# Do these commands on the CUBIT command line FIRST

# To change the working directory, e.g.: cd "{path_to_my_repo}/STEPS_Validation/vesicles/visualization/activetransport"
# So visuals can be set manually: import abaqus mesh geometry "meshes/sphere_0.5D_2088tets.inp"

cubit.cmd('color Volume 1 white')
ves_colour1 = 'yellow'
# We start at volume index 2
volindex = 2
ves_radius = '25e-3'
scale = 1e-6

with open('path.pkl', 'rb') as f:
    for t, vesDct in enumerate(pickle.load(f)):
        print(t)
        startvolindex = volindex
        for vidx, vpos in vesDct.items():
            position = ' '.join(str(x / scale) for x in vpos)
            cubit.cmd(f'create sphere radius {ves_radius}')
            cubit.cmd(f'color Volume {volindex} {ves_colour1}')
            cubit.cmd(f'move volume {volindex} location {position}')
            volindex += 1
        cubit.cmd(f'hardcopy "images/path_{t}.jpg" jpg')
        for vidx in range(startvolindex, volindex):
            cubit.silent_cmd(f'delete volume {vidx}')

