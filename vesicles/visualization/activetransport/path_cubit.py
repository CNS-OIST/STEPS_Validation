#!python
import pickle

cubit.cmd('color Volume 1 white')
ves_colour1 = 'yellow'
# We start at volume index 2
volindex = 2
ves_radius = '25e-3'
scale = 1e-6

with open('data/path.pkl', 'rb') as f:
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

