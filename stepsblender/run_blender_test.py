import argparse
import subprocess
import shutil
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(FILE_PATH, 'data')
MODEL_DIR = os.path.join(FILE_PATH, 'models')
FRAMES_DIR = os.path.join(DATA_DIR, 'frames')

# Models should at least generate 20 timesteps
DEFAULT_TSTART = 10
DEFAULT_TEND = 11

def printStage(msg):
    print('='*len(msg))
    print(msg)
    print('='*len(msg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'command',
        action='store',
        metavar='COMMAND',
        help='Action to execute ("clean", "run", "render", or "all" to run all of them in this order)')
    parser.add_argument('--python',
                        action='store',
                        metavar='/path/to/python',
                        help='Python binary used for STEPS and stepsblender',
                        default='python3')
    parser.add_argument('--mpi',
                        action='store',
                        metavar='/path/to/mpirun',
                        help='MPI run executable',
                        default='mpirun')
    parser.add_argument('-n',
                        action='store',
                        type=int,
                        help='number of mpi rank for steps simulation',
                        default=2)
    parser.add_argument('--model',
                        action='store',
                        type=str,
                        help='python script name in ./models',
                        default='default_model')
    parser.add_argument('--blenderPath',
                        type=str,
                        action='store',
                        metavar='/path/to/blender',
                        help='Path to the blender executable',
                        default='blender')
    parser.add_argument('-tstart',
                        action='store',
                        type=int,
                        help='Timesteps at which rendering should start',
                        default=DEFAULT_TSTART)
    parser.add_argument('-tend',
                        action='store',
                        type=int,
                        help='Timesteps at which rendering should end',
                        default=DEFAULT_TEND)

    args = parser.parse_args()

    if args.command in ['clean', 'all']:
        printStage('Clean all data')

        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)

        print('OK')
    if args.command in ['run', 'all']:
        printStage('Run model simulations')

        os.makedirs(DATA_DIR, exist_ok=True)

        scriptPath = os.path.join(MODEL_DIR, f'{args.model}.py')
        ret = subprocess.run(['mpirun', '-n', f'{args.n}', args.python, scriptPath])
        ret.check_returncode()

        print('OK')
    if args.command in ['render', 'all']:
        dataPath = os.path.join(DATA_DIR, args.model)
        commonParams = [
            args.python, '-m', 'stepsblender.load', dataPath, '--blenderPath', args.blenderPath, '--render'
        ]

        # Default render
        printStage('Run default rendering')
        framesPath = os.path.join(FRAMES_DIR, f'{args.model}_default_render')
        ret = subprocess.run(
            commonParams +
            ['--outputPath', framesPath, '--renderStart', f'{args.tstart}', '--renderEnd', f'{args.tend}'])
        ret.check_returncode()

        # additional steps render
        printStage('Run time interpolation rendering')
        tscale = 5
        dataPath = os.path.join(DATA_DIR, args.model)
        framesPath = os.path.join(FRAMES_DIR, f'{args.model}_multisteps_render')
        start = tscale * args.tstart
        end = tscale * args.tend
        ret = subprocess.run(commonParams + [
            '--outputPath', framesPath, '--timeScale', f'{tscale}', '--renderStart', f'{start}',
            '--renderEnd', f'{end}'
        ])
        ret.check_returncode()

        print('OK')
