import rallpack1_dist
import argparse
import sys
import os.path as path
import steps.mpi
import numpy as np

# By default, meshes are searched for under meshes/
# and reference data under data/rallpack1_correct

class ArgumentParser0(argparse.ArgumentParser):
    def error(self,message):
        if steps.mpi.rank == 0:
            super(ArgumentParser0, self).error(message)
        else:
            sys.exit(2)

    def print_help(self):
        if steps.mpi.rank == 0:
            super(ArgumentParser0, self).print_help()

parser = ArgumentParser0(description='Run Rallpack test 1')
parser.add_argument('--seed', '-seed',  type=int,
                    default=7,
                    help='seed for RNG')
parser.add_argument('--meshdir', '-M',
                    default='meshes',
                    help='directory containing mesh files')
parser.add_argument('--mesh', '-m',
                    default='axon_cube_L1000um_D866nm_1978tets',
                    help='mesh file name')
parser.add_argument('--meshfmt',
                    default='xml',
                    help='mesh file format')
parser.add_argument('--meshscale',
                    default=1,
                    help='mesh scaling factor')
parser.add_argument('--datadir', '-D',
                    default='data/rallpack1_correct',
                    help='directory containing validation data files v0 an vx')
parser.add_argument('--out', '-o', metavar='FILE',
                    default=None,
                    help='write simulation data to FILE in CSV format')
parser.add_argument('--plot', '-p', metavar='FILE', nargs='?',
                    default=False, const=None,
                    help='plot results')
parser.add_argument('--verbose', '-v', action='count',
                    help='verbose output')
parser.add_argument('--param', '-P', action='append',
                    help='override simulation parameter')
parser.add_argument('--dump-param', action='store_true',
                    help='display simulation parameters and exit')

args = parser.parse_args()

if args.param != None:
    for k,v in [kv.split('=') for kv in args.param]:
        rallpack1_dist.sim_parameters[k]=type(rallpack1_dist.sim_parameters[k])(v)

meshfile = path.join(args.meshdir,args.mesh)
v0data = path.join(args.datadir,'v0')
v1data = path.join(args.datadir,'vx')

if args.dump_param and steps.mpi.rank == 0:
    for k,v in rallpack1.sim_parameters.iteritems():
        print k+'='+str(v)

if steps.mpi.rank == 0:
    print "mesh_scale = " + str(args.meshscale)

simdata, rms_err_0um, rms_err_1000um = rallpack1_dist.run_comparison(args.seed,meshfile,args.meshfmt,args.meshscale,v0data,v1data,verbose=args.verbose)

# Print table of results and reference data
if steps.mpi.rank == 0:
    if args.out != None:
        npt = simdata.shape[1]
        out = sys.stdout
        if args.out != '-':
            out = open(args.out,"w")

        print >>out,"t,steps_0um,ref_0um,steps_1000um,ref_1000um"
        np.savetxt(out,simdata.T,fmt='%.7g',delimiter=',')
        print >>out,"\n"
        if args.out != '-':
            out.close()

    print 'rms error 0um:    %.7g mV' % (rms_err_0um*1.0e3)
    print 'rms error 1000um: %.7g mV' % (rms_err_1000um*1.0e3)

    if args.plot != False:
        import matplotlib.pyplot as plt

        # Transform results to msec and mV
        simdata *= 1.0e3

        plt.subplot(211)
        plt.plot(simdata[0,:], simdata[2,:], 'k-' ,label = 'Correct, 0um', linewidth=3)
        plt.plot(simdata[0,:], simdata[1,:], 'r--', label = 'STEPS, 0um', linewidth=3)
        plt.legend(loc='best')
        plt.ylabel('Potential (mV)')
        plt.subplot(212)
        plt.plot(simdata[0,:], simdata[4,:], 'k-' ,label = 'Correct, 1000um', linewidth=3)
        plt.plot(simdata[0,:], simdata[3,:], 'r--', label = 'STEPS, 1000um', linewidth=3)
        plt.legend(loc='best')
        plt.ylabel('Potential (mV)')
        plt.xlabel('Time (ms)')

        if False: #args.plot != None:
            plt.savefig(args.plot)
        else:
            plt.show()

