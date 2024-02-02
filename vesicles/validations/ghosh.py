import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad


#unormalised for now
def Qnew(beta, sigma):
    return ((2.0) / sigma**2) * np.sqrt(
        beta * np.sin(beta)) * np.exp(-(beta**2 / sigma**2))


def sig(tau):
    return np.sqrt(2 * tau)


def Q(tau, theta):
    return (1.0 / tau) * np.sqrt(theta * np.sin(theta)) * np.exp(-(theta**2) /
                                                                 (2 * tau))


model = Model()
with model:
    ssys = SurfaceSystem.Create()

    SA = Species.Create()

    with ssys:
        # Give surface diffusion coefficient of 1uM^2/s
        Diffusion(SA, 1e-12)

# Convert to radius of 50nm
mesh = TetMesh.LoadAbaqus('meshes/sphere_rad1_64ktets.inp', 50e-9)

# Pre-process some arc distances
surf_tri_arc = np.zeros(len(mesh.surface))
surf_tri0_xyz = mesh.surface[0].center

for i, tri in enumerate(mesh.surface):
    chord_length = np.linalg.norm(tri.center - surf_tri0_xyz)
    # Don't forget to normalise chord length
    surf_tri_arc[i] = 2 * np.arcsin(chord_length / 100e-9)

surf_tri_arc = np.array(surf_tri_arc)

with mesh:
    comp = Compartment.Create(mesh.tets)
    surf = Patch.Create(mesh.surface, comp, None, ssys)

rng = RNG('mt19937', 1000, 133)
sim = Simulation('Tetexact', model, mesh, rng, MPI.EF_NONE)

NITER = 100000

tpnts = np.arange(0, 0.0032, 0.0001)
res = np.zeros((NITER, len(tpnts)))

resQ_mean = np.zeros(len(tpnts))
resQ_std = np.zeros(len(tpnts))
resQ = []
beta = np.arange(0, np.pi, 0.01)

rs = ResultSelector(sim)

counts = rs.TRIS(mesh.surface).SA.Count

sim.toSave(counts)

for n in range(NITER):
    if MPI.rank == 0 and not n % 1000:
        print(n, 'of', NITER)

    sim.newRun()
    sim.TRI(mesh.surface[0]).SA.Count = 1

    for tidx in range(len(tpnts)):
        t = tpnts[tidx]
        sim.run(t)
        counts.save()
        res[n][tidx] = surf_tri_arc[counts.data[-1, -1, :] == 1][0]

    counts.clear()
    counts._dataHandler.saveData = []
    counts._dataHandler.saveTime = []

Qns = [1] + [0] * (len(beta) - 1)
resQ.append(Qns)
for tidx in range(1, len(tpnts)):
    t = tpnts[tidx]

    tau = (2 * 1e-12 * t) / 50e-9**2
    Qtau = lambda theta: Q(tau, theta)
    invN, err = quad(Qtau, 0, np.pi)

    Qns = []
    sigma = sig(tau)
    for b in beta:
        Qns.append(Qnew(b, sigma) / invN)

    resQ.append(Qns)

    resQ_mean[tidx] = sum(Qns * beta) / sum(Qns)

    #calculating the std is a bit trickier
    beta_shift = beta - resQ_mean[tidx]

    devs2 = beta_shift * beta_shift
    sumdevs2 = sum(np.array(Qns) * devs2)
    normsumdevs2 = sumdevs2 / sum(Qns)
    sqsumdevs2 = np.sqrt(normsumdevs2)
    resQ_std[tidx] = sqsumdevs2

lw = 3

tidx = 10
plt.hist(res[:, tidx], label='STEPS model', density=True, bins=20)
plt.plot(beta, resQ[tidx], label='Ghosh', linewidth=lw)
plt.title("Time: " + str(1e3 * tpnts[tidx]) + 'ms')
plt.ylabel('Q')
plt.xlabel('Angular displacement (rad)')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/ghosh_t1.pdf", dpi=300, bbox_inches='tight')
plt.close()

tidx = 20
plt.hist(res[:, tidx], label='STEPS model', density=True, bins=20)
plt.plot(beta, resQ[tidx], label='Ghosh', linewidth=lw)
plt.title("Time: " + str(1e3 * tpnts[tidx]) + 'ms')
plt.ylabel('Q')
plt.xlabel('Angular displacement (rad)')
plt.legend(loc='best')
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/ghosh_t2.pdf", dpi=300, bbox_inches='tight')
plt.close()

tidx = 30
plt.hist(res[:, tidx], label='STEPS model', density=True, bins=20)
plt.plot(beta, resQ[tidx], label='Ghosh', linewidth=lw)
plt.title("Time: " + str(1e3 * tpnts[tidx]) + 'ms')
plt.ylabel('Q')
plt.xlabel('Angular displacement (rad)')
plt.legend(loc='best')
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/ghosh_t3.pdf", dpi=300, bbox_inches='tight')
plt.close()

res_mean = np.mean(res, axis=0)
res_std = np.std(res, axis=0)

plt.errorbar(tpnts * 1e3, res_mean, res_std, label='STEPS model', linewidth=2)
plt.errorbar((0.00004 + tpnts) * 1e3,
             resQ_mean,
             resQ_std,
             label='Ghosh',
             linewidth=2)
plt.xlabel('Time(ms)')
plt.ylabel('Angular displacement (rad)')
plt.ylim(0, 2.5)
plt.legend(loc=2)
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/ghosh_AD1.pdf", dpi=300, bbox_inches='tight')
plt.close()

plt.plot(tpnts * 1e3, res_mean**2, label='STEPS model', linewidth=lw)
plt.plot(tpnts * 1e3,
         resQ_mean**2,
         linestyle='--',
         label='Ghosh',
         linewidth=lw)
plt.xlabel('Time(ms)')
plt.ylabel('Angular displacement$^2$ (rad$^2$)')
plt.ylim(0, 3)
plt.legend(loc='best')
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/ghosh_AD2.pdf", dpi=300, bbox_inches='tight')
plt.close()
