import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

#unormalised for now
def Qnew(beta, sigma):
    return ((2.0) / sigma**2) * np.sqrt(
        beta * np.sin(beta)) * np.exp(-(beta**2 / sigma**2))


def sig(tau):
    return np.sqrt(2 * tau)


def Q(tau, theta):
    return (1.0 / tau) * np.sqrt(theta * np.sin(theta)) * np.exp(-(theta**2) /
                                                                 (2 * tau))
def v_u(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
    
ves_radius = 25e-9
dcst = 0.11e-12
dt=0.0001

model = Model()
with model:
    
    vsys = VolumeSystem.Create()
    vssys = VesicleSurfaceSystem.Create()

    Ves1 = Vesicle.Create(ves_radius*2, 0, vssys)

    SA = Species.Create()

    with vssys:
        # Give surface diffusion coefficient of DCST
        Diffusion(SA, dcst)
                
mesh = TetMesh.LoadAbaqus('meshes/sphere_rad1_64ktets.inp', 1e-6)

with mesh:
    comp = Compartment.Create(mesh.tets)

rng = RNG('mt19937', 1000, 133)
sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

#NITER = 100000 # better, but slow
NITER = 10000

tpnts = np.arange(0, 0.0032, dt)

resQ_mean = np.zeros(len(tpnts))
resQ_std = np.zeros(len(tpnts))
resQ = []
beta = np.arange(0, np.pi, 0.01)

rs = ResultSelector(sim)
rs_SApos = rs.comp.VESICLES()('surf').SA.Pos

sim.toSave(rs_SApos, dt=dt)


for n in range(NITER):
    if MPI.rank == 0 and not n % 1:
        print(n, 'of', NITER)

    sim.newRun()
    sim.setVesicleDT(dt)
    
    v = sim.comp.addVesicle(Ves1)
    v.Pos = [0,0,0] # make the vector calculation nice and easy
    v('surf').SA.Count = 1
    
    for t in tpnts:
        sim.run(t)


res_data = rs_SApos.data[:,:,0]

res = np.zeros(res_data.shape)

for i in range(NITER):
    pos0 = res_data[i][0][0][0]
    for j in range(len(tpnts)):
        pos = res_data[i][j][0][0] # the two zeros are vesicle and species index
        res[i][j] = np.arccos(np.clip(np.dot(v_u(pos), v_u(pos0)), -1.0, 1.0))


Qns = [1] + [0] * (len(beta) - 1)
resQ.append(Qns)
for tidx in range(1, len(tpnts)):
    t = tpnts[tidx]

    tau = (2 * dcst * t) / ves_radius**2
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
fig.savefig("plots/ghosh_vessurf_t1.pdf", dpi=300, bbox_inches='tight')
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
fig.savefig("plots/ghosh_vessurf_t2.pdf", dpi=300, bbox_inches='tight')
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
fig.savefig("plots/ghosh_vessurf_t3.pdf", dpi=300, bbox_inches='tight')
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
fig.savefig("plots/ghosh_vessurf_AD1.pdf", dpi=300, bbox_inches='tight')
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
fig.savefig("plots/ghosh_vessurf_AD2.pdf", dpi=300, bbox_inches='tight')
plt.close()
