import pickle
from pylab import *

DT = 0.0002
T_END = 0.3
MESHFILE = 'MFT_wmito0.28_cylinder_565958tets_19042022.inp'
scale = 1

ifile = open(
    'data/rothman_' + str(MESHFILE) + '_' + str(scale) + '_' + str(DT) + '_' +
    str(T_END), 'rb')

mito_vol_frac = pickle.load(ifile)
tpnts_6 = pickle.load(ifile)
ves_mean = pickle.load(ifile)
tpnts = tpnts_6 / 6.0

plot(tpnts, ves_mean[1:] / (tpnts_6), linewidth=3)
xlabel('Time (s)')
ylabel('D ($\mu$$m^2$/s)')

ylim(0, 0.06)
fig = gcf()
fig.set_size_inches(3.4, 3.4)
fig.savefig("plots/rothman.pdf", dpi=300, bbox_inches='tight')
close()
