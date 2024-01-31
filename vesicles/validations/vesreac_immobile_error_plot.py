import steps.interface

from steps.saving import HDF5Handler
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import numpy as np
import os
import sys

if __name__ == '__main__':
    hdfPrefix = 'data/vesreac_immobile_reactants'
    if len(sys.argv) > 1:
        hdfPrefix = sys.argv[1]

    allConcs = {}
    i = 0
    while os.path.isfile(hdfPrefix + f'_{i}.h5'):
        print(f"loading {hdfPrefix + f'_{i}.h5'}")
        with HDF5Handler(hdfPrefix + f'_{i}') as hdf:
            for group in hdf:
                globals().update(group.parameters)
                concs, = group.results
                time = concs.time[0]
                key = (DCST, vesDt, voltet)
                allConcs.setdefault(key, [])
                allConcs[key] += list(concs.data[:,:,0])
        i += 1
    for key, vals in allConcs.items():
        allConcs[key] = np.mean(vals, axis=0)
    allDCST, allVesDt, allVolTets = [sorted(set(vals)) for vals in zip(*allConcs.keys())]

    # Compute rates
    allrates = {}
    for voltet in allVolTets:
        for vesDt in allVesDt:
            for DCST in allDCST:
                meanconc = allConcs[(DCST, vesDt, voltet)]
                analytical = lambda t, k: CONC_INIT * np.exp(-t * k * NB_A / volfact)
                (fitted_rate, *_), _ = curve_fit(analytical, time, meanconc, p0=KCST)
                allrates[(DCST, vesDt, voltet)] = fitted_rate

    # Errors on rates
    for DCST in allDCST:
        fig = plt.figure()
        rates = []
        for voltet in allVolTets:
            gt_rate = allrates[(DCST, allVesDt[0], voltet)]
            rates.append([abs(allrates[(DCST, vesDt, voltet)] - gt_rate) / gt_rate * 100 for vesDt in allVesDt])
        rates = np.array(rates)
        tetsizes = (np.array(allVolTets) * 3 / 4 / np.pi) ** (1/3)
        y, x = np.meshgrid(tetsizes * 1e9, allVesDt)
        c = plt.pcolormesh(x, y, rates.T, vmin=0, vmax=20)
        threshold = [np.sqrt(15*DCST*vdt)*1e9 for vdt in allVesDt]
        plt.plot(allVesDt, threshold, 'r', linewidth=2)
        plt.xscale('log')
        plt.ylabel('Tetrahedron radius $r_{tet}$ [nm]')
        plt.xlabel('Vesicle $\Delta t$ [s]')
        plt.ylim([min(tetsizes*1e9), max(tetsizes*1e9)])
        plt.colorbar(c, label='Error on estimated rate [%]', extend='max')
        fig.set_size_inches(3.7, 3.4)
        fig.savefig(f'plots/vesreac_immobile_reactants_error_DCST_{DCST}.pdf',
                    dpi=300, bbox_inches='tight')
        plt.close()
