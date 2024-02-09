import steps.interface

from steps.saving import HDF5Handler
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

mesh_ntets = [291, 577, 991, 2088, 3414, 11773, 41643, 265307]
mesh_sizetets = [119.77, 96.58, 81.19, 63.76, 54.21, 36.03, 23.68, 12.79]
ntets2sizetets = {ntet: sztet for ntet,
                  sztet in zip(mesh_ntets, mesh_sizetets)}

if __name__ == '__main__':
    hdfPrefix = 'data/vesreac_error'

    allValues = {}
    i = 0
    while os.path.isfile(hdfPrefix + f'_{i}.h5'):
        print(f"loading {hdfPrefix + f'_{i}.h5'}")
        with HDF5Handler(hdfPrefix + f'_{i}') as hdf:
            for group in hdf:
                globals().update(group.parameters)
                for rs, rskey in zip(group.results, ['foi', 'soAA']):
                    time = rs.time[0]
                    key = (ntets, D, rskey)
                    allValues.setdefault(key, [])
                    allValues[key] += list(rs.data[:, :, 0])
        i += 1
    for key, vals in allValues.items():
        allValues[key] = np.mean(vals, axis=0)
    allNtets, allD, _ = [sorted(set(vals)) for vals in zip(*allValues.keys())]

    allErrors = {}
    for (ntets, D, tpe), meanVal in allValues.items():
        if tpe == 'foi':
            def foi_findrate(x, K): return spec_A_foi_N * np.exp(-K * x)
            poptfoi, _ = curve_fit(foi_findrate, time, meanVal, p0=KCST_foi)
            allErrors[(ntets, D, tpe)] = 100*abs(poptfoi[0]-KCST_foi)/KCST_foi
        elif tpe == 'soAA':
            def soAA_findrate(x, K): return (
                1.0 / CONCA_soAA + ((x * K))) * 1e-6
            poptAAv, _ = curve_fit(soAA_findrate, time,
                                   (1.0 / meanVal) * 1e-6, p0=KCST_soAA)
            allErrors[(ntets, D, tpe)] = 100 * \
                abs(poptAAv[0]-KCST_soAA)/KCST_soAA

    lw = 3
    sizetets = [ntets2sizetets[ntets] for ntets in allNtets]

    foi_error = [allErrors[(ntet, 0, 'foi')] for ntet in allNtets]
    plt.plot(sizetets, foi_error, linewidth=lw, marker='o')
    plt.xlim(0, 130)
    plt.gca().invert_xaxis()
    plt.ylim(0, 2)
    plt.xlabel("Average tetrahedron size (nm)")
    plt.ylabel("Error in 2nd order reaction rate (%)")
    fig = plt.gcf()
    fig.set_size_inches(3.4, 3.4)
    fig.savefig("plots/vesreac_error_size_foi.pdf",
                dpi=300, bbox_inches='tight')
    plt.close()

    so_error_D0 = [allErrors[(ntet, 0, 'soAA')] for ntet in allNtets]
    so_error_D0_1 = [allErrors[(ntet, 0.1, 'soAA')] for ntet in allNtets]
    plt.plot(sizetets, so_error_D0,
             label='D=0$\mu m^2s^{-1}$', linewidth=lw, marker='o')
    plt.plot(sizetets, so_error_D0_1,
             label='D=0.1$\mu m^2s^{-1}$', linewidth=lw, marker='o')
    plt.legend()
    plt.xlim(0, 130)
    plt.gca().invert_xaxis()
    plt.ylim(0, 15)
    plt.xlabel("Average tetrahedron size (nm)")
    plt.ylabel("Error in 2nd order reaction rate (%)")
    fig = plt.gcf()
    fig.set_size_inches(3.4, 3.4)
    fig.savefig("plots/vesreac_error_size.pdf", dpi=300, bbox_inches='tight')
    plt.close()
