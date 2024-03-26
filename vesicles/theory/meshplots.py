import steps.utilities.meshio as smeshio
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"


def dist(p1, p2):
    return np.sqrt( pow(p1[0]-p2[0], 2) + pow(p1[1]-p2[1], 2) + pow(p1[2]-p2[2], 2))

for (meshfile, scale, xmin, xmax, dx) in [('pyr_axon_2021_cyt.inp',1e-6, 0, 100, 1),
                                            ('sphere_2D_291tets.inp',0.25e-6, 80, 180, 2),
                                                ('sphere_2D_265307tets.inp', 0.25e-6, 5, 25, 0.2)]:

    mesh = smeshio.importAbaqus('../validations/meshes/' + meshfile, scale)[0]

    average_edge = []

    for t in range(mesh.ntets):
        v1idx, v2idx, v3idx, v4idx = mesh.getTet(t)
    
        v1 = mesh.getVertex(v1idx)
        v2 = mesh.getVertex(v2idx)
        v3 = mesh.getVertex(v3idx)
        v4 = mesh.getVertex(v4idx)
    
        average_edge.append(np.mean([dist(v1, v2), dist(v1, v3), dist(v1, v4), dist(v2, v3), dist(v2, v4), dist(v3, v4)])*1e9)
    
    meshbins = np.arange(xmin, xmax+1, dx)

    plt.hist(average_edge, label='Average edge', bins=meshbins, alpha=0.7)
    plt.xlabel("Tetrahedron average edge length (nm)")
    plt.ylabel("Bin count")
    plt.xlim(xmin,xmax)
    fig = plt.gcf()
    fig.set_size_inches(3.4, 3.4)
    plt.savefig('plots/'+meshfile + '_tetlengths_average.pdf', dpi=300, bbox_inches='tight')
    plt.close()


