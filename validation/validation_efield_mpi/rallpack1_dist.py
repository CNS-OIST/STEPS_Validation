# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# -*- coding: utf-8 -*-
#
# Rallpack1 model
# Author Iain Hepburn
#
# Adapted for MPI testing with TetOpSplit.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import steps.quiet
import steps.model as smodel
import steps.geom as sgeom
import steps.rng as srng
import steps.utilities.meshio as meshio
import steps.mpi.solver as ssolver
import steps.utilities.geom_decompose as gd

import numpy as np
import numpy.linalg as la
import operator
import time

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

sim_parameters = {
## Rallpack1:
    'R_A'  :    1.0,          # axial resistivity Ω·m
    'R_M'  :    4.0,          # membrane resistivity Ω·m²
    'C_M'  :    0.01,         # membrane capacity F/m²
    'E_M'  :   -0.065,        # p.d. across membrane V
    'Iinj' :    0.1e-9,       # injection current A
    'diameter': 1.0e-6,       # cylinder diameter m
    'length':   1.0e-3,       # cylinder length m
# STEPS
    'sim_end':  0.25,         # simulation stop time s
    'EF_dt':    5.0e-5,       # E-field evaluation time step s
    'EF_solver': ssolver.EF_DV_PETSC
}


def print0(string):
    if steps.mpi.rank == 0:
        print str(string)


def ROIset(x):
    return set_uint(list(x))

def boundary_tets(mesh):
    return (i for i in range(mesh.ntets)
            if any([nb == -1 for nb in mesh.getTestNeighb(i)]))

def boundary_tris(mesh):
    def btris(tet):
        return [mesh.getTetTriNeighb(tet)[face] for face in range(4)
                if mesh.getTetTetNeighb(tet)[face]==-1]

    return (tri for tet in range(mesh.ntets) for tri in btris(tet))

def zminmax_tris(mesh):
    minz = mesh.getBoundMin()[2]
    maxz = mesh.getBoundMax()[2]
    zeps = (maxz - minz)/mesh.ntets

    minz_check = minz + zeps
    maxz_check = maxz - zeps

    zmin_tris = []
    zmin_vset = set()
    zmax_tris = []
    zmax_vset = set()

    for tri in boundary_tris(mesh):
        vertices = mesh.getTri(tri)
        ptverts = [mesh.getVertex(vi) for vi in vertices]

        if all([v[2] <= minz_check for v in ptverts]):
            zmin_tris.append(tri)
            for v in vertices: zmin_vset.add(v)
        elif all([v[2] >= maxz_check for v in ptverts]):
            zmax_tris.append(tri)
            for v in vertices: zmax_vset.add(v)

    return (zmin_tris,zmin_vset,zmax_tris,zmax_vset)

# Find the Z-axially closest and furthest vertices

def radial_extrema(geom, vset):
    r2min = r2max = None
    vmin  = vmax  = None

    for v in vset:
        x = geom.getVertex(v)
        s = x[0]*x[0] + x[1]*x[1]
        if r2min > s or r2min == None:
            r2min = s
            vmin = v
        if r2max < s or r2max == None:
            r2max = s
            vmax = v

    return (vmin,vmax)

# If a property must be held by both tetrahedral neighbours of triangles
# in a set S, then for each such triangle t0 there is a consistency neighbourhood
# of tets N(t) such that tet T in N(t) if there exists a path
# t0,T0,t1,T1,...,tn,Tn=T for some n, where ti is the shared face of Ti and T(i+1).
# The function consistent_neighbourhood_part partitions S into a disjoint union
# S1, S2, ..., Sk such that all the neighbouring tets of triangles in Si belong
# to the same consistency neighbourhood.

def consistent_neighbourhood_part(mesh, tri_set):
    unvisited = set(tri_set)
    parts = []

    while unvisited:
        tri = unvisited.pop()
        tet_queue = [tet for tet in mesh.getTriTetNeighb(tri) if tet >= 0]
        tri_part = [tri]

        while tet_queue:
            tet = tet_queue.pop()
            for face in [tri for tri in mesh.getTetTriNeighb(tet) if tri in unvisited]:
                unvisited.remove(face)
                tri_part.append(face)
                for other_tet in [tet2 for tet2 in mesh.getTriTetNeighb(face) if tet2 != tet and tet2 !=-1]:
                    tet_queue.append(other_tet)

        parts.append(tri_part)

    return parts

# Make host partition for mesh tets and specified triangles, respecting requirement that
# tets separated by a triangle in the given set must belong to the same host.
# Returns (tet_hosts, tri_hosts).

def host_assignment_by_axis(mesh, tri_set):
    def tet_neighbs(tri):
        return [tet for tet in mesh.getTriTetNeighb(tri) if tet != -1]

    tet_hosts = gd.binTetsByAxis(mesh, steps.mpi.nhosts)
    tri_hosts = {}

    for part in consistent_neighbourhood_part(mesh, tri_set):
        tets = set.union(*(set(tet_neighbs(tri)) for tri in part))
        if not tets: continue

        tet0 = tets.pop()
        host = tet_hosts[tet0]
        for tri in part: tri_hosts[tri] = host
        for tet in tets: tet_hosts[tet] = host

    return (tet_hosts,tri_hosts)

# Build steps geom object, and report lower and upper surface vertixes
# for later use.

def build_geometry(mesh_path, file_format="xml", scale=1.):

    scale = float(scale)

    if file_format == "xml":
        mesh = meshio.loadMesh(mesh_path)[0]
    elif file_format == "msh":
        mesh = meshio.importGmsh(mesh_path+".msh", scale)[0]
    elif file_format == "inp":
        mesh = meshio.importAbaqus(mesh_path+".inp", scale)[0]
    else :
        raise TypeError("File format "+str(file_format)+" not available: choose among xml, msh, inp" )

    cyto = sgeom.TmComp('cyto', mesh, range(mesh.ntets))

    (zmin_tris,zmin_vset,zmax_tris,zmax_vset) = zminmax_tris(mesh)
    memb_tris = list(mesh.getSurfTris())
    for tri in zmin_tris: memb_tris.remove(tri)
    for tri in zmax_tris: memb_tris.remove(tri)
    memb = sgeom.TmPatch('memb', mesh, memb_tris, cyto)
    membrane = sgeom.Memb('membrane', mesh, [memb] )

    mesh.addROI('v_zmin',sgeom.ELEM_VERTEX,zmin_vset)
    mesh.addROI('v_zmin_sample',sgeom.ELEM_VERTEX,radial_extrema(mesh,zmin_vset))
    mesh.addROI('v_zmax_sample',sgeom.ELEM_VERTEX,radial_extrema(mesh,zmax_vset))
    return mesh


def build_model(mesh, param):
    mdl = smodel.Model()
    memb = sgeom.castToTmPatch(mesh.getPatch('memb'))

    ssys = smodel.Surfsys('ssys', mdl)
    memb.addSurfsys('ssys')

    L = smodel.Chan('L', mdl)
    Leak = smodel.ChanState('Leak', mdl, L)

    # membrane conductance
    area_cylinder = np.pi * param['diameter'] * param['length']
    L_G_tot = area_cylinder / param['R_M']
    g_leak_sc = L_G_tot / len(memb.tris)
    OC_L = smodel.OhmicCurr('OC_L', ssys, chanstate = Leak, erev = param['E_M'], g = g_leak_sc)

    return mdl


def init_sim(model, mesh, seed, param):
    rng = srng.create('r123', 512)
    # previous setup
    # rng.initialize(seed)
    rng.initialize(steps.mpi.rank + 1000)

    memb = sgeom.castToTmPatch(mesh.getPatch('memb'))

    # partition geometry across hosts

    t0 = time.time()
    (tet_hosts,tri_hosts) = host_assignment_by_axis(mesh, memb.tris)
    t1 = time.time()
    print0(".... assign to host took " + str(t1-t0))
    # sim = ssolver.TetOpSplit(model, mesh, rng, True, tet_hosts, tri_hosts)
    sim = ssolver.TetOpSplit(model, mesh, rng, param['EF_solver'], tet_hosts, tri_hosts)
    t2 = time.time()
    print0(".... TetOpSplit c-tor took " + str(t2-t1))

    # Correction factor for deviation between mesh and model cylinder:
    area_cylinder = np.pi * param['diameter'] * param['length']
    area_mesh_factor = sim.getPatchArea('memb') / area_cylinder

    # Set initial conditions
    t0 = time.time()
    sim.reset()
    t1 = time.time()
    print0(".... sim.reset() took " + str(t1-t0))

    for t in memb.tris: sim.setTriCount(t, 'Leak', 1)

    sim.setEfieldDT(param['EF_dt'])
    sim.setMembPotential('membrane', param['E_M'])
    sim.setMembVolRes('membrane', param['R_A'])
    sim.setMembCapac('membrane', param['C_M']/area_mesh_factor)

    v_zmin = mesh.getROIData('v_zmin')
    I = param['Iinj']/len(v_zmin)
    for v in v_zmin: sim.setVertIClamp(v, I)
    t2 = time.time()
    print0(".... setting all the others took " + str(t2-t1))

    return sim

# Run simulation and sample potential every dt until t_end
def run_sim(sim, dt, t_end, vertices, verbose=False):
    N = int(np.ceil(t_end/dt))+1
    result = np.zeros((N, len(vertices)))
    nvert = len(vertices)

    for l in xrange(N):
        if verbose and steps.mpi.rank == 0: print "sim time (ms): ", dt*l*1.0e3
        t1 = time.time()
        sim.run(l*dt)
        t2 = time.time()
        print0("~~~~~ run time step  [done in "+ str(t2-t1)  +" sec]  ~~~~~")
        result[l,:] = [sim.getVertV(v) for v in vertices]

    return result


# Returns RMS error, table containing computed end-point voltages
# and reference voltage data.

def run_comparison(seed, mesh_file, mesh_format, mesh_scale, v0_datafile, v1_datafile, verbose=False):
    # sample at same interval as rallpack1 reference data
    sim_dt = 5.0e-5

    def snarf(fname):
        F = open(fname, 'r')
        for line in F: yield tuple([float(x) for x in line.split()])
        F.close()

    vref_0um = np.array([v for (t,v) in snarf(v0_datafile)])
    vref_1000um = np.array([v for (t,v) in snarf(v1_datafile)])

    #mesh_file = "/gpfs/bbp.cscs.ch/project/proj40/meshes_steps_scaling/cylinder_1246549verts.msh"

    print0("~~~~~ build geometry [start] ~~~~~")
    print0("loading mesh located in: " + mesh_file)
    t1 = time.time()
    geom = build_geometry(mesh_file, mesh_format, scale=mesh_scale)
    t2 = time.time()
    print0("~~~~~ build geometry [done in "+ str(t2-t1)  +" sec]  ~~~~~")
    print0("~~~~~ build model    [start] ~~~~~")
    t1 = time.time()
    model = build_model(geom, sim_parameters)
    t2 = time.time()
    print0("~~~~~ build model    [done in "+ str(t2-t1)  +" sec]  ~~~~~")
    print0("~~~~~ initialize sim [start] ~~~~~")
    t1 = time.time()
    sim = init_sim(model, geom, seed, sim_parameters)
    t2 = time.time()
    print0("~~~~~ initialize sim [done in "+ str(t2-t1)  +" sec]  ~~~~~")

    # grab sample vertices
    zmin_sample = geom.getROIData('v_zmin_sample')
    n_zmin_sample = len(zmin_sample)
    zmax_sample = geom.getROIData('v_zmax_sample')
    n_zmax_sample = len(zmax_sample)
    vertices =  zmin_sample + zmax_sample

    result = run_sim(sim, sim_dt, sim_parameters['sim_end'], vertices, verbose=verbose)

    vmean_0um = np.mean(result[:,0:n_zmin_sample], axis=1)
    vmean_1000um = np.mean(result[:,n_zmin_sample:], axis=1)
    npt = min(len(vmean_0um),len(vref_0um))

    data = np.zeros((5,npt))
    data[0,:] = np.linspace(0, stop=npt*sim_dt, num=npt, endpoint=False)
    data[1,:] = vmean_0um[0:npt]
    data[2,:] = vref_0um[0:npt]
    data[3,:] = vmean_1000um[0:npt]
    data[4,:] = vref_1000um[0:npt]

    # rms difference
    err_0um = vref_0um[0:npt] - vmean_0um[0:npt]
    rms_err_0um = la.norm(err_0um)/np.sqrt(npt)

    err_1000um = vref_1000um[0:npt] - vmean_1000um[0:npt]
    rms_err_1000um = la.norm(err_1000um)/np.sqrt(npt)

    return data, rms_err_0um, rms_err_1000um

