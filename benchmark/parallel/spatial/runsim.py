####################################################################################
#
#     STEPS - STochastic Engine for Pathway Simulation
#     Copyright (C) 2007-2019 Okinawa Institute of Science and Technology, Japan.
#     Copyright (C) 2003-2006 University of Antwerp, Belgium.
#
#     See the file AUTHORS for details.
#     This file is part of STEPS.
#
#     STEPS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License version 2,
#     as published by the Free Software Foundation.
#
#     STEPS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#################################################################################                                                                     

#  This script is adapted from the following script

#  https://github.com/CNS-OIST/STEPS_Example/blob/master/publication_models/Anwar_J%20Neurosci_2013/StochasticCaburst_dendrite.py
#  H Anwar, I Hepburn, H Nedelescu, W Chen and E De Schutter
#  Stochastic calcium mechanisms cause dendritic calcium spike variability
#  J Neuroscience 2013
#
#  *StochasticCaburst_dendrite.py : The stochastic calcium burst
#  model, on the realistic dendrite morphology. Each dendritic branch is
#  modeled with well-mixed cytosolic compartments.

# This script has been modified for performance benchmark (Weiliang Chen, OIST, 2019)
                                                                                                                                                  
import steps
import math
import time
from random import *
import steps.model as smodel
import steps.geom as sgeom
import steps.rng as srng
import steps.utilities.meshio as meshio
import steps.utilities.metis_support as metis_support
import steps.mpi
import steps.mpi.solver as ssolver
import steps.utilities.geom_decompose as gd
from extra.constants_withampa_yunliang import *
import numpy as np

import sys
import os
import pickle

try:
    from cpuinfo import get_cpu_info
except:
    print("Please install py-cpuinfo module package.")
    sys.exit()

timestr = time.strftime("%Y%m%d")

if steps.mpi.rank == 0:
    benchmark_file = open("benchmark/ver_%s" % steps.__version__ + "_" + timestr + ".csv", 'a')
    benchmark_file.write("Test Date," + timestr + "\n")
    benchmark_file.write("CPU Info\n")
    for key, value in get_cpu_info().items():
        benchmark_file.write("{0},{1}\n".format(key, value))
    benchmark_file.write("Num. Procs,%i" % (steps.mpi.nhosts))
    benchmark_file.write("\n\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
NTIMEPOINTS = 5
BK_ro=BK_ro*1.5

mesh_file = 'purkinje/10-2012-02-09-001.CNG'

segment_file = open("meshes/" + mesh_file+'_segment_info.txt', 'r')

smooth_tris = pickle.load(segment_file)
smooth_tets = pickle.load(segment_file)
spiny_tris = pickle.load(segment_file)
spiny_tets = pickle.load(segment_file)

branch_mapping_file = open("meshes/" + mesh_file+'_tetmap', 'r')
tet_branches = pickle.load(branch_mapping_file)

######Glutamate transient#######
# Reference (Rudolph et al. 2011)
#Units (mM)
f1 = file("./extra/Glut_Pulse_MVR.dat","r")
l1 = f1.read()
d1 = l1.split()

Glut = [0.0]*25001

count = 0
for i in d1:
    Glut[count] = float(i)
    count = count+1

########################### BIOCHEMICAL MODEL ###############################
def gen_model(): 
    mdl = smodel.Model()

    # Calcium
    Ca = smodel.Spec('Ca', mdl)

    # Correct the calcium valence as a trick to prevent large calcium influx from large compensated calcium current
    Ca.setValence(2)

    # Pump
    Pump = smodel.Spec('Pump', mdl)
    # CaPump
    CaPump = smodel.Spec('CaPump', mdl)

    # iCBsf
    iCBsf = smodel.Spec('iCBsf', mdl)
    # iCBsCa
    iCBsCa = smodel.Spec('iCBsCa', mdl)
    # iCBCaf
    iCBCaf = smodel.Spec('iCBCaf', mdl)
    # iCBCaCa
    iCBCaCa = smodel.Spec('iCBCaCa', mdl)

    # CBsf
    CBsf = smodel.Spec('CBsf', mdl)
    # CBsCa
    CBsCa = smodel.Spec('CBsCa', mdl)
    # CBCaf
    CBCaf = smodel.Spec('CBCaf', mdl)
    # CBCaCa
    CBCaCa = smodel.Spec('CBCaCa', mdl)

    # PV
    PV = smodel.Spec('PV', mdl)
    # PVMg
    PVMg = smodel.Spec('PVMg', mdl)
    # PVCa
    PVCa = smodel.Spec('PVCa', mdl)
    # Mg
    Mg = smodel.Spec('Mg', mdl)

    # Vol/surface systems
    vsys = smodel.Volsys('vsys', mdl)
    ssys = smodel.Surfsys('ssys', mdl)

    diff_Ca = smodel.Diff('diff_Ca', vsys, Ca)
    diff_Ca.setDcst(DCST)
    diff_CBsf = smodel.Diff('diff_CBsf', vsys, CBsf)
    diff_CBsf.setDcst(DCB)
    diff_CBsCa = smodel.Diff('diff_CBsCa', vsys, CBsCa)
    diff_CBsCa.setDcst(DCB)
    diff_CBCaf = smodel.Diff('diff_CBCaf', vsys, CBCaf)
    diff_CBCaf.setDcst(DCB)
    diff_CBCaCa = smodel.Diff('diff_CBCaCa', vsys, CBCaCa)
    diff_CBCaCa.setDcst(DCB)
    diff_PV = smodel.Diff('diff_PV', vsys, PV)
    diff_PV.setDcst(DPV)
    diff_PVCa = smodel.Diff('diff_PVCa', vsys, PVCa)
    diff_PVCa.setDcst(DPV)
    diff_PVMg = smodel.Diff('diff_PVMg', vsys, PVMg)
    diff_PVMg.setDcst(DPV)

    #Pump
    PumpD_f = smodel.SReac('PumpD_f', ssys, ilhs=[Ca], slhs=[Pump], srhs=[CaPump])
    PumpD_f.setKcst(P_f_kcst)

    PumpD_b = smodel.SReac('PumpD_b', ssys, slhs=[CaPump], irhs=[Ca], srhs=[Pump])
    PumpD_b.setKcst(P_b_kcst)

    PumpD_k = smodel.SReac('PumpD_k', ssys, slhs=[CaPump], srhs=[Pump])
    PumpD_k.setKcst(P_k_kcst)

    #iCBsf-fast
    iCBsf1_f = smodel.Reac('iCBsf1_f', vsys, lhs=[Ca,iCBsf], rhs=[iCBsCa], kcst = iCBsf1_f_kcst)
    iCBsf1_b = smodel.Reac('iCBsf1_b', vsys, lhs=[iCBsCa], rhs=[Ca,iCBsf], kcst = iCBsf1_b_kcst)

    #iCBsCa
    iCBsCa_f = smodel.Reac('iCBsCa_f', vsys, lhs=[Ca,iCBsCa], rhs=[iCBCaCa], kcst = iCBsCa_f_kcst)
    iCBsCa_b = smodel.Reac('iCBsCa_b', vsys, lhs=[iCBCaCa], rhs=[Ca,iCBsCa], kcst = iCBsCa_b_kcst)

    #iCBsf_slow
    iCBsf2_f = smodel.Reac('iCBsf2_f', vsys, lhs=[Ca,iCBsf], rhs=[iCBCaf], kcst = iCBsf2_f_kcst)
    iCBsf2_b = smodel.Reac('iCBsf2_b', vsys, lhs=[iCBCaf], rhs=[Ca,iCBsf], kcst = iCBsf2_b_kcst)

    #iCBCaf
    iCBCaf_f = smodel.Reac('iCBCaf_f', vsys, lhs=[Ca,iCBCaf], rhs=[iCBCaCa], kcst = iCBCaf_f_kcst)
    iCBCaf_b = smodel.Reac('iCBCaf_b', vsys, lhs=[iCBCaCa], rhs=[Ca,iCBCaf], kcst = iCBCaf_b_kcst)

    #CBsf-fast
    CBsf1_f = smodel.Reac('CBsf1_f', vsys, lhs=[Ca,CBsf], rhs=[CBsCa], kcst = CBsf1_f_kcst)
    CBsf1_b = smodel.Reac('CBsf1_b', vsys, lhs=[CBsCa], rhs=[Ca,CBsf], kcst = CBsf1_b_kcst)

    #CBsCa
    CBsCa_f = smodel.Reac('CBsCa_f', vsys, lhs=[Ca,CBsCa], rhs=[CBCaCa], kcst = CBsCa_f_kcst)
    CBsCa_b = smodel.Reac('CBsCa_b', vsys, lhs=[CBCaCa], rhs=[Ca,CBsCa], kcst = CBsCa_b_kcst)

    #CBsf_slow
    CBsf2_f = smodel.Reac('CBsf2_f', vsys, lhs=[Ca,CBsf], rhs=[CBCaf], kcst = CBsf2_f_kcst)
    CBsf2_b = smodel.Reac('CBsf2_b', vsys, lhs=[CBCaf], rhs=[Ca,CBsf], kcst = CBsf2_b_kcst)

    #CBCaf
    CBCaf_f = smodel.Reac('CBCaf_f', vsys, lhs=[Ca,CBCaf], rhs=[CBCaCa], kcst = CBCaf_f_kcst)
    CBCaf_b = smodel.Reac('CBCaf_b', vsys, lhs=[CBCaCa], rhs=[Ca,CBCaf], kcst = CBCaf_b_kcst)

    #PVca
    PVca_f = smodel.Reac('PVca_f', vsys, lhs=[Ca,PV], rhs=[PVCa], kcst = PVca_f_kcst)
    PVca_b = smodel.Reac('PVca_b', vsys, lhs=[PVCa], rhs=[Ca,PV], kcst = PVca_b_kcst)

    #PVmg
    PVmg_f = smodel.Reac('PVmg_f', vsys, lhs=[Mg,PV], rhs=[PVMg], kcst = PVmg_f_kcst)
    PVmg_b = smodel.Reac('PVmg_b', vsys, lhs=[PVMg], rhs=[Mg,PV], kcst = PVmg_b_kcst)


    ###### CaP channel ##############

    CaPchan = smodel.Chan('CaPchan', mdl)

    CaP_m0 = smodel.ChanState('CaP_m0', mdl, CaPchan)
    CaP_m1 = smodel.ChanState('CaP_m1', mdl, CaPchan)
    CaP_m2 = smodel.ChanState('CaP_m2', mdl, CaPchan)
    CaP_m3 = smodel.ChanState('CaP_m3', mdl, CaPchan)


    CaPm0m1 = smodel.VDepSReac('CaPm0m1', ssys, slhs = [CaP_m0], srhs = [CaP_m1], k= lambda V: 1.0e3 *3.* alpha_cap(V*1.0e3)* Qt)
    CaPm1m2 = smodel.VDepSReac('CaPm1m2', ssys, slhs = [CaP_m1], srhs = [CaP_m2], k= lambda V: 1.0e3 *2.* alpha_cap(V*1.0e3)* Qt)
    CaPm2m3 = smodel.VDepSReac('CaPm2m3', ssys, slhs = [CaP_m2], srhs = [CaP_m3], k= lambda V: 1.0e3 *1.* alpha_cap(V*1.0e3)* Qt)

    CaPm3m2 = smodel.VDepSReac('CaPm3m2', ssys, slhs = [CaP_m3], srhs = [CaP_m2], k= lambda V: 1.0e3 *3.* beta_cap(V*1.0e3)* Qt)
    CaPm2m1 = smodel.VDepSReac('CaPm2m1', ssys, slhs = [CaP_m2], srhs = [CaP_m1], k= lambda V: 1.0e3 *2.* beta_cap(V*1.0e3)* Qt)
    CaPm1m0 = smodel.VDepSReac('CaPm1m0', ssys, slhs = [CaP_m1], srhs = [CaP_m0], k= lambda V: 1.0e3 *1.* beta_cap(V*1.0e3)* Qt)

    OC_CaP = smodel.GHKcurr('OC_CaP', ssys, CaP_m3, Ca, virtual_oconc = Ca_oconc, computeflux = True)
    OC_CaP.setP(CaP_P)


    ##### BK channel ################################


    BKchan = smodel.Chan('BKchan', mdl)

    BK_C0 = smodel.ChanState('BK_C0', mdl, BKchan)
    BK_C1 = smodel.ChanState('BK_C1', mdl, BKchan)
    BK_C2 = smodel.ChanState('BK_C2', mdl, BKchan)
    BK_C3 = smodel.ChanState('BK_C3', mdl, BKchan)
    BK_C4 = smodel.ChanState('BK_C4', mdl, BKchan)
    BK_O0 = smodel.ChanState('BK_O0', mdl, BKchan)
    BK_O1 = smodel.ChanState('BK_O1', mdl, BKchan)
    BK_O2 = smodel.ChanState('BK_O2', mdl, BKchan)
    BK_O3 = smodel.ChanState('BK_O3', mdl, BKchan)
    BK_O4 = smodel.ChanState('BK_O4', mdl, BKchan)


    BKCAC0 = smodel.SReac('BKCAC0', ssys, slhs = [BK_C0], ilhs = [Ca], srhs = [BK_C1], kcst = c_01)
    BKCAC1 = smodel.SReac('BKCAC1', ssys, slhs = [BK_C1], ilhs = [Ca], srhs = [BK_C2], kcst = c_12)
    BKCAC2 = smodel.SReac('BKCAC2', ssys, slhs = [BK_C2], ilhs = [Ca], srhs = [BK_C3], kcst = c_23)
    BKCAC3 = smodel.SReac('BKCAC3', ssys, slhs = [BK_C3], ilhs = [Ca], srhs = [BK_C4], kcst = c_34)

    BKC0 = smodel.SReac('BKC0', ssys, slhs = [BK_C1], srhs = [BK_C0], irhs = [Ca], kcst = c_10)
    BKC1 = smodel.SReac('BKC1', ssys, slhs = [BK_C2], srhs = [BK_C1], irhs = [Ca], kcst = c_21)
    BKC2 = smodel.SReac('BKC2', ssys, slhs = [BK_C3], srhs = [BK_C2], irhs = [Ca], kcst = c_32)
    BKC3 = smodel.SReac('BKC3', ssys, slhs = [BK_C4], srhs = [BK_C3], irhs = [Ca], kcst = c_43)

    BKCAO0 = smodel.SReac('BKCAO0', ssys, slhs = [BK_O0], ilhs = [Ca], srhs = [BK_O1], kcst = o_01)
    BKCAO1 = smodel.SReac('BKCAO1', ssys, slhs = [BK_O1], ilhs = [Ca], srhs = [BK_O2], kcst = o_12)
    BKCAO2 = smodel.SReac('BKCAO2', ssys, slhs = [BK_O2], ilhs = [Ca], srhs = [BK_O3], kcst = o_23)
    BKCAO3 = smodel.SReac('BKCAO3', ssys, slhs = [BK_O3], ilhs = [Ca], srhs = [BK_O4], kcst = o_34)

    BKO0 = smodel.SReac('BKO0', ssys, slhs = [BK_O1], srhs = [BK_O0], irhs = [Ca], kcst = o_10)
    BKO1 = smodel.SReac('BKO1', ssys, slhs = [BK_O2], srhs = [BK_O1], irhs = [Ca], kcst = o_21)
    BKO2 = smodel.SReac('BKO2', ssys, slhs = [BK_O3], srhs = [BK_O2], irhs = [Ca], kcst = o_32)
    BKO3 = smodel.SReac('BKO3', ssys, slhs = [BK_O4], srhs = [BK_O3], irhs = [Ca], kcst = o_43)

    BKC0O0 = smodel.VDepSReac('BKC0O0', ssys, slhs = [BK_C0], srhs = [BK_O0], k=lambda V: f_0(V))
    BKC1O1 = smodel.VDepSReac('BKC1O1', ssys, slhs = [BK_C1], srhs = [BK_O1], k=lambda V: f_1(V))
    BKC2O2 = smodel.VDepSReac('BKC2O2', ssys, slhs = [BK_C2], srhs = [BK_O2], k=lambda V: f_2(V))
    BKC3O3 = smodel.VDepSReac('BKC3O3', ssys, slhs = [BK_C3], srhs = [BK_O3], k=lambda V: f_3(V))
    BKC4O4 = smodel.VDepSReac('BKC4O4', ssys, slhs = [BK_C4], srhs = [BK_O4], k=lambda V: f_4(V))

    BKO0C0 = smodel.VDepSReac('BKO0C0', ssys, slhs = [BK_O0], srhs = [BK_C0], k=lambda V: b_0(V))
    BKO1C1 = smodel.VDepSReac('BKO1C1', ssys, slhs = [BK_O1], srhs = [BK_C1], k=lambda V: b_1(V))
    BKO2C2 = smodel.VDepSReac('BKO2C2', ssys, slhs = [BK_O2], srhs = [BK_C2], k=lambda V: b_2(V))
    BKO3C3 = smodel.VDepSReac('BKO3C3', ssys, slhs = [BK_O3], srhs = [BK_C3], k=lambda V: b_3(V))
    BKO4C4 = smodel.VDepSReac('BKO4C4', ssys, slhs = [BK_O4], srhs = [BK_C4], k=lambda V: b_4(V))

    OC_BK0 = smodel.OhmicCurr('OC_BK0', ssys, chanstate = BK_O0, erev = BK_rev, g = BK_G )
    OC_BK1 = smodel.OhmicCurr('OC_BK1', ssys, chanstate = BK_O1, erev = BK_rev, g = BK_G )
    OC_BK2 = smodel.OhmicCurr('OC_BK2', ssys, chanstate = BK_O2, erev = BK_rev, g = BK_G )
    OC_BK3 = smodel.OhmicCurr('OC_BK3', ssys, chanstate = BK_O3, erev = BK_rev, g = BK_G )
    OC_BK4 = smodel.OhmicCurr('OC_BK4', ssys, chanstate = BK_O4, erev = BK_rev, g = BK_G )


    ###### SK channel ##################


    SKchan = smodel.Chan('SKchan', mdl)

    SK_C1 = smodel.ChanState('SK_C1', mdl, SKchan)
    SK_C2 = smodel.ChanState('SK_C2', mdl, SKchan)
    SK_C3 = smodel.ChanState('SK_C3', mdl, SKchan)
    SK_C4 = smodel.ChanState('SK_C4', mdl, SKchan)
    SK_O1 = smodel.ChanState('SK_O1', mdl, SKchan)
    SK_O2 = smodel.ChanState('SK_O2', mdl, SKchan)


    SKCAC1 = smodel.SReac('SKCAC1', ssys, slhs = [SK_C1], ilhs = [Ca], srhs = [SK_C2], kcst = dirc2_t)
    SKCAC2 = smodel.SReac('SKCAC2', ssys, slhs = [SK_C2], ilhs = [Ca], srhs = [SK_C3], kcst = dirc3_t)
    SKCAC3 = smodel.SReac('SKCAC3', ssys, slhs = [SK_C3], ilhs = [Ca], srhs = [SK_C4], kcst = dirc4_t)

    SKC1 = smodel.SReac('SKC1', ssys, slhs = [SK_C2], srhs = [SK_C1], irhs = [Ca], kcst = invc1_t)
    SKC2 = smodel.SReac('SKC2', ssys, slhs = [SK_C3], srhs = [SK_C2], irhs = [Ca], kcst = invc2_t)
    SKC3 = smodel.SReac('SKC3', ssys, slhs = [SK_C4], srhs = [SK_C3], irhs = [Ca], kcst = invc3_t)

    SKC3O1 = smodel.SReac('SKC3O1', ssys, slhs = [SK_C3], srhs = [SK_O1], kcst = diro1_t)
    SKC4O2 = smodel.SReac('SKC4O2', ssys, slhs = [SK_C4], srhs = [SK_O2], kcst = diro2_t)

    SKO1C3 = smodel.SReac('SKO1C3', ssys, slhs = [SK_O1], srhs = [SK_C3], kcst = invo1_t)
    SKO2C4 = smodel.SReac('SKO2C4', ssys, slhs = [SK_O2], srhs = [SK_C4], kcst = invo2_t)

    OC1_SK = smodel.OhmicCurr('OC1_SK', ssys, chanstate = SK_O1, erev = SK_rev, g = SK_G )
    OC2_SK = smodel.OhmicCurr('OC2_SK', ssys, chanstate = SK_O2, erev = SK_rev, g = SK_G )

    ###### AMPA channel ###########

    AMPA = smodel.Chan('AMPA', mdl)
    AMPA_C = smodel.ChanState('AMPA_C', mdl, AMPA)
    AMPA_C1 = smodel.ChanState('AMPA_C1', mdl, AMPA)
    AMPA_C2 = smodel.ChanState('AMPA_C2', mdl, AMPA)
    AMPA_D1 = smodel.ChanState('AMPA_D1', mdl, AMPA)
    AMPA_D2 = smodel.ChanState('AMPA_D2', mdl, AMPA)
    AMPA_O = smodel.ChanState('AMPA_O', mdl, AMPA)

    AMPACC1 = smodel.SReac('AMPACC1', ssys, slhs = [AMPA_C], srhs = [AMPA_C1], kcst = 0.0)
    AMPAC1C2 = smodel.SReac('AMPAC1C2', ssys, slhs = [AMPA_C1], srhs = [AMPA_C2], kcst = 0.0)
    AMPAC2O = smodel.SReac('AMPAC2O', ssys, slhs = [AMPA_C2], srhs = [AMPA_O], kcst = ro)

    AMPAC1C = smodel.SReac('AMPAC1C', ssys, slhs = [AMPA_C1], srhs = [AMPA_C], kcst = ru1)
    AMPAC2C1 = smodel.SReac('AMPAC2C1', ssys, slhs = [AMPA_C2], srhs = [AMPA_C1], kcst = ru2)
    AMPAOC2 = smodel.SReac('AMPAOC2', ssys, slhs = [AMPA_O], srhs = [AMPA_C2], kcst = rc)

    AMPAD1 = smodel.SReac('AMPAD1', ssys, slhs = [AMPA_C1], srhs = [AMPA_D1], kcst = rd)
    AMPAD2 = smodel.SReac('AMPAD2', ssys, slhs = [AMPA_C2], srhs = [AMPA_D2], kcst = rd)

    AMPARD1 = smodel.SReac('AMPARD1', ssys, slhs = [AMPA_D1], srhs = [AMPA_C1], kcst = rr)
    AMPARD2 = smodel.SReac('AMPARD2', ssys, slhs = [AMPA_D2], srhs = [AMPA_C2], kcst = rr)

    OC_AMPAR1 = smodel.OhmicCurr('OC_AMPAR1', ssys, chanstate = AMPA_O, erev = AMPA_rev, g = AMPA_G )

    ###### Leak current channel #####

    L = smodel.Chan('L', mdl)
    Leak = smodel.ChanState('Leak', mdl, L)

    OC_L = smodel.OhmicCurr('OC_L', ssys, chanstate = Leak, erev = L_rev, g = L_G)
    return mdl

##################################

########### MESH & COMPARTMENTALIZATION #################

##########Import Mesh
def gen_geom():
    mesh = meshio.loadMesh("meshes/" + mesh_file)[0]
    surftris = mesh.getSurfTris()

    ########## Create an intracellular compartment i.e. cytosolic compartment


    cyto = sgeom.TmComp('cyto', mesh, smooth_tets+spiny_tets)
    cyto.addVolsys('vsys')

    smooth = sgeom.TmPatch('smooth', mesh, smooth_tris, cyto)
    smooth.addSurfsys('ssys')

    spiney = sgeom.TmPatch('spiney', mesh, spiny_tris, cyto)
    spiney.addSurfsys('ssys')

    membrane = sgeom.Memb('membrane', mesh, [smooth, spiney])

    branch_tets = gd.getTetPartitionTable(tet_branches)
    branch_list = []
    for branch in branch_tets:
        if "soma" in branch:
            continue
        mesh.addROI(branch, steps.geom.ELEM_TET, branch_tets[branch])
        branch_list.append(branch)
    return mesh

# # # # # # # # # # # # # # # # # # # # # # # # SIMULATION  # # # # # # # # # # # # # # # # # # # # # #

def benchmarkSim(sim):
    t0 = time.time()
    sim.setTemp(TEMPERATURE+273.15)
    smooth_area = sim.getPatchArea('smooth')
    spiney_area = sim.getPatchArea('spiney')

    #Total pump is 1e-15 mol/cm2 ---> 1e-11 mol/m2
    #pumpnbs per unit area (im m2) is Total pump times AVOGADRO's NUMBER (1e-11 mol/m2 * 6.022e23 /mol )
    pumpnbs = 6.022141e12*smooth_area

    sim.setPatchCount('smooth', 'Pump', round(pumpnbs))
    sim.setPatchCount('smooth', 'CaPump', 0)

    sim.setPatchCount('smooth', 'CaP_m0' , round(CaP_ro*smooth_area*CaP_m0_p))
    sim.setPatchCount('smooth', 'CaP_m1' , round(CaP_ro*smooth_area*CaP_m1_p))
    sim.setPatchCount('smooth', 'CaP_m2' , round(CaP_ro*smooth_area*CaP_m2_p))
    sim.setPatchCount('smooth', 'CaP_m3' , round(CaP_ro*smooth_area*CaP_m3_p))


    sim.setPatchCount('smooth', 'BK_C0' , round(BK_ro*smooth_area*BK_C0_p))
    sim.setPatchCount('smooth', 'BK_C1' , round(BK_ro*smooth_area*BK_C1_p))
    sim.setPatchCount('smooth', 'BK_C2' , round(BK_ro*smooth_area*BK_C2_p))
    sim.setPatchCount('smooth', 'BK_C3' , round(BK_ro*smooth_area*BK_C3_p))
    sim.setPatchCount('smooth', 'BK_C4' , round(BK_ro*smooth_area*BK_C4_p))

    sim.setPatchCount('smooth', 'BK_O0' , round(BK_ro*smooth_area*BK_O0_p))
    sim.setPatchCount('smooth', 'BK_O1' , round(BK_ro*smooth_area*BK_O1_p))
    sim.setPatchCount('smooth', 'BK_O2' , round(BK_ro*smooth_area*BK_O2_p))
    sim.setPatchCount('smooth', 'BK_O3' , round(BK_ro*smooth_area*BK_O3_p))
    sim.setPatchCount('smooth', 'BK_O4' , round(BK_ro*smooth_area*BK_O4_p))


    sim.setPatchCount('smooth', 'SK_C1' , round(SK_ro*smooth_area*SK_C1_p))
    sim.setPatchCount('smooth', 'SK_C2' , round(SK_ro*smooth_area*SK_C2_p))
    sim.setPatchCount('smooth', 'SK_C3' , round(SK_ro*smooth_area*SK_C3_p))
    sim.setPatchCount('smooth', 'SK_C4' , round(SK_ro*smooth_area*SK_C4_p))

    sim.setPatchCount('smooth', 'SK_O1' , round(SK_ro*smooth_area*SK_O1_p))
    sim.setPatchCount('smooth', 'SK_O2' , round(SK_ro*smooth_area*SK_O2_p))

    sim.setPatchCount('smooth', 'AMPA_C' , round(AMPA_receptors))
    sim.setPatchCount('smooth', 'AMPA_C1' , 0)
    sim.setPatchCount('smooth', 'AMPA_C2' , 0)
    sim.setPatchCount('smooth', 'AMPA_O' , 0)
    sim.setPatchCount('smooth', 'AMPA_D1' , 0)
    sim.setPatchCount('smooth', 'AMPA_D2' , 0)

    sim.setPatchCount('smooth', 'Leak', int(L_ro_proximal * smooth_area))


    #Total pump is 1e-15 mol/cm2 ---> 1e-11 mol/m2
    #pumpnbs per unit area (im m2) is Total pump times AVOGADRO's NUMBER (1e-11 mol/m2 * 6.022e23 /mol )
    pumpnbs = 6.022141e12*spiney_area

    sim.setPatchCount('spiney', 'Pump', round(pumpnbs))
    sim.setPatchCount('spiney', 'CaPump', 0)

    sim.setPatchCount('spiney', 'CaP_m0' , round(CaP_ro*spiney_area*CaP_m0_p))
    sim.setPatchCount('spiney', 'CaP_m1' , round(CaP_ro*spiney_area*CaP_m1_p))
    sim.setPatchCount('spiney', 'CaP_m2' , round(CaP_ro*spiney_area*CaP_m2_p))
    sim.setPatchCount('spiney', 'CaP_m3' , round(CaP_ro*spiney_area*CaP_m3_p))

    sim.setPatchCount('spiney', 'BK_C0' , round(BK_ro*spiney_area*BK_C0_p))
    sim.setPatchCount('spiney', 'BK_C1' , round(BK_ro*spiney_area*BK_C1_p))
    sim.setPatchCount('spiney', 'BK_C2' , round(BK_ro*spiney_area*BK_C2_p))
    sim.setPatchCount('spiney', 'BK_C3' , round(BK_ro*spiney_area*BK_C3_p))
    sim.setPatchCount('spiney', 'BK_C4' , round(BK_ro*spiney_area*BK_C4_p))

    sim.setPatchCount('spiney', 'BK_O0' , round(BK_ro*spiney_area*BK_O0_p))
    sim.setPatchCount('spiney', 'BK_O1' , round(BK_ro*spiney_area*BK_O1_p))
    sim.setPatchCount('spiney', 'BK_O2' , round(BK_ro*spiney_area*BK_O2_p))
    sim.setPatchCount('spiney', 'BK_O3' , round(BK_ro*spiney_area*BK_O3_p))
    sim.setPatchCount('spiney', 'BK_O4' , round(BK_ro*spiney_area*BK_O4_p))

    sim.setPatchCount('spiney', 'SK_C1' , round(SK_ro*spiney_area*SK_C1_p))
    sim.setPatchCount('spiney', 'SK_C2' , round(SK_ro*spiney_area*SK_C2_p))
    sim.setPatchCount('spiney', 'SK_C3' , round(SK_ro*spiney_area*SK_C3_p))
    sim.setPatchCount('spiney', 'SK_C4' , round(SK_ro*spiney_area*SK_C4_p))

    sim.setPatchCount('spiney', 'SK_O1' , round(SK_ro*spiney_area*SK_O1_p))
    sim.setPatchCount('spiney', 'SK_O2' , round(SK_ro*spiney_area*SK_O2_p))

    sim.setPatchCount('spiney', 'AMPA_C' , 0)
    sim.setPatchCount('spiney', 'AMPA_C1' , 0)
    sim.setPatchCount('spiney', 'AMPA_C2' , 0)
    sim.setPatchCount('spiney', 'AMPA_O' , 0)
    sim.setPatchCount('spiney', 'AMPA_D1' , 0)
    sim.setPatchCount('spiney', 'AMPA_D2' , 0)

    sim.setPatchCount('spiney', 'Leak', int(L_ro_spiny * spiney_area))

    sim.setCompConc('cyto', 'Ca', Ca_iconc)
    sim.setCompConc('cyto', 'Mg', Mg_conc)

    sim.setCompConc('cyto', 'iCBsf', iCBsf_conc)
    sim.setCompConc('cyto', 'iCBCaf', iCBCaf_conc)
    sim.setCompConc('cyto', 'iCBsCa', iCBsCa_conc)
    sim.setCompConc('cyto', 'iCBCaCa', iCBCaCa_conc)

    sim.setCompConc('cyto', 'CBsf', CBsf_conc)
    sim.setCompConc('cyto', 'CBCaf', CBCaf_conc)
    sim.setCompConc('cyto', 'CBsCa', CBsCa_conc)
    sim.setCompConc('cyto', 'CBCaCa', CBCaCa_conc)

    sim.setCompConc('cyto', 'PV', PV_conc)
    sim.setCompConc('cyto', 'PVCa', PVCa_conc)
    sim.setCompConc('cyto', 'PVMg', PVMg_conc)


    sim.setEfieldDT(EF_DT)

    sim.setMembPotential('membrane', init_pot)

    sim.setMembVolRes('membrane', Ra)

    sim.setMembCapac('membrane',memb_capac_proximal)

    for tri in spiny_tris:
        sim.setTriCapac(tri, memb_capac_spiny)

    init_time = time.time() - t0
    t0 = time.time()
    for l in range(NTIMEPOINTS):
        sim.setPatchSReacK('smooth', 'AMPACC1', 1.0e-3 *rb*Glut[l+2000])
        sim.setPatchSReacK('smooth', 'AMPAC1C2', 1.0e-3 *rb*Glut[l+2000])
        sim.run(TIMECONVERTER*l)
    sim_run_time = time.time() - t0
    return init_time, sim_run_time

rng=srng.create('mt19937',512)
rng.initialize(1)

t0 = time.time()
m = gen_model() 
if steps.mpi.rank == 0: benchmark_file.write("Model setup time (sec),%e\n" % (time.time() - t0))

t0 = time.time()
mesh = gen_geom()
if steps.mpi.rank == 0: 
    benchmark_file.write("Geometry setup time (sec),%e\n" % (time.time() - t0)) 
    benchmark_file.write("Parallel TetOPSplit\n")

t0 = time.time()
PARTITION_FILE = "meshes/" + mesh_file + ".metis.epart." + str(steps.mpi.nhosts)
mpi_tet_partitions = metis_support.readPartition(PARTITION_FILE)
mpi_tri_partitions = gd.partitionTris(mesh, mpi_tet_partitions, mesh.getSurfTris())
r = srng.create("r123", 512)
r.initialize(1)
sim = ssolver.TetOpSplit(m, mesh, rng, ssolver.EF_DV_PETSC, mpi_tet_partitions, mpi_tri_partitions)

if steps.mpi.rank == 0: benchmark_file.write("Solver construction time (sec),%e\n" % (time.time() - t0))

init_time, sim_run_time = benchmarkSim(sim)

if steps.mpi.rank == 0: 
    benchmark_file.write("Solver initial time (sec),%e\n" % (init_time))
    benchmark_file.write("Solver run time (sec),%e\n\n" % (sim_run_time))
        
    benchmark_file.close()
