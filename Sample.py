import sys
import random
import numpy as np

sys.path.append("core")
sys.path.append("models")
import MPS
import MPO
import SpinModels as SpMd
import Contractor as ct

L = 10
D = 40
Jx = 1.5*np.ones(L)
Jy = 1.*np.ones(L)
Jz = 1.*np.ones(L)
g = 0.5*np.ones(L)
h = 1.5*np.ones(L)
offset = 0
IsingModel = SpMd.Ising(L, Jz, g, h, offset)
H = IsingModel.hamil


# Test of DMRG
gs = MPS.MPS(L, D, 2)
#gs.setProductState(md.Up)
gs.setRandomState()
gs.adjustD(20)
Emin = ct.dmrg(H, gs, D)

# Test of fitApplyMPO
applyH = ct.fitApplyMPO(H, gs, D, tol=1e-6)
print("<gs|H|gs>=", Emin)
print("<gs|Hgs>=", np.real(ct.contractMPS(gs, applyH)))
print()

# Test of sumMPS
randMPS = []
coef = []
overlapGs = 0
for i in range(5):
	tmpMPS = MPS.MPS(L, D, 2)
	tmpMPS.setRandomState()
	randMPS.append(tmpMPS)
	tmpCoef = random.random()
	coef.append(tmpCoef)
	overlapGs += ct.contractMPS(tmpMPS, gs) * tmpCoef
sumRandMPS = ct.sumMPS(randMPS, coef, D, silent=True)
print("Sum of overlap:", overlapGs)
print("Overlap of sum:", ct.contractMPS(sumRandMPS, gs))
