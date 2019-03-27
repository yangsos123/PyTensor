import sys
import random
import numpy as np
import numpy.random
import copy

sys.path.append("core")
sys.path.append("models")
import MPS
import MPO
import SpinModels as SpMd
import Contractor as ct

L = 20
D = 20
Jx = 1.0*np.ones(L)
Jy = 1.*np.ones(L)
Jz = 1.*np.ones(L)
g = -1.05*np.ones(L)
h = 0.5*np.ones(L)
offset = 0
IsingModel = SpMd.Ising(L, Jz, g, h, offset)
H = IsingModel.hamil


# Test of DMRG & fitApplyMPO & entanglement entropy
gs = MPS.MPS(L, D, 2)
#gs.setProductState(SpMd.Up)
gs.setRandomState()
Emin = ct.dmrg(H, gs, D)
applyH = ct.fitApplyMPO(H, gs, D, tol=1e-6)
print("<gs|H|gs>=", Emin)
print("<gs|Hgs>=", np.real(ct.contractMPS(gs, applyH)))
print("Ground state entropy", np.exp(gs.getEntanglementEntropy(L//2)))
print()



"""
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
"""

"""
# Test of time evolution
UMPO = IsingModel.getUMPOIsing(-0.01, imag=False)
Sz = MPO.MPO(L, 1, 2)
Sz.setProductOperator(np.identity(2))
Sz.setA(L//2, (SpMd.PauliSigma[3,:,:]).reshape((2,2,1,1)))
SzMPS = MPO.MPSfromMPO(Sz)
idMPO = MPO.MPO(L, 1, 2)
idMPO.setProductOperator(np.identity(2))
idMPS = MPO.MPSfromMPO(idMPO)
doubleU = MPO.extendMPO(UMPO)
HMPS = MPO.MPSfromMPO(H)
UtMPS = copy.deepcopy(idMPS)

for i in range(100):
	Z = ct.contractMPS(UtMPS, idMPS)
	print(i*0.01, np.real(ct.contractMPS(UtMPS, HMPS) / Z),
				  np.real(ct.contractMPS(UtMPS, SzMPS) / Z))
	UtMPS = ct.fitApplyMPO(doubleU, UtMPS, D)
	#UtMPS.gaugeCond(2)
"""
