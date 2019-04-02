import sys
import random
import numpy as np
import numpy.random
import copy

sys.path.append("..")
from core import Contractor as ct
from core import MPO
from core import MPS
from models import Boson as Bs
from models import Spin as Sp


L = 10
D = 20
Jx = 1.
Jy = 1.
Jz = 1.
g = -1.05
h = 0.5
offset = 0
Nmax = 4
IsingModel = Sp.Ising(L, Jz, g, h, offset)
HeisenbergModel = Sp.Heisenberg(L, Jx, Jy, Jz, g, h, offset)
BoseHubbardModel = Bs.BoseHubbard(
    L, Nmax, t=1., U=0.1, mu=1., V=0.5, Vint=0.2, offset=0)


# Test of DMRG & fitApplyMPO & entanglement entropy & total spin projector
#H = IsingModel.hamil
H = HeisenbergModel.hamil
gs = MPS.MPS(L, D, 2)
# gs.setProductState(Sp.Up)

#H = BoseHubbardModel.hamil
#gs = MPS.MPS(L, D, Nmax+1)
gs.setRandomState()
Emin = ct.dmrg(H, gs, D)
gs.saveMPS("saveGsMPS")
H.saveMPO("saveHMPO")
applyH = ct.fitApplyMPO(H, gs, D, tol=1e-4, silent=False, maxRound=20)
gsFile = MPS.loadMPS("saveGsMPS")
HFile = MPO.loadMPO("saveHMPO")
print("Emin =", Emin)
print("<gsF|HF|gsF> =", np.real(ct.contractMPSMPO(gsFile, HFile, gsFile)))
print("<gs|H gs> =", np.real(ct.contractMPS(gs, applyH)))
print("Ground state entanglement entropy is",
      np.exp(gs.getEntanglementEntropy(L // 2)))
print()

totSzMPO = Sp.getSumSzMPO(L)
print("Total spin of ground state if",
      np.real(ct.contractMPSMPO(gs, totSzMPO, gs)))
print()
totS = 2
P = Sp.getSumSzProjector(L, totS)
gsp = ct.exactApplyMPO(P, gs)
print("Total spin after being projected to S =", totS, "is",
      np.real(ct.contractMPSMPO(gsp, totSzMPO, gsp) / ct.contractMPS(gsp, gsp)))


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
Sz.setA(L//2, (Sp.PauliSigma[3,:,:]).reshape((2,2,1,1)))
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
