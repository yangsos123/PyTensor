import sys
import numpy as np
from scipy.misc import comb

sys.path.append("core")
import MPS
import MPO
import Models as md
import Contractor as ct

L = 10
D = 20
Jx = 1.5*np.ones(L)
Jy = 1.*np.ones(L)
Jz = 1.*np.ones(L)
g = 0.5*np.ones(L)
h = 1.5*np.ones(L)
offset = 0
HeiModel = md.Heisenberg(L, Jx, Jy, Jz, g, h, offset)
H = HeiModel.hamil

# Test of DMRG
gs = MPS.MPS(L, D, 2)
#gs.setProductState(md.Up)
gs.setRandomState()
Emin = ct.findGroundState(H, gs, D)

# Test of apply MPO to MPS
applyH = ct.fitApplyMPO(H, gs, 20, tol=1e-4)
print(np.real(ct.contractMPS(gs, applyH)))
