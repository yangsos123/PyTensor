import sys
import random
import numpy as np
import numpy.random
import copy

sys.path.append("../core")
sys.path.append("../models")
import MPS
import MPO
import Spin as Sp
import Contractor as ct


def ChebyTn(H, M, D, mode, input, saveT = False):
	# input is an MPS if mode=="LDoS" and a list of MPOs if mode=="DoS"
	L = H.L
	s = H.s
	mu = np.zeros((M))
	if mode == "DoS":
		Tm = MPO.MPO(L, 1, s)
		Tm.setProductOperator(np.identity(s[0]))
		Tm = MPO.MPSfromMPO(Tm)
		Hp = MPO.extendMPO(H)
		
	elif mode == "LDoS":
		Tm = input
		Hp = copy.deepcopy(H)
	else:
		print("Mode not supported!")
		exit(3)

	T = ct.fitApplyMPO(Hp, Tm, D)




	return 0
