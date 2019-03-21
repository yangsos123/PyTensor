"""
Define MPO and some common Hamiltonians
Transform between MPS and MPO
"""

import sys
import numpy as np
import numpy.random
import scipy.linalg as LA
import SiteOp
import MPS


class MPO:
	def __init__(self, L, D, s):
		self.L = L
		self.D = D
		self.ops = []
		for i in range(L):
			self.ops.append(SiteOp.siteOp(s, 1 if i==0 else D,
											 1 if i==L-1 else D))

	def setA(self, k, Ak):
		if (Ak.shape[0]!=Ak.shape[1]):
			print("Error: inconsistent physical dimension!")
			exit(2)
		elif (k<0 or k>=self.L):
			print("Error: k",k,"out of range!")
			exit(2)
		else:
			self.ops[k].A = Ak
			self.ops[k].s = Ak.shape[0]
			self.ops[k].Dl = Ak.shape[2]
			self.ops[k].Dr = Ak.shape[3]

	def setProductOperator(self, localOp):
		for i in range(self.L):
			s = localOp.shape[0]
			self.ops[i].A = localOp.reshape((s,s,1,1))
			self.ops[i].Dl = 1
			self.ops[i].Dr = 1
			self.ops[i].s = s
		self.D = 1


def MPSfromMPO(mpo):
	"""
	Input:             Return:
	mpo:
	i1 i2      iL      i1j1 i2j2      iLjL
	├  ┼ ... ┼ ┤        └    ┴  ... ┴  ┘
	j1 j2      jL
	"""
	L = mpo.L
	mps = MPS.MPS(L, 1, 1)
	for i in range(L):
		opShape = mpo.ops[i].A.shape
		mps.setA(i, mpo.ops[i].A.reshape((opShape[0]*opShape[1],
										  opShape[2], opShape[3])))
	return mps


def MPOfromMPS(mps):
	# Inverse transformation of MPSfromMPO
	L = mps.L
	mpo = MPO(L, 1, 1)
	for i in range(L):
		opShape = mps.sites[i].A.shape
		mpo.setA(i, mps.sites[i].A.reshape((int(np.sqrt(opShape[0])),
				int(np.sqrt(opShape[0])), opShape[1], opShape[2])))
	return mpo


def extendMPO(simpleMPO):
	# Make an MPO compatible with MPS got from MPSfromMPO
	L = simpleMPO.L
	doubleMPO = MPO(L, 1, 1)
	for i in range(L):
		s = simpleMPO.ops[i].s
		idPhy = np.identity(s).reshape((s,s,1,1))
		doubleMPO.setA(i, np.kron(simpleMPO.ops[i].A, idPhy))
	return doubleMPO
