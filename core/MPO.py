"""
Define MPO
Transform between MPS and MPO
Get MPO of real / imaginary revolution operator
"""

import sys
import numpy as np
import numpy.random
import scipy.linalg as LA
import SiteOp
import MPS
import Contractor as ct


class MPO:
	def __init__(self, L, D, s):
		self.L = L
		self.D = D
		self.s = s
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
			if (Ak.shape[2]>self.D):
				self.D = Ak.shape[2]
			if (Ak.shape[3]>self.D):
				self.D = Ak.shape[3]

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


def getUMPO(L, s, hi, hrestL, hrestR, t, imag = True):
	# Return e^{-Ht} if imag else e^{-iHt}
	# e^{-iHt} ~ e^{-iH_odd t/2}e^{-iH_even t}e^{-iH_odd t/2}
	# t should be a small value; Error ~ O(t^3)
	# Only for neareast neighbours
	if (imag==True):
		exph = LA.expm(-1j*hi*t)
		exphHalf = LA.expm(-1j*hi*t/2)
		exphrestL = LA.expm(-1j*hrestL*t/2)
		if (L%2==1):
			exphrestR = LA.expm(-1j*hrestR*t)
		else:
			exphrestR = LA.expm(-1j*hrestR*t/2)
	else:
		exph = LA.expm(-hi*t)
		exphHalf = LA.expm(-hi*t/2)
		exphrestL = LA.expm(-hrestL*t/2)
		if (L%2==1):
			exphrestR = LA.expm(-hrestR*t)
		else:
			exphrestR = LA.expm(-hrestR*t/2)
	exphrestL = exphrestL.reshape((s, s, 1, 1))
	exphrestR = exphrestR.reshape((s, s, 1, 1))

	exph = exph.reshape((s, s, s, s))
	exph = np.swapaxes(exph, 1, 2)
	exph = exph.reshape((s*s, s*s))
	svdU, svdS, svdV = LA.svd(exph, full_matrices=False)
	svdL = svdU.reshape((s, s, 1, -1))
	svdR = np.einsum('i,ij->ij', svdS, svdV).reshape((-1, 1, s, s))
	svdR = np.swapaxes(svdR, 0, 2)
	svdR = np.swapaxes(svdR, 1, 3)

	Ueven = MPO(L, 1, s)
	for i in range(L//2):
		Ueven.setA(i*2, svdL)
		Ueven.setA(i*2+1, svdR)
	if (L%2==1):
		Ueven.setA(L-1, exphrestR)

	exphHalf = exphHalf.reshape((s, s, s, s))
	exphHalf = np.swapaxes(exphHalf, 1, 2)
	exphHalf = exphHalf.reshape((s*s, s*s))
	svdUHalf, svdSHalf, svdVHalf = LA.svd(exphHalf, full_matrices=False)
	svdLHalf = svdUHalf.reshape((s, s, 1, -1))
	svdRHalf = np.einsum('i,ij->ij', svdSHalf, svdVHalf).reshape((-1, 1, s, s))
	svdRHalf = np.swapaxes(svdRHalf, 0, 2)
	svdRHalf = np.swapaxes(svdRHalf, 1, 3)

	UoddHalf = MPO(L, 1, s)
	UoddHalf.setA(0, exphrestL)
	for i in range((L-1)//2):
		UoddHalf.setA(i*2+1, svdLHalf)
		UoddHalf.setA(i*2+2, svdRHalf)
	if (L%2==0):
		UoddHalf.setA(L-1, exphrestR)

	UMPO = ct.joinMPO(UoddHalf, Ueven)
	UMPO = ct.joinMPO(UMPO, UoddHalf)
	return UMPO
