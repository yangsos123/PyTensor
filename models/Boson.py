"""
Generate Hamiltonians / time evolution MPOs of
	Bose-Hubbard model
Haven't been tested yet
"""

import sys
sys.path.append("../core")
import copy
import numpy as np
import numpy.random
import scipy.linalg as LA
import MPS
import MPO
import Contractor as ct


class BoseHubbard:
	# H = - \sum_{i=1}^{L-1} t_i (b_i^dag b_{i+1} + h.c.)
	#     + \sum_{i=1}^{L} U_i n_i (n_i - 1) / 2
	#	  + \sum_{i=1}^{L} (- \mu_i n_i + V_i n_i)
	#     + \sum_{i=1}^{L-1} V_{int} n_i n_{i+1}
	# Maximum occupation number N
	def __init__(self, L, Nmax, t, U, mu, V, Vint, offset):
		self.L = L
		self.d = Nmax + 1
		self.t = t
		self.U = U
		self.mu = mu
		self.V = V
		self.Vint = Vint
		self.hamil = MPO.MPO(L, 5, self.d)

		self.Z = np.zeros((5, self.d, self.d), dtype = complex)
		self.Z[0,:,:] = np.identity(self.d, dtype = complex)				# Id
		self.Z[1,:,:] = np.diag(np.arange(self.d)*(np.arange(self.d)-1)+0j)	# n(n-1)
		self.Z[2,:,:] = np.diag(np.sqrt(np.arange(self.d-1)+1)+0j, -1)		# b^dag
		self.Z[3,:,:] = np.diag(np.sqrt(np.arange(self.d-1)+1)+0j, 1)		# b
		self.Z[4,:,:] = np.diag(np.arange(self.d)+0j)						# n

		opL = np.zeros((1, 5, 5), dtype=complex)
		opL[0,0,0] = 1
		opL[0,1,2] = 1
		opL[0,2,3] = 1
		opL[0,3,4] = 1
		opL[0,4,4] = self.V[0]-self.mu[0]
		opL[0,4,1] = self.U[0]/2
		self.hamil.setA(0, np.einsum('ijk,kmn->mnij', opL, Z))
		opR = np.zeros((5, 1, 5), dtype=complex)
		opR[0,0,4] = self.V[0]-self.mu[0]
		opR[0,0,1] = self.U[0]/2
		opR[1,0,3] = -self.t[i]
		opR[2,0,2] = -self.t[i]
		opR[3,0,4] = Vint
		opR[4,0,0] = 1
		self.hamil.setA(L-1, np.einsum('ijk,kmn->mnij', opR, Z))
		for i in range(1,L-1):
			opM = np.zeros((5, 5, 5), dtype=complex)
			opM[0,0,0] = 1
			opM[0,1,2] = 1
			opM[0,2,3] = 1
			opM[0,3,4] = 1
			opM[0,4,4] = self.V[0]-self.mu[0]
			opM[0,4,1] = self.U[0]/2
			opM[1,4,3] = -self.t[i]
			opM[2,4,2] = -self.t[i]
			opM[3,4,4] = Vint
			opM[4,4,0] = 1
			self.hamil.setA(i, np.einsum('ijk,kmn->mnij', opM, Z))


def getSumNMPO(L, Nmax)
	d = Nmax + 1
	Z = np.zeros((2, d, d), dtype = complex)
	Z[0,:,:] = np.identity(d, dtype = complex)			# Id
	Z[1,:,:] = np.diag(np.arange(d)+0j)					# n
	sumN = MPO.MPO(L, 2, d)
	opL = np.zeros((1, 2, 2), dtype=complex)
	opL[0,0,0] = 1
	opL[0,1,1] = 1
	sumN.setA(0, np.einsum('ijk,kmn->mnij', opL, Z))
	opR = np.zeros((2, 1, 2), dtype=complex)
	opR[0,0,1] = 1
	opR[1,0,0] = 1
	sumN.setA(L-1, np.einsum('ijk,kmn->mnij', opR, Z))
	for i in range(1,L-1):
		opM = np.zeros((2, 2, 2), dtype=complex)
		opM[0,0,0] = 1
		opM[0,1,1] = 1
		opM[1,1,0] = 1
		sumN.setA(i, np.einsum('ijk,kmn->mnij', opM, Z))
	return sumN
