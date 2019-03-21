"""
Define each site for MPS
"""

import numpy as np
import numpy.random
import scipy.linalg as LA


class site:
	def __init__(self, s, Dl, Dr):
		self.s = s			# physical dimension
		self.Dl = Dl		# left bond dimension
		self.Dr = Dr		# right bond dimension
		self.A = np.zeros((s, Dl, Dr), dtype=complex)

	def gaugeR(self, L, cutD):
		"""
		Left multiply a matrix L to A, then do SVD (if need to reduce bond
		dimension) / QR decomposition. Set value of A to Herimitian matrix
		and return the rest (right) part.
		"""
		self.A = np.einsum('ij,kjl->kil', L, self.A)
		self.Dl = self.A.shape[1]
		R = 0

		if (self.Dr==1):
			R = LA.norm(self.A)
			self.A = self.A / R
			R = np.array([[R]])
		else:
			aux = self.A.reshape((self.s*self.Dl, self.Dr))
			if (cutD ==0):
				Q, R = LA.qr(aux)
				self.A = Q.reshape((self.s, self.Dl, -1))
			else:
				U, S, Vdag = LA.svd(aux, full_matrices=False)
				if (S.shape[0] > cutD):
					U = U[:,0:cutD]
					S = S[0:cutD]
					Vdag = Vdag[0:cutD,:]
				R = np.einsum('i,ij->ij', S, Vdag)
				self.A = U.reshape((self.s, self.Dl, -1))
		self.Dr = self.A.shape[2]
		return R

	def gaugeL(self, R, cutD):
		self.A = np.einsum('ijk,kl->ijl', self.A, R)
		self.Dr = self.A.shape[2]
		L = 0

		if (self.Dl==1):
			L = LA.norm(self.A)
			self.A = self.A / L
			L = np.array([[L]])
		else:
			aux = np.swapaxes(self.A, 0, 1)
			aux = aux.reshape((self.Dl, self.s*self.Dr))
			if (cutD ==0):
				L, Q = LA.rq(aux)
				self.A = np.swapaxes(Q.reshape((-1, self.s, self.Dr)), 0, 1)
			else:
				U, S, Vdag = LA.svd(aux, full_matrices=False)
				if (S.shape[0] > cutD):
					U = U[:,0:cutD]
					S = S[0:cutD]
					Vdag = Vdag[0:cutD,:]

				L = np.einsum('ij,j->ij', U, S)
				self.A = np.swapaxes(Vdag.reshape((-1, self.s, self.Dr)), 0, 1)
		self.Dl = self.A.shape[1]
		return L
