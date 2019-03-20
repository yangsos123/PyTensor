"""
Define MPS, do gauge transformations, change dimension and initialize with
random state or product state
"""

import sys
import numpy as np
import numpy.random
import scipy.linalg as LA
import Site


class MPS:
	def __init__(self, L, D, s):
		self.L = L			# length
		self.sites = []
		self.D = D
		for i in range(L):
			self.sites.append(Site.site(s,
				1 if i==0 else D, 1 if i==L-1 else D))


	def setA(self, k, Ak):
		if (k<0 or k>=self.L):
			print("Error: k",k,"out of range!")
			exit(2)
		self.sites[k].A = Ak
		self.sites[k].s = Ak.shape[0]
		self.sites[k].Dl = Ak.shape[1]
		self.sites[k].Dr = Ak.shape[2]


	def gaugeCond(self, dir, normal = 1, cutD = 0, silent = False):
		# dir = 1 -> left
		# dir = 2 -> right
		if dir == 1:
			if (silent == False):
				print("Set gauge condition L")
		elif dir == 2:
			if (silent == False):
				print("Set gauge condition R")
		else:
			print("Wrong gauge condition parameter")
			exit(1)
		tmp = np.array([[1]])
		for i in range(self.L):
			if (dir == 1):
				tmp = self.sites[self.L-i-1].gaugeL(tmp, cutD)
			elif (dir == 2):
				tmp = self.sites[i].gaugeR(tmp, cutD)
		if (normal == 0):
			self.sites[0].A *= np.complex(tmp)
		if (cutD > 0 and cutD < self.D):
			self.D = cutD


	def gaugeCondMixed(self, left, k, right, cutD = 0, silent = True):
		# 0 - k-1 -> left
		# k+1 - L-1 -> right
		if (silent == False):
			print("Set mixed gauge condition centered at", k)
		tmp = np.identity(self.sites[left].Dl, dtype=complex)
		#print("Tmp shape", tmp.shape)
		for i in range(left,k):
			tmp = self.sites[i].gaugeR(tmp, cutD)
			#print("Tmp shape", tmp.shape)

		self.sites[k].A = np.einsum('ij,kjl->kil', tmp, self.sites[k].A)
		self.sites[k].Dl = self.sites[k].A.shape[1]

		tmp = np.identity(self.sites[right].Dr, dtype=complex)
		for i in range(right-k):
			tmp = self.sites[right-i].gaugeL(tmp, cutD)
		self.sites[k].A = np.einsum('ijk,kl->ijl', self.sites[k].A, tmp)
		self.sites[k].Dr = self.sites[k].A.shape[2]
		if (cutD > 0 and cutD < self.D):
			self.D = cutD


	def adjustD(self, DPrime):
		if (DPrime < self.D):
			# Only when self.D is not so large
			# For larger self.D, use compressMPS in Contractor.py
			self.gaugeCond(2, normal=0, cutD=DPrime)
		else:
			self.sites[0].A = np.append(self.sites[0].A,
				np.zeros((self.sites[0].s, 1, DPrime - self.sites[0].Dr)), axis=2)
			self.sites[0].Dr = DPrime
			for i in range(1, self.L-1):
				self.sites[i].A = np.append(self.sites[i].A,
					np.zeros((self.sites[i].s, self.sites[i].Dl, DPrime - self.sites[i].Dr)),
					axis=2)
				self.sites[i].Dr = DPrime
				self.sites[i].A = np.append(self.sites[i].A,
					np.zeros((self.sites[i].s, DPrime - self.sites[i].Dl, self.sites[i].Dr)),
					axis=1)
				self.sites[i].Dl = DPrime
			self.sites[self.L-1].A = np.append(self.sites[self.L-1].A,
				np.zeros((self.sites[self.L-1].s, DPrime - self.sites[self.L-1].Dl, 1)),
				axis=1)
			self.sites[self.L-1].Dl = DPrime
			self.D = DPrime


	def applyLocalOperator(self, k, op):
		if (k<0 or k>=self.L):
			print("Error: k out of range!")
			return False
		opShape = op.shape
		if (opShape[0]==self.sites[k].s and opShape[1]==self.sites[k].s):
			self.sites[k].A = np.einsum('ij,jkl->ikl', op, self.sites[k].A)
			return True
		else:
			print("Error: shape does not fit!")
			return False


	def setRandomState(self):
		print("Set random state")
		for i in range(self.L):
			self.sites[i].A = numpy.random.random((self.sites[i].s,
								self.sites[i].Dl, self.sites[i].Dr)) \
							+ 1j * numpy.random.random((self.sites[i].s,
								self.sites[i].Dl, self.sites[i].Dr))
		self.gaugeCond(2, normal = 1, cutD = self.D, silent=True)


	def setProductState(self, localState):
		sLocal = localState.shape[0]
		for i in range(self.L):
			self.sites[i].s = sLocal
			self.sites[i].A = localState.reshape((sLocal, 1, 1))
			self.sites[i].Dl = 1
			self.sites[i].Dr = 1
		self.D = 1
