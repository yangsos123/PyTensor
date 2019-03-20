import sys

import numpy as np
import numpy.random
import scipy.linalg as LA
import MPS
import MPO


PauliSigma = np.array(
	[[[1,0],[0,1]], [[0,1],[1,0]], [[0,-1j], [1j,0]], [[1,0],[0,-1]]])

Up = np.array([1,0])
Dn = np.array([0,1])
Xp = np.array([1/np.sqrt(2),1/np.sqrt(2)])
Xm = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
Yp = np.array([1/np.sqrt(2),1j/np.sqrt(2)])
Ym = np.array([1/np.sqrt(2),-1j/np.sqrt(2)])
Em = np.array([0,0])

class Ising:
	def __init__(self, L, J, g, h, offset):
		self.L = L
		self.J = J
		self.g = g
		self.h = h
		self.offset = offset
		self.hamil = MPO.MPO(L, 3, 2)
		opL = np.array([[ [1,0,0,0], [0,0,0,J[0]], [offset/L,g[0],0,h[0]] ]])
		opR = np.array([ [[offset/L,g[L-1],0,h[L-1]]], [[0,0,0,1]], [[1,0,0,0]] ])
		self.hamil.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
		self.hamil.ops[L-1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
		for i in range(1, L-1):
			opM = np.array([ [ [1,0,0,0], [0,0,0,J[i]], [offset/L,g[i],0,h[i]] ],
							 [ [0,0,0,0], [0,0,0,0], [0,0,0,1] ],
							 [ [0,0,0,0], [0,0,0,0], [1,0,0,0] ] ])
			self.hamil.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)


class Heisenberg:
	def __init__(self, L, Jx, Jy, Jz, g, h, offset):
		self.L = L
		self.Jx = Jx
		self.Jy = Jy
		self.Jz = Jz
		self.g = g
		self.h = h
		self.offset = offset
		self.hamil = MPO.MPO(L, 5, 2)
		opL = np.array([[ [1,0,0,0], [0,Jx[0],0,0], [0,0,Jy[0],0], [0,0,0,Jz[0]],
						  [offset/L,g[0],0,h[0]] ]])
		opR = np.array([ [[offset/L,g[L-1],0,h[L-1]]], [[0,1,0,0]], [[0,0,1,0]],
					     [[0,0,0,1]], [[1,0,0,0]] ])
		self.hamil.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
		self.hamil.ops[L-1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
		for i in range(1, L-1):
			opM = np.array([ [ [1,0,0,0], [0,Jx[i],0,0], [0,0,Jy[i],0],
							   [0,0,0,Jz[i]], [offset/L,g[i],0,h[i]] ],
							 [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,1,0,0] ],
							 [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,1,0] ],
							 [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,1] ],
							 [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0] ]
						   ])
			self.hamil.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)
