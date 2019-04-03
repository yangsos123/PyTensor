"""
Currently only support spin-half models.
Generate Hamiltonians / time evolution MPOs of
	Ising model
	Heiseneberg model,
Get MPO of SumSx, Sy, Sz (squared),
Get MPO of projector to certain total spin.
"""

import sys
import copy
import numpy as np
import numpy.random
import scipy.linalg as LA

from models import Common
from core import MPS
from core import MPO
from core import Contractor as ct

PauliSigma = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [
                      [0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

Up = np.array([1, 0])
Dn = np.array([0, 1])
Xp = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
Xm = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
Yp = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
Ym = np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])
Em = np.array([0, 0])


class Ising:
    """
    Class of Ising model. The Hamiltonian is:
    H = \sum_{i=1}^{L-1} J[i]sigma_i^z sigma_{i+1}^z
        + \sum_{i=1}^{L} (g[i]sigma_i^x + h[i]sigma_{i}^z) + offset.
    """

    def __init__(self, L, J, g, h, offset):
        """
        Initialization.
        Input: system size L (int), interaction J, longitudinal field g, transverse field h 
               (J, g and h are either float or numpy float array of length L) and offset (float).
        Will generate the MPO of Hamiltonian in self.hamil.
        """
        self.L = L
        self.J = Common.toArray(L, J)
        self.g = Common.toArray(L, g)
        self.h = Common.toArray(L, h)
        self.offset = offset
        self.hamil = MPO.MPO(L, 3, 2)
        opL = np.array([[[1, 0, 0, 0], [0, 0, 0, self.J[0]], [
                        offset / L, self.g[0], 0, self.h[0]]]])
        opR = np.array([[[offset / L, self.g[L - 1], 0, self.h[L - 1]]],
                        [[0, 0, 0, 1]], [[1, 0, 0, 0]]])
        self.hamil.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
        self.hamil.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
        for i in range(1, L - 1):
            opM = np.array([[[1, 0, 0, 0], [0, 0, 0, self.J[i]], [offset / L, self.g[i], 0, self.h[i]]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]])
            self.hamil.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)

    def getUMPOIsing(self, t, imag=True, cutD=0):
        """
        Return the time-evolution unitary MPO of Ising model.
        Works only when the model is translational invariant!
        """
        hi = self.J[0] * np.kron(PauliSigma[3, :, :], PauliSigma[3, :, :]) + \
            self.h[0] * np.kron(PauliSigma[3, :, :], PauliSigma[0, :, :]) + \
            self.g[0] * np.kron(PauliSigma[1, :, :],
                                PauliSigma[0, :, :])

        hL = self.h[0] * PauliSigma[3, :, :] + self.g[0] * PauliSigma[1, :, :]
        hR = self.h[0] * PauliSigma[3, :, :] + self.g[0] * PauliSigma[1, :, :]
        return MPO.getUMPO(self.L, 2, hi, hL, hR, t, imag=imag, cutD=cutD)


class Heisenberg:
    """
    Class of Heisenberg model. The Hamiltonian is:
    H = \sum_{p=x,y,z; i=1}^{L-1} J_{p}[i]sigma_i^p sigma_{i+1}^p
        + \sum_{i=1}^{L} (g[i]sigma_i^x + h[i]sigma_{i}^z) + offset.
    """

    def __init__(self, L, Jx, Jy, Jz, g, h, offset):
        """
        Initialization.
        Input: system size L (int), interaction Jx, Jy and Jz, longitudinal field g, transverse field h 
               (Jx, Jy, Jz, g and h are either float or numpy float array of length L) and offset (float).
        Will generate the MPO of Hamiltonian in self.hamil.
        """
        self.L = L
        self.Jx = Common.toArray(L, Jx)
        self.Jy = Common.toArray(L, Jy)
        self.Jz = Common.toArray(L, Jz)
        self.g = Common.toArray(L, g)
        self.h = Common.toArray(L, h)
        self.offset = offset
        self.hamil = MPO.MPO(L, 5, 2)
        opL = np.array([[[1, 0, 0, 0], [0, self.Jx[0], 0, 0], [0, 0, self.Jy[0], 0], [0, 0, 0, self.Jz[0]],
                         [offset / L, self.g[0], 0, self.h[0]]]])
        opR = np.array([[[offset / L, self.g[L - 1], 0, self.h[L - 1]]], [[0, 1, 0, 0]], [[0, 0, 1, 0]],
                        [[0, 0, 0, 1]], [[1, 0, 0, 0]]])
        self.hamil.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
        self.hamil.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
        for i in range(1, L - 1):
            opM = np.array([[[1, 0, 0, 0], [0, self.Jx[i], 0, 0], [0, 0, self.Jy[i], 0],
                             [0, 0, 0, self.Jz[i]], [offset / L, self.g[i], 0, self.h[i]]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [
                                0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [
                                0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [
                                0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [
                                0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
                            ])
            self.hamil.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)

    def getUMPOHeisenberg(self, t, imag=True, cutD=0):
        """
        Return the time-evolution unitary MPO of Heisenberg model.
        Works only when the model is translational invariant!
        """
        hi = self.Jx[0] * np.kron(PauliSigma[1, :, :], PauliSigma[1, :, :]) + \
            self.Jy[0] * np.kron(PauliSigma[2, :, :], PauliSigma[2, :, :]) + \
            self.Jz[0] * np.kron(PauliSigma[3, :, :], PauliSigma[3, :, :]) + \
            self.h[0] * np.kron(PauliSigma[3, :, :], PauliSigma[0, :, :]) + \
            self.g[0] * np.kron(PauliSigma[1, :, :],
                                PauliSigma[0, :, :])

        hL = self.h[0] * PauliSigma[3, :, :] + self.g[0] * PauliSigma[1, :, :]
        hR = self.h[0] * PauliSigma[3, :, :] + self.g[0] * PauliSigma[1, :, :]
        return MPO.getUMPO(self.L, 2, hi, hL, hR, t, imag=imag, cutD=cutD)


def getSumSxMPO(L):
    """
    Return \sum_{i=1}^L sigma_i^x.
    """
    sumSx = MPO.MPO(L, 2, 2)
    opL = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]]])
    opR = np.array([[[0, 1, 0, 0]], [[1, 0, 0, 0]]])
    sumSx.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
    sumSx.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
    for i in range(1, L - 1):
        opM = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]],
                        [[0, 0, 0, 0], [1, 0, 0, 0]]])
        sumSx.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)
    return sumSx


def getSumSyMPO(L):
    """
    Return \sum_{i=1}^L sigma_i^y.
    """
    sumSy = MPO.MPO(L, 2, 2)
    opL = np.array([[[1, 0, 0, 0], [0, 0, 1, 0]]])
    opR = np.array([[[0, 0, 1, 0]], [[1, 0, 0, 0]]])
    sumSy.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
    sumSy.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
    for i in range(1, L - 1):
        opM = np.array([[[1, 0, 0, 0], [0, 0, 1, 0]],
                        [[0, 0, 0, 0], [1, 0, 0, 0]]])
        sumSy.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)
    return sumSy


def getSumSzMPO(L):
    """
    Return \sum_{i=1}^L sigma_i^z.
    """
    sumSz = MPO.MPO(L, 2, 2)
    opL = np.array([[[1, 0, 0, 0], [0, 0, 0, 1]]])
    opR = np.array([[[0, 0, 0, 1]], [[1, 0, 0, 0]]])
    sumSz.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
    sumSz.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
    for i in range(1, L - 1):
        opM = np.array([[[1, 0, 0, 0], [0, 0, 0, 1]],
                        [[0, 0, 0, 0], [1, 0, 0, 0]]])
        sumSz.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)
    return sumSz


def getSumSx2MPO(L):
    """
    Return (\sum_{i=1}^L sigma_i^x) ^ 2
    """
    sumSx2 = MPO.MPO(L, 2, 2)
    opL = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
    opR = np.array([[[1, 0, 0, 0]], [[0, 2, 0, 0]], [[1, 0, 0, 0]]])
    sumSx2.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
    sumSx2.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
    for i in range(1, L - 1):
        opM = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 2, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]])
        sumSx2.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)
    return sumSx2


def getSumSy2MPO(L):
    """
    Return (\sum_{i=1}^L sigma_i^y) ^ 2
    """
    sumSy2 = MPO.MPO(L, 2, 2)
    opL = np.array([[[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]])
    opR = np.array([[[1, 0, 0, 0]], [[0, 0, 2, 0]], [[1, 0, 0, 0]]])
    sumSy2.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
    sumSy2.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
    for i in range(1, L - 1):
        opM = np.array([[[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
                        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 2, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]])
        sumSy2.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)
    return sumSy2


def getSumSz2MPO(L):
    """
    Return (\sum_{i=1}^L sigma_i^z) ^ 2
    """
    sumSz2 = MPO.MPO(L, 2, 2)
    opL = np.array([[[1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]])
    opR = np.array([[[1, 0, 0, 0]], [[0, 0, 0, 2]], [[1, 0, 0, 0]]])
    sumSz2.ops[0].A = np.einsum('ijk,kml->mlij', opL, PauliSigma)
    sumSz2.ops[L - 1].A = np.einsum('ijk,kml->mlij', opR, PauliSigma)
    for i in range(1, L - 1):
        opM = np.array([[[1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]])
        sumSz2.ops[i].A = np.einsum('ijk,kml->mlij', opM, PauliSigma)
    return sumSz2


def getSumSzProjector(L, S):
    """
    Return the projector of total spin S. S is twice the physical spin (so that it is an integer).
    L and S have to be simultaneously even or odd.
    """
    if (L % 2 != S % 2):
        print("Error: impossible total spin")
        exit(2)
    Z = np.zeros((2, 2, 2), dtype=complex)
    Z[0, 0, 0] = 1
    Z[1, 1, 1] = 1
    P = MPO.MPO(L, 1, 2)
    Smin = 0
    Smax = 0
    for i in range(L):
        Dl = (Smax - Smin) // 2 + 1
        tmpA = np.zeros((Dl, Dl + 1, 2))
        np.fill_diagonal(tmpA[:, :, 0], 1)
        np.fill_diagonal(tmpA[:, 1:, 1], 1)

        if (Smax + 2 - L + i <= S):
            Smax += 1
        else:
            Smax -= 1
            tmpA = tmpA[:, 1:]
        if (Smin - 2 + L - i >= S):
            Smin -= 1
        else:
            Smin += 1
            tmpA = tmpA[:, :-1]
        tmpA = np.einsum('ijk,mnk->mnij', tmpA, Z)
        P.setA(i, tmpA)
    return P
