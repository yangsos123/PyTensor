"""
Define MPO,
Transform between MPS and MPO,
Get MPO of real / imaginary revolution operator.
"""

import os
import numpy as np
import numpy.random
import scipy.linalg as LA

from core import SiteOp
from core import MPS
from core import Contractor as ct


class MPO:
    """
    Class for MPO.
    """

    def __init__(self, L, D, s):
        """
        Initialize an MPO class with length L (int), bond dimension D (int) and physical dimension s 
            (int or numpy int array of length L). 
        D and s can be changed later.
        self.ops is a list of A matrices in siteOp type (import SiteOp and try help(SiteOp) for more information), 
            and are initially zero.
        """
        self.L = L
        self.D = D
        self.s = np.ones((L), dtype=int) * s if isinstance(s, int) else s
        self.ops = []
        for i in range(L):
            self.ops.append(SiteOp.siteOp(self.s[i], 1 if i == 0 else D,
                                          1 if i == L - 1 else D))

    def setA(self, k, Ak):
        """
        Set the A matrix of this MPO at site k to Ak (numpy complex array in shape (s, s, Dl, Dr)).
        The bond dimension and physical dimension at this site will be changed correspondingly.
        """
        if (Ak.shape[0] != Ak.shape[1]):
            print("Error: inconsistent physical dimension!")
            exit(2)
        elif (k < 0 or k >= self.L):
            print("Error: k", k, "out of range!")
            exit(2)
        else:
            if (Ak.shape[0] != self.s[k]):
                print(Ak.shape[0])
                print(self.s[k])
                print("Warning: inconsistent physical dimension!")
            self.ops[k].A = Ak
            self.ops[k].s = Ak.shape[0]
            self.s[k] = Ak.shape[0]
            self.ops[k].Dl = Ak.shape[2]
            self.ops[k].Dr = Ak.shape[3]
            if (Ak.shape[2] > self.D):
                self.D = Ak.shape[2]
            if (Ak.shape[3] > self.D):
                self.D = Ak.shape[3]

    def setProductOperator(self, localOp):
        """
        Set the MPO to a product operator with localOp (numpy complex array of shape (s, s)).
        """
        for i in range(self.L):
            s = localOp.shape[0]
            self.ops[i].A = localOp.reshape((s, s, 1, 1))
            self.ops[i].Dl = 1
            self.ops[i].Dr = 1
            self.ops[i].s = s
            self.s[i] = s
        self.D = 1

    def saveMPO(self, directory):
        """
        Save this MPO to a directory. Each MPO needs one directory.
        """
        if os.path.isdir(directory):
            print("Directory already exists! Will cover the saved MPO.")
        else:
            os.makedirs(directory)
            print("Directory " + directory + " created! Will save MPO")

        np.savetxt(directory + "/L", np.array([self.L]), fmt='%i')
        np.savetxt(directory + "/s", self.s, fmt='%i')
        for i in range(self.L):
            np.savetxt(directory + "/D_" + str(i),
                       np.array([self.ops[i].Dl, self.ops[i].Dr]), fmt='%i')
            tmpA = self.ops[i].A.reshape(-1)
            np.savetxt(directory + "/A_" + str(i),
                       numpy.column_stack([tmpA.real, tmpA.imag]))


def loadMPO(directory):
    """
    Return an MPO loaded from a directory (saved by savedMPO function in MPO class).
    """
    if (not os.path.isdir(directory)):
        print("Error: no such MPO directory!")
        exit(2)
    else:
        L = np.loadtxt(directory + "/L", dtype=int)
        s = np.loadtxt(directory + "/s", dtype=int)
        newMPO = MPO(L, 1, s)
        for i in range(L):
            Ds = np.loadtxt(directory + "/D_" + str(i), dtype=int)
            AsReal, AsImag = np.loadtxt(
                directory + "/A_" + str(i), unpack=True)
            newMPO.setA(i,
                        (AsReal + 1j * AsImag).reshape(s[i], s[i], Ds[0], Ds[1]))
    return newMPO


def MPSfromMPO(mpo):
    """
    Return the MPS version of MPO by make physical dimension s*s.
    Input:             Return:
    mpo:               mps:
    i1 i2      iL      i1j1 i2j2      iLjL
    ├  ┼ ... ┼ ┤        └    ┴  ... ┴  ┘
    j1 j2      jL
    """
    L = mpo.L
    mps = MPS.MPS(L, 1, 1)
    for i in range(L):
        opShape = mpo.ops[i].A.shape
        mps.setA(i, mpo.ops[i].A.reshape((opShape[0] * opShape[1],
                                          opShape[2], opShape[3])))
    return mps


def MPOfromMPS(mps):
    """
    Inverse transformation of MPSfromMPO.
    """
    L = mps.L
    mpo = MPO(L, 1, 1)
    for i in range(L):
        opShape = mps.sites[i].A.shape
        mpo.setA(i, mps.sites[i].A.reshape((int(np.sqrt(opShape[0])),
                                            int(np.sqrt(opShape[0])), opShape[1], opShape[2])))
    return mpo


def extendMPO(simpleMPO):
    """
    Return an modified MPO (doubleMPO) from the originally one (simpleMPO) so that it is compatible with an
    MPS (mpsMPO, oringinally mpoMPO) got from MPSfromMPO. 
    i.e., MPOfromMPO(doubleMPO|mpsMPO>) = simpleMPO*mpoMPO.
    """
    L = simpleMPO.L
    doubleMPO = MPO(L, 1, 1)
    for i in range(L):
        s = simpleMPO.ops[i].s
        idPhy = np.identity(s).reshape((s, s, 1, 1))
        doubleMPO.setA(i, np.kron(simpleMPO.ops[i].A, idPhy))
    return doubleMPO


def getUMPO(L, s, hi, hrestL, hrestR, t, imag=True, cutD=0):
    """
    Input: length of system L (int), physical dimension s (int), 
           local / nearest neighbour operators hi (1<=i<=L-2) 
             (numpy complex array of shape (s, s) or (s**2, s**2)),
           local / nearest neighbour operators hrestL (i=0), hrestR (i=L-1),
           evolution time t.
    Return: UMPO = e^{-Ht} if imag else e^{-iHt}.

    This function actually calculate e^{-iHt} ~ e^{-iH_odd t/2}e^{-iH_even t}e^{-iH_odd t/2} and 
    the error = O(t^3). Try to choose smaller t and apply the UMPO many times.
    """
    if (imag == True):
        exph = LA.expm(-1j * hi * t)
        exphHalf = LA.expm(-1j * hi * t / 2)
        exphrestL = LA.expm(-1j * hrestL * t / 2)
        if (L % 2 == 1):
            exphrestR = LA.expm(-1j * hrestR * t)
        else:
            exphrestR = LA.expm(-1j * hrestR * t / 2)
    else:
        exph = LA.expm(-hi * t)
        exphHalf = LA.expm(-hi * t / 2)
        exphrestL = LA.expm(-hrestL * t / 2)
        if (L % 2 == 1):
            exphrestR = LA.expm(-hrestR * t)
        else:
            exphrestR = LA.expm(-hrestR * t / 2)
    exphrestL = exphrestL.reshape((s, s, 1, 1))
    exphrestR = exphrestR.reshape((s, s, 1, 1))

    exph = exph.reshape((s, s, s, s))
    exph = np.swapaxes(exph, 1, 2)
    exph = exph.reshape((s * s, s * s))
    svdU, svdS, svdV = LA.svd(exph, full_matrices=False)
    svdL = svdU.reshape((s, s, 1, -1))
    svdR = np.einsum('i,ij->ij', svdS, svdV).reshape((-1, 1, s, s))
    svdR = np.swapaxes(svdR, 0, 2)
    svdR = np.swapaxes(svdR, 1, 3)

    Ueven = MPO(L, 1, s)
    for i in range(L // 2):
        Ueven.setA(i * 2, svdL)
        Ueven.setA(i * 2 + 1, svdR)
    if (L % 2 == 1):
        Ueven.setA(L - 1, exphrestR)

    exphHalf = exphHalf.reshape((s, s, s, s))
    exphHalf = np.swapaxes(exphHalf, 1, 2)
    exphHalf = exphHalf.reshape((s * s, s * s))
    svdUHalf, svdSHalf, svdVHalf = LA.svd(exphHalf, full_matrices=False)
    svdLHalf = svdUHalf.reshape((s, s, 1, -1))
    svdRHalf = np.einsum('i,ij->ij', svdSHalf, svdVHalf).reshape((-1, 1, s, s))
    svdRHalf = np.swapaxes(svdRHalf, 0, 2)
    svdRHalf = np.swapaxes(svdRHalf, 1, 3)

    UoddHalf = MPO(L, 1, s)
    UoddHalf.setA(0, exphrestL)
    for i in range((L - 1) // 2):
        UoddHalf.setA(i * 2 + 1, svdLHalf)
        UoddHalf.setA(i * 2 + 2, svdRHalf)
    if (L % 2 == 0):
        UoddHalf.setA(L - 1, exphrestR)

    UMPO = ct.joinMPO(UoddHalf, Ueven, cutD=cutD)
    UMPO = ct.joinMPO(UMPO, UoddHalf, cutD=cutD)
    return UMPO
