"""
Define MPS
Transform an MPS into (mixed) canonical form
Get entanglement entropy
Set an MPS to a random or product state
"""

import os
import numpy as np
import numpy.random
import copy
import scipy.linalg as LA

from core import Site
from core import Contractor as ct


class MPS:
    def __init__(self, L, D, s):
        self.L = L			# length
        self.sites = []
        self.D = D			# D = max{Dl, Dr}
        self.s = np.ones((L), dtype=int) * s if isinstance(s, int) else s
        for i in range(L):
            self.sites.append(Site.site(self.s[i],
                                        1 if i == 0 else D, 1 if i == L - 1 else D))

    def setA(self, k, Ak):
        if (k < 0 or k >= self.L):
            print("Error: k", k, "out of range!")
            exit(2)
        if (self.s[k] != Ak.shape[0]):
            print(Ak.shape[0])
            print(self.s[k])
            print("Warning: inconsistent physical dimension!")
        self.sites[k].A = Ak
        self.sites[k].s = Ak.shape[0]
        self.s[k] = Ak.shape[0]
        self.sites[k].Dl = Ak.shape[1]
        self.sites[k].Dr = Ak.shape[2]

    def gaugeCond(self, dir, normal=1, cutD=0, silent=False):
        """
        Transform MPS into canonical form
        dir = 1 -> left
        dir = 2 -> right
        """
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
                tmp = self.sites[self.L - i - 1].gaugeL(tmp, cutD)
            elif (dir == 2):
                tmp = self.sites[i].gaugeR(tmp, cutD)
        if (cutD > 0 and cutD < self.D):
            self.D = cutD
        if (normal == 0):
            self.sites[0].A *= np.complex(tmp)
        return np.complex(tmp)

    def gaugeCondMixed(self, left, k, right, cutD=0, silent=True):
        """
        Set mixed gauge condition:
        left - k-1 -> left
        k+1 - right -> right
        """
        if (silent == False):
            print("Set mixed gauge condition centered at", k)
        tmp = np.identity(self.sites[left].Dl, dtype=complex)
        for i in range(left, k):
            tmp = self.sites[i].gaugeR(tmp, cutD)

        self.sites[k].A = np.einsum('ij,kjl->kil', tmp, self.sites[k].A)
        self.sites[k].Dl = self.sites[k].A.shape[1]

        tmp = np.identity(self.sites[right].Dr, dtype=complex)
        for i in range(right - k):
            tmp = self.sites[right - i].gaugeL(tmp, cutD)
        self.sites[k].A = np.einsum('ijk,kl->ijl', self.sites[k].A, tmp)
        self.sites[k].Dr = self.sites[k].A.shape[2]
        if (cutD > 0 and cutD < self.D):
            self.D = cutD

    def getEntanglementEntropy(self, k, cutD=0):
        # Return the entanglement entropy between sites 0 ~ k and k+1 ~ L-1
        if (k > self.L - 1 or k < 0):
            print("Error: inconsistent length!")
            exit(2)

        if (cutD == 0):
            cutD = self.D
        tmpMPS = copy.deepcopy(self)
        tmpMPS.gaugeCond(2, cutD=cutD)
        tmpV = ct.contractMPSR(
            tmpMPS, tmpMPS, k, tmpMPS.L - 1, np.array([[1]]))[0]
        tmpV = tmpV.reshape((tmpMPS.sites[k].Dl, tmpMPS.sites[k].Dl))
        eigs, vecs = LA.eig(tmpV)
        eigs = np.real(eigs)
        eigsPositive = np.where(eigs > 1e-9, eigs, 1.)
        entropy = np.abs(-np.sum(eigsPositive * np.log(eigsPositive)))
        return entropy

    def adjustD(self, DPrime):
        # Adjust the bond dimension of MPS to DPrime
        if (DPrime < self.D):
            # Shrink
            # gaugeCond for small self.D. Otherwise compressMPS in Contractor
            # self.gaugeCond(2, normal=0, cutD=DPrime)
            self = ct.compressMPS(self, DPrime, silent=True)
        else:
            # Enlarge
            for i in range(self.L):
                DrPrime = DPrime if i < self.L - 1 else 1
                self.sites[i].A = np.append(self.sites[i].A,
                                            np.zeros((self.sites[i].s, self.sites[i].Dl,
                                                      DrPrime - self.sites[i].Dr)), axis=2)
                self.sites[i].Dr = DrPrime
                DlPrime = DPrime if i > 0 else 1
                self.sites[i].A = np.append(self.sites[i].A,
                                            np.zeros((self.sites[i].s, DlPrime - self.sites[i].Dl,
                                                      self.sites[i].Dr)), axis=1)
                self.sites[i].Dl = DlPrime
            self.D = DPrime

    def applyLocalOperator(self, k, op):
        # Apply a local operator to MPS
        if (k < 0 or k >= self.L):
            print("Error: k out of range!")
            return False
        opShape = op.shape
        if (opShape[0] == self.sites[k].s and opShape[1] == self.sites[k].s):
            self.sites[k].A = np.einsum('ij,jkl->ikl', op, self.sites[k].A)
            return True
        else:
            print("Error: shape does not fit!")
            return False

    def setRandomState(self):
        # print("Set random state")
        for i in range(self.L):
            self.sites[i].A = numpy.random.random((self.sites[i].s,
                                                   self.sites[i].Dl, self.sites[i].Dr)) \
                + 1j * numpy.random.random((self.sites[i].s,
                                            self.sites[i].Dl, self.sites[i].Dr))
        self.gaugeCond(2, normal=1, cutD=self.D, silent=True)

    def setProductState(self, localState):
        sLocal = localState.shape[0]
        for i in range(self.L):
            self.sites[i].s = sLocal
            self.sites[i].A = localState.reshape((sLocal, 1, 1))
            self.sites[i].Dl = 1
            self.sites[i].Dr = 1
        self.D = 1

    def saveMPS(self, directory):
        if os.path.isdir(directory):
            print("Directory already exists! Will cover the saved MPS.")
        else:
            os.makedirs(directory)
            print("Directory " + directory + " created! Will save MPO")

        np.savetxt(directory + "/L", np.array([self.L]), fmt='%i')
        np.savetxt(directory + "/s", self.s, fmt='%i')
        for i in range(self.L):
            np.savetxt(directory + "/D_" + str(i),
                       np.array([self.sites[i].Dl, self.sites[i].Dr]), fmt='%i')
            tmpA = self.sites[i].A.reshape(-1)
            np.savetxt(directory + "/A_" + str(i),
                       numpy.column_stack([tmpA.real, tmpA.imag]))


def loadMPS(directory):
    if (not os.path.isdir(directory)):
        print("Error: no such MPS directory!")
        exit(2)
    else:
        L = np.loadtxt(directory + "/L", dtype=int)
        s = np.loadtxt(directory + "/s", dtype=int)
        newMPS = MPS(L, 1, s)
        for i in range(L):
            Ds = np.loadtxt(directory + "/D_" + str(i), dtype=int)
            AsReal, AsImag = np.loadtxt(
                directory + "/A_" + str(i), unpack=True)
            newMPS.setA(i,
                        (AsReal + 1j * AsImag).reshape(s[i], Ds[0], Ds[1]))
    return newMPS
