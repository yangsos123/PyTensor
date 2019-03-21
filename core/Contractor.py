"""
Contract MPSs and / or MPOs
Find ground state of am MPO
Apply MPO to MPS
Transform between MPS and MPO
"""

import sys
import copy
import time
import numpy as np
import numpy.random
import scipy.linalg as LA
import scipy.sparse.linalg as sp_linalg

sys.path.append("MPS.py")
import MPS
sys.path.append("MPO.py")
import MPO


def contractMPSL(bra, ket, start, end, left):
	if (bra.L != ket.L):
		print("Error: inconsistent length!")
		exit(1)
	else:
		L = bra.L

	allRes = []
	res = np.einsum('ij,kil->jkl', left, np.conj(bra.sites[start].A))
	res = np.einsum('jkl,kjm->lm', res, ket.sites[start].A)
	allRes.append(res)
	for i in range(start+1, end+1):
		res = np.einsum('ij,kil->jkl', res, np.conj(bra.sites[i].A))
		res = np.einsum('jkl,kjm->lm', res, ket.sites[i].A)
		allRes.append(res)
	return allRes


def contractMPSR(bra, ket, start, end, right):
	if (bra.L != ket.L):
		print("Error: inconsistent length!")
		exit(1)
	else:
		L = bra.L

	allRes = []
	res = np.einsum('ij,kli->jkl', right, np.conj(bra.sites[end].A))
	res = np.einsum('jkl,kmj->lm', res, ket.sites[end].A)
	allRes.append(res)
	for i in reversed(range(start, end)):
		res = np.einsum('ij,kli->jkl', res, np.conj(bra.sites[i].A))
		res = np.einsum('jkl,kmj->lm', res, ket.sites[i].A)
		allRes.append(res)
	return allRes[::-1]


def contractMPS(bra, ket):
	left = np.array([[1]])
	return np.complex(contractMPSL(bra, ket, 0, bra.L-1, left)[-1])


def contractMPSMPOL(bra, op, ket, start, end, left):
	if (bra.L != ket.L or bra.L != op.L):
		print("Error: inconsistent length!")
		exit(1)
	else:
		L = bra.L

	allRes = []
	res = np.einsum('ijk,pim->jkpm', left, np.conj(bra.sites[start].A))
	res = np.einsum('jkpm,pqjr->kmqr', res, op.ops[start].A)
	res = np.einsum('kmqr,qkn->mrn', res, ket.sites[start].A)
	allRes.append(res)
	for i in range(start+1, end+1):
		res = np.einsum('ijk,pim->jkpm', res, np.conj(bra.sites[i].A))
		res = np.einsum('jkpm,pqjr->kmqr', res, op.ops[i].A)
		res = np.einsum('kmqr,qkn->mrn', res, ket.sites[i].A)
		allRes.append(res)
	return allRes


def contractMPSMPOR(bra, op, ket, start, end, right):
	if (bra.L != ket.L or bra.L != op.L):
		print("Error: inconsistent length!")
		exit(1)
	else:
		L = bra.L
	allRes = []
	res = np.einsum('ijk,pmi->jkpm', right, np.conj(bra.sites[end].A))
	res = np.einsum('jkpm,pqrj->kmqr', res, op.ops[end].A)
	res = np.einsum('kmqr,qnk->mrn', res, ket.sites[end].A)
	allRes.append(res)
	for i in reversed(range(start, end)):
		res = np.einsum('ijk,pmi->jkpm', res, np.conj(bra.sites[i].A))
		res = np.einsum('jkpm,pqrj->kmqr', res, op.ops[i].A)
		res = np.einsum('kmqr,qnk->mrn', res, ket.sites[i].A)
		allRes.append(res)
	return allRes[::-1]


def contractMPSMPO(bra, op, ket):
	left = np.array([[[1]]])
	return np.complex(contractMPSMPOL(bra, op, ket, 0, bra.L-1, left)[-1])


def findGroundState(hamil, gs, cutD, tol = 1e-8, maxRound = 0, silent=False):
	L = hamil.L
	gs.adjustD(cutD)
	gs.gaugeCondMixed(0,0,L-1, cutD=cutD)
	idMat = np.array([[[1]]], dtype=complex)
	contractR = contractMPSMPOR(gs, hamil, gs, 0, L-1, idMat)
	energy = np.real(contractR[0][0,0,0] / contractMPS(gs, gs))
	energyLastRound = energy
	print("Starting findGroundState with initial energy", energy)

	round = 0
	done = False
	pos = 0
	right = True
	totalStartTime = time.clock()
	while (done == False):
		start = time.clock()
		if (right == True):
			opL = idMat if pos==0 else contractL[0]
			opR = idMat if pos==L-1 else contractR[pos+1]
		else:
			opL = idMat if pos==0 else contractL[pos-1]
			opR = idMat if pos==L-1 else contractR[0]

		Hp = np.einsum('ijk,pqjn->ikpqn', opL, hamil.ops[pos].A)
		Hp = np.einsum('ikpqn,mnl->pimqkl', Hp, opR)
		HpDim = gs.sites[pos].s*gs.sites[pos].Dl*gs.sites[pos].Dr
		#print(gs.sites[pos].s, gs.sites[pos].Dl, gs.sites[pos].Dr)
		Hp = Hp.reshape((HpDim, HpDim))
		w, M = sp_linalg.eigs(Hp, k=1, which='SR')
		#print("W ",w,"M ", M)
		newEnergy = np.real(w[0])
		newSitePos = M[:,0]
		gs.sites[pos].A = newSitePos.reshape((gs.sites[pos].s,
			gs.sites[pos].Dl, gs.sites[pos].Dr))
		if (newEnergy-energy > 1e-8 and newEnergy-energy > np.abs(energy)*1e-10):
			if (newEnergy-energy > np.abs(energy)*tol):
				print("Error: energy is increasing! Old energy =", energy,
					  "New energy =", newEnergy)
				exit(1)
			else:
				print("Warning: energy is increasing! Old energy =", energy,
					  "New energy =", newEnergy, "! Continue.")
		energy = newEnergy
		end = time.clock()
		if (silent == False):
			print("Round", round, ", pos", pos,
					", moving", "right ," if right==True else "left ,",
					"energy", energy, ", wall time", end-start, "s")

		if (right == True):
			if (pos < L-1):
				pos += 1
				gs.gaugeCondMixed(pos-1, pos, pos, cutD=cutD)
			else:
				right = False
				pos -= 1
				round += 1
				if (np.abs(energyLastRound - energy) < tol):
					done = True
					print("Energy converged since last round!")
					print("Ground state energy =", energy,
						  ", total wall time", end - totalStartTime, "s")
				elif (maxRound > 0 and round == maxRound):
					done = True
					print("Reached maximum round!")
				else:
					gs.gaugeCondMixed(pos, pos, pos+1, cutD=cutD)
					energyLastRound = energy
		else:
			if (pos > 0):
				pos -= 1
				gs.gaugeCondMixed(pos, pos, pos+1, cutD=cutD)
			else:
				right = True
				pos += 1
				round += 1
				if (np.abs(energyLastRound - energy) < tol):
					done = True
					print("Energy converged since last round!")
					print("Ground state energy =", energy,
						  ", total wall time", end - totalStartTime, "s")
				elif (maxRound > 0 and round == maxRound):
					done = True
					print("Reached maximum round!")
				else:
					gs.gaugeCondMixed(pos-1, pos, pos, cutD=cutD)
					energyLastRound = energy

		if (right == True):
			if (pos == 1):
				contractL = contractMPSMPOL(gs, hamil, gs, 0, 0, idMat)
				contractR = contractMPSMPOR(gs, hamil, gs, 0, L-1, idMat)
			else:
				contractL = contractMPSMPOL(gs, hamil, gs, pos-1, pos-1, contractL[0])
		else:
			if (pos == L-2):
				contractR = contractMPSMPOR(gs, hamil, gs, L-1, L-1, idMat)
				contractL = contractMPSMPOL(gs, hamil, gs, 0, L-1, idMat)
			else:
				contractR = contractMPSMPOR(gs, hamil, gs, pos+1, pos+1, contractR[0])
	print()
	return energy


"""
def compressMPS(initMPS, cutD, givenMPS = False, newMPS = MPS.MPS(1,1,1),
					tol = 1e-5, maxRound = 0, silent = False):
	# One site version
	print("Compress MPS")
	L = initMPS.L
	s = initMPS.sites[0].s
	if (givenMPS == False):
		newMPS = MPS.MPS(L, cutD, s)
		newMPS.setRandomState()

	newMPS.gaugeCondMixed(0, 0, L-1, cutD = cutD)
	idMat = np.array([[1]], dtype=complex)
	contractL = idMat
	contractR = contractMPSR(newMPS, initMPS, 0, L-1, idMat)
	dist0 = contractMPS(initMPS, initMPS)
	overlap = contractMPS(initMPS, newMPS)
	dist = dist0 + contractMPS(newMPS, newMPS) - overlap - np.conjugate(overlap)
	dist = np.sqrt(np.abs(dist) / np.real(dist0))
	lastDist = dist
	print("Starting compressMPS with initial distance", dist)
	totalStartTime = time.clock()

	round = 0
	done = False
	pos = 0
	right = True
	while (done == False):
		start = time.clock()
		if (right == True):
			opL = idMat if pos==0 else contractL[0]
			opR = idMat if pos==L-1 else contractR[pos+1]
		else:
			opL = idMat if pos==0 else contractL[pos-1]
			opR = idMat if pos==L-1 else contractR[0]

		M = np.einsum('ij,kjm->ikm', opL, initMPS.sites[pos].A)
		M = np.einsum('ikm,lm->kil', M, opR)

		newMPS.sites[pos].A = M
		#overlap = contractMPS(initMPS, newMPS)
		#dist = dist0 + contractMPS(newMPS, newMPS) - overlap - np.conjugate(overlap)
		overlap = np.einsum('ijk,ijk->', np.conj(M), M)
		dist = dist0 - overlap
		dist = np.sqrt(np.abs(dist) / np.real(dist0))

		if (dist - lastDist > 1e-8):
			if (dist - lastDist > 1e-8 * lastDist):
				print("Error: distance is increasing! Old dist =", lastDist,
					  "New dist =", dist)
				exit(1)
			else:
				print("Warning: distance is increasing! Old dist =", lastDist,
					  "New dist =", dist, "! Continue.")
		end = time.clock()
		if (silent==False):
			print("Round", round, ", pos", pos,
					", moving", "right ," if right==True else "left ,",
					"distance", dist, ", wall time", end-start, "s")

		if (right == True):
			if (pos < L-1):
				pos += 1
				newMPS.gaugeCondMixed(pos-1, pos, pos, cutD=cutD)
			else:
				right = False
				pos -= 1
				round += 1
				if (dist < tol):
					done = True
					print("Distance converged (", dist,
						  "), total wall time", end - totalStartTime, "s")
				elif (maxRound > 0 and round == maxRound):
					done = True
					print("Reached maximum round!")
				else:
					newMPS.gaugeCondMixed(pos, pos, pos+1, cutD=cutD)
		else:
			if (pos > 0):
				pos -= 1
				newMPS.gaugeCondMixed(pos, pos, pos+1, cutD=cutD)
			else:
				right = True
				pos += 1
				round += 1
				if (dist < tol):
					done = True
					print("Distance converged", dist,
						  ", total wall time", end - totalStartTime, "s")
				elif (maxRound > 0 and round == maxRound):
					done = True
					print("Reached maximum round!")
				else:
					newMPS.gaugeCondMixed(pos-1, pos, pos, cutD=cutD)

		if (right == True):
			if (pos == 1):
				contractL = contractMPSL(newMPS, initMPS, 0, 0, idMat)
				contractR = contractMPSR(newMPS, initMPS, 0, L-1, idMat)
			else:
				contractL = contractMPSL(newMPS, initMPS, pos-1, pos-1, contractL[0])
		else:
			if (pos == L-2):
				contractR = contractMPSR(newMPS, initMPS, L-1, L-1, idMat)
				contractL = contractMPSL(newMPS, initMPS, 0, L-1, idMat)
			else:
				contractR = contractMPSR(newMPS, initMPS, pos+1, pos+1, contractR[0])
	print()
	return newMPS
"""


def compressMPS(initMPS, cutD, givenMPS = False, newMPS = MPS.MPS(1,1,1),
					tol = 1e-7, maxRound = 0, silent = False):
	# Two site version
	print("Compress MPS")
	L = initMPS.L
	s = initMPS.sites[0].s
	if (givenMPS == False):
		newMPS = MPS.MPS(L, cutD, s)
		newMPS.setRandomState()

	newMPS.gaugeCondMixed(0, 0, L-1, cutD = cutD)
	idMat = np.array([[1]], dtype=complex)
	contractR = contractMPSR(newMPS, initMPS, 0, L-1, idMat)
	dist0 = contractMPS(initMPS, initMPS)
	overlap = contractMPS(initMPS, newMPS)
	dist = dist0 + contractMPS(newMPS, newMPS) - overlap - np.conjugate(overlap)
	dist = np.sqrt(np.abs(dist) / np.real(dist0))
	lastDist = dist
	print("Starting compressMPS with initial distance", dist)
	totalStartTime = time.clock()

	round = 0
	done = False
	pos = 0
	right = True
	while (done == False):
		start = time.clock()
		if (right == True):
			opL = idMat if pos==0 else contractL[0]
			opR = idMat if pos==L-2 else contractR[pos+2]
		else:
			opL = idMat if pos==0 else contractL[pos-1]
			opR = idMat if pos==L-2 else contractR[0]

		M = np.einsum('ij,pjk->ipk', opL, initMPS.sites[pos].A)
		M = np.einsum('ipk,qkl->ipql', M, initMPS.sites[pos+1].A)
		M = np.einsum('ipql,ml->piqm', M, opR)

		M = M.reshape((M.shape[0]*M.shape[1], M.shape[2]*M.shape[3]))
		U, S, Vdag = LA.svd(M, full_matrices=False)
		if (S.shape[0]>cutD):
			U = U[:,:cutD]
			#print(np.sum(S[0:cut]**2)/np.sum(S**2))
			S = S[0:cutD]
			Vdag = Vdag[0:cutD,:]
		if (right == True):
			SV = np.einsum('i,ij->ij', S, Vdag)
			SV = np.swapaxes(SV.reshape((-1, s, newMPS.sites[pos+1].Dr)), 0, 1)
			newMPS.setA(pos, U.reshape((s,newMPS.sites[pos].Dl,-1)))
			newMPS.setA(pos+1, SV)
		else:
			US = np.einsum('ij,j->ij', U, S)
			Vdag = np.swapaxes(Vdag.reshape((-1, s, newMPS.sites[pos+1].Dr)), 0, 1)
			newMPS.setA(pos, US.reshape((s,newMPS.sites[pos].Dl,-1)))
			newMPS.setA(pos+1, Vdag)


		overlap = np.einsum('ij,ij->', np.conj(M), M)
		dist = dist0 - overlap
		dist = np.sqrt(np.abs(dist) / np.real(dist0))

		if (dist - lastDist > 1e-8):
			if (dist - lastDist > 1e-8 * lastDist):
				print("Error: distance is increasing! Old dist =", lastDist,
					  "New dist =", dist)
				exit(1)
			else:
				print("Warning: distance is increasing! Old dist =", lastDist,
					  "New dist =", dist, "! Continue.")
		end = time.clock()

		if (silent==False):
			print("Round", round, ", pos", pos,
					", moving", "right ," if right==True else "left ,",
					"distance", dist, ", wall time", end-start, "s")

		if (right == True):
			if (pos < L-2):
				pos += 1
			else:
				right = False
				pos -= 1
				round += 1
				if (dist < tol):
					done = True
					print("Distance converged (", dist,
						  "), total wall time", end - totalStartTime, "s")
				elif (maxRound > 0 and round == maxRound):
					done = True
					print("Reached maximum round!")
				else:
					newMPS.gaugeCondMixed(pos, pos, L-1, cutD=cutD)
		else:
			if (pos > 0):
				pos -= 1
			else:
				right = True
				pos += 1
				round += 1
				if (dist < tol):
					done = True
					print("Distance converged", dist,
						  ", total wall time", end - totalStartTime, "s")
				elif (maxRound > 0 and round == maxRound):
					done = True
					print("Reached maximum round!")
				else:
					newMPS.gaugeCondMixed(pos-1, pos, pos, cutD=cutD)

		if (right == True):
			if (pos == 1):
				contractL = contractMPSL(newMPS, initMPS, 0, 0, idMat)
				contractR = contractMPSR(newMPS, initMPS, 0, L-1, idMat)
			else:
				contractL = contractMPSL(newMPS, initMPS, pos-1, pos-1, contractL[0])
		else:
			if (pos == L-3):
				contractR = contractMPSR(newMPS, initMPS, L-1, L-1, idMat)
				contractL = contractMPSL(newMPS, initMPS, 0, L-1, idMat)
			else:
				contractR = contractMPSR(newMPS, initMPS, pos+2, pos+2, contractR[0])
	print()
	return newMPS


def exactApplyMPO(op, ket):
	if (op.L != ket.L):
		print("Error: inconsistent length!")
		exit(1)
	else:
		L = ket.L
	newKet = MPS.MPS(L,1,1)
	for i in range(L):
		tmp = np.einsum('ijkl,jmn->ikmln', op.ops[i].A, ket.sites[i].A)
		tmp = tmp.reshape((op.ops[i].s, ket.sites[i].Dl*op.ops[i].Dl,
								  ket.sites[i].Dr*op.ops[i].Dr))
		newKet.setA(i, tmp)
	return newKet


def fitApplyMPO(op, ket, cutD, tol=1e-7):
	newKet = exactApplyMPO(op, ket)
	newKet = compressMPS(newKet, cutD, tol=tol)
	return newKet


def sumMPS(kets, coef, cutD):
	num = len(kets)
