"""
Contract MPSs and / or MPOs
Find ground state of am MPO (DMRG)
Sum up / compress MPS
Apply MPO to MPS
"""

import sys
import copy
import time
import numpy as np
import numpy.random
import scipy.linalg as LA
import scipy.sparse.linalg as sp_linalg

import MPS
import MPO


def contractMPSL(bra, ket, start, end, left):
	"""
	Input: left =
	bra ┌ ┬ ... ┬
	ket └ ┴ ... ┴       if start>0 and [[1]] if start==0
	    0 1 ... start-1
	Return a list of length end-start+1:
	  bra ┌ ┬ ... ┬                 bra ┌ ┬ ... ┬
	[ ket └ ┴ ... ┴        ...      ket └ ┴ ... ┴   ]
	      0 1 ... start,         ,      0 1 ... end
	"""
	if (bra.L != ket.L):
		print("Error: inconsistent length!")
		exit(2)
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
	# Similar to contractMPSL, but start from the right
	if (bra.L != ket.L):
		print("Error: inconsistent length!")
		exit(2)
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
	# Give the overlap of bra and ket
	left = np.array([[1]])
	return np.complex(contractMPSL(bra, ket, 0, bra.L-1, left)[-1])


def contractMPSMPOL(bra, op, ket, start, end, left):
	# Similar to contractMPS series, with an operator between bra and ket
	if (bra.L != ket.L or bra.L != op.L):
		print("Error: inconsistent length!")
		exit(2)
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
		exit(2)
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


def dmrg(hamil, gs, cutD, tol = 1e-8, maxRound = 0, silent=False):
	"""
	DMRG
	Input an MPO Hamiltonian (hamil) and an initial state (gs)
	Return the ground state energy and the ground state vector is stored in gs
	Algorithm: see arXiv:1008.3477v2, Chapter 6.3
	"""
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
	# Sweep and optimize until convergence
	while (done == False):
		start = time.clock()
		if (right == True):
			opL = idMat if pos==0 else contractL[0]
			opR = idMat if pos==L-1 else contractR[pos+1]
		else:
			opL = idMat if pos==0 else contractL[pos-1]
			opR = idMat if pos==L-1 else contractR[0]
		"""
		opL =                   opR =
		gs     ┌ ┬ ... ┬        gs     ┬ ...  ┬   ┐
		hamil  ├ ┼ ... ┼        hamil  ┼ ...  ┼   ┤
		gs     └ ┴ ... ┴        gs     ┴ ...  ┴   ┘
		       0 1 ... pos-1       pos+1 ... L-2 L-1
		"""

		Hp = np.einsum('ijk,pqjn->ikpqn', opL, hamil.ops[pos].A)
		Hp = np.einsum('ikpqn,mnl->pimqkl', Hp, opR)
		HpDim = gs.sites[pos].s*gs.sites[pos].Dl*gs.sites[pos].Dr
		Hp = Hp.reshape((HpDim, HpDim))
		"""
		Hp =
		gs       ┌ ┬ ... ┬ i   m ┬ ...  ┬   ┐
                             p
		hamil    ├ ┼ ... ┼ j ┼ n ┼ ...  ┼   ┤
                             q
		gs       └ ┴ ... ┴ k   l ┴ ...  ┴   ┘
		         0 1 ...     pos   ... L-2 L-1
		"""
		w, M = sp_linalg.eigs(Hp, k=1, which='SR')
		newEnergy = np.real(w[0])
		newSitePos = M[:,0]
		gs.sites[pos].A = newSitePos.reshape((gs.sites[pos].s,
			gs.sites[pos].Dl, gs.sites[pos].Dr))

		if (newEnergy-energy > 1e-8 and newEnergy-energy > np.abs(energy)*1e-10):
			if (newEnergy-energy > np.abs(energy)*tol):
				print("Error: energy is increasing! Old energy =", energy,
					  "New energy =", newEnergy)
				exit(3)
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


def sumMPS(kets, coef, cutD, givenMPS = False, newMPS = 0,
					tol = 1e-7, maxRound = 0, silent = False, compress = False):
	"""
	Sum up MPS.
	Input:  a list of MPS and a list of corresponding coefficients
	Output: approximation of their sum in MPS with bond dimension cutD
	Algorithm: see also arXiv:1008.3477v2, Chapter 4.5.2, in compressing MPS.
		Only need to add up all M (P in the paper) with coefficients as weights.
		Here each time two sites are modified together.
	"""
	num = len(kets)
	if (compress == False):
		print("Sum up MPS")
	L = kets[0].L
	s = kets[0].sites[0].s
	for i in range(num):
		if kets[i].L != L:
			print("Error: inconsistent length!")
			exit(2)
		if kets[i].sites[0].s != s:
			print("Error: inconsistent physical dimension!")
			exit(2)
	if (givenMPS == False):
		newMPS = MPS.MPS(L, cutD, s)
		newMPS.setRandomState()

	newMPS.gaugeCondMixed(0, 0, L-1, cutD = cutD)
	idMat = np.array([[1]], dtype=complex)
	contractL = []
	contractR = []
	for i in range(num):
		contractL.append([idMat])
		contractR.append(contractMPSR(newMPS, kets[i], 0, L-1, idMat))

	dist0 = 0
	for i in range(num):
		dist0 += contractMPS(kets[i], kets[i]) * np.abs(coef[i]) ** 2
		for j in range(i):
			tmp = contractMPS(kets[i], kets[j]) * coef[j] * np.conj(coef[i])
			dist0 += tmp + np.conj(tmp)

	overlap = 0
	for i in range(num):
		overlap += contractMPS(kets[i], newMPS) * np.conj(coef[i])

	dist = dist0 + contractMPS(newMPS, newMPS) - overlap - np.conj(overlap)
	dist = np.sqrt(np.abs(dist) / np.real(dist0))
	lastDist = dist
	print("Starting with initial distance", dist)
	totalStartTime = time.clock()

	round = 0
	done = False
	pos = 0
	right = True
	while (done == False):
		start = time.clock()
		opL = []
		opR = []
		for i in range(num):
			if (right == True):
				tmpOpL = idMat if pos==0 else contractL[i][0]
				tmpOpR = idMat if pos==L-2 else contractR[i][pos+2]
			else:
				tmpOpL = idMat if pos==0 else contractL[i][pos-1]
				tmpOpR = idMat if pos==L-2 else contractR[i][0]
			opL.append(tmpOpL)
			opR.append(tmpOpR)

		M = np.zeros((newMPS.sites[pos].s,newMPS.sites[pos].Dl,
				      newMPS.sites[pos+1].s,newMPS.sites[pos+1].Dr),
					 dtype=complex)
		for i in range(num):
			tmpM = np.einsum('ij,pjk->ipk', opL[i], kets[i].sites[pos].A)
			tmpM = np.einsum('ipk,qkl->ipql', tmpM, kets[i].sites[pos+1].A)
			tmpM = np.einsum('ipql,ml->piqm', tmpM, opR[i])
			M += tmpM * coef[i]
			"""
			tmpM =
			newMPS       ┌ ┬ ... ┬ i       m ┬  ...  ┬   ┐
	                                 p   q
			kets[i]      └ ┴ ... ┴ j ┴ k ┴ l ┴  ...  ┴   ┘
			             0 1 ...    pos pos+1   ... L-2 L-1
			"""

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
				exit(3)
			else:
				print("Warning: distance is increasing! Old dist =", lastDist,
					  "New dist =", dist, "! Continue.")
		end = time.clock()

		if (silent==False):
			print("Round", round, ", pos", pos, "and", pos+1,
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

		for i in range(num):
			if (right == True):
				if (pos == 1):
					contractL[i]=contractMPSL(newMPS, kets[i], 0, 0, idMat)
					contractR[i]=contractMPSR(newMPS, kets[i], 0, L-1, idMat)
				else:
					contractL[i]=contractMPSL(newMPS, kets[i], pos-1, pos-1,
											  contractL[i][0])
			else:
				if (pos == L-3):
					contractR[i]=contractMPSR(newMPS, kets[i], L-1, L-1, idMat)
					contractL[i]=contractMPSL(newMPS, kets[i], 0, L-1, idMat)
				else:
					contractR[i]=contractMPSR(newMPS, kets[i], pos+2, pos+2,
											  contractR[i][0])
	print()
	return newMPS


def compressMPS(initMPS, cutD, givenMPS = False, newMPS = 0,
					tol = 1e-7, maxRound = 0, silent = False):
	# Call sumMPS
	print("Compress MPS")
	return sumMPS([initMPS], [1.], cutD, givenMPS = givenMPS, newMPS = newMPS,
		tol = tol, maxRound = maxRound, silent = silent, compress = True)


def exactApplyMPO(op, ket):
	# Act an MPO on an MPS exactly
	if (op.L != ket.L):
		print("Error: inconsistent length!")
		exit(2)
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
	# First act exactly, then compress
	newKet = exactApplyMPO(op, ket)
	newKet = compressMPS(newKet, cutD, tol=tol)
	return newKet
