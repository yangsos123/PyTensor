"""
Define each site for MPO
"""

import numpy as np
import numpy.random
import scipy.linalg as LA

class siteOp:
	def __init__(self, s, Dl, Dr):
		self.s = s			# physical dimension
		self.Dl = Dl		# left bond dimension
		self.Dr = Dr		# right bond dimension
		self.A = np.zeros((s, s, Dl, Dr), dtype=complex)
