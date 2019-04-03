"""
Define each site for MPO.
"""

import numpy as np
import numpy.random
import scipy.linalg as LA


class siteOp:
    """
    Store the MPO information at each site.
    """

    def __init__(self, s, Dl, Dr):
        """
        Initialize a siteOp class with physical dimension s, left bond dimension Dl
            and right bond dimension Dr. They will be stored in self.s, self.Dl
            and self.Dr correspondingly. self.A will store the A matrix (numpy 
            complex array of shape (s,s,Dl,Dr)) and is initially zero.
        """
        self.s = s			# physical dimension
        self.Dl = Dl		# left bond dimension
        self.Dr = Dr		# right bond dimension
        self.A = np.zeros((s, s, Dl, Dr), dtype=complex)
