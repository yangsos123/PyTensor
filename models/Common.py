import numpy as np

def toArray(L, f):
	if (isinstance(f, int) or isinstance(f, float)):
		return np.ones((L)) * f
	else:
		return f
