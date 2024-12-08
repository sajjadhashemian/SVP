import numpy as np
from numba import njit, prange, gdb
from numpy.linalg import norm, pinv
import math
np.random.seed(1337)


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def __decision_svp(B, R, sigma, sample_size, _seed):
	def sample(A, A_, n, R, sigma):
		r = np.random.normal(R, sigma)
		direction = np.random.normal(0,1,n)
		v = r * direction
		z = A @ np.round(A_ @ v)
		return z, norm(z)
	
	n, m = B.shape
	short_vector = np.zeros(n, dtype=np.float64)
	len_vector = 2 ** norm(B)
	B_pinv = pinv(B)

	np.random.seed(_seed+np.random.randint(1,_seed))
	for counter in range(sample_size):
		z, norm_z = sample(B, B_pinv, n, R, sigma)
		if(norm_z>1e-5 and norm_z<len_vector):
			len_vector = norm_z
			short_vector = z
			if(len_vector<=R+1e-5):
				return short_vector, len_vector, counter, True
	return short_vector, len_vector, 0, False


def decision_svp(B, R, C=0.5, _seed=1337):
	n, m = B.shape
	s, l, c, verdict = __decision_svp(B.astype(float), float(R), float(R), int(2**(C*m)), _seed)
	l = norm(s)
	if(verdict==False):
		return s, l, -C
	else:
		x = math.log2(c+1)/m
		return s, l, x


def __search_svp(B, n):
	l, r = 0, min(norm(x) for x in B)
	s, L = -1, r
	for _ in range(int(math.log2(n)+1)):
		m=(l+r)/2
		_s, _L = decision_svp(B, m)
		if(_L<L):
			s, L = _s, _L
		else:
			_s, _L = s, L
		r=_L
		if(_L>m):
			l=m
		else:
			l=m/2
	return s,L
