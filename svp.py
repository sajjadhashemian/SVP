import numpy as np
# import cupy as cp
from numba import njit, prange
from numpy.linalg import norm, inv, pinv, lstsq
import math
import decision_svp
# import sys
np.random.seed(1337)
# config.THREADING_LAYER = 'omp'  # or 'omp'


@njit(parallel=True, fastmath=True, cache=True, nogil=True, debug=True)
def __decision_svp(B, R, sigma, batch, _seed):
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
	for i1 in range(batch[0]):
		for i2 in range(batch[1]):
			for i3 in range(batch[2]):
				for i4 in range(batch[3]):
					z, norm_z = sample(B, B_pinv, n, R, sigma)
					if(norm_z>1e-5 and norm_z<len_vector):
						len_vector = norm_z
						short_vector = z
						if(len_vector<=R+1e-5):
							return short_vector, len_vector, [i1, i2, i3, i4], True
	return short_vector, len_vector, [0, 0, 0, 0], False


def decision_svp__(B, R, C=0.5, _seed=1337):
	n, m = B.shape
	temp0 = min(20, max(0, C*m-40))
	temp1 = min(30, max(0, C*m-(40+temp0)))
	batch_size = [2**20, 2**20, 2**temp0, 2**temp1]
	t = [int(round(x)) for x in batch_size]
	s, l, c, verdict = decision_svp.decision_svp(B.astype(float), float(R), float(R), int(2**(C*m)), _seed)
	l = norm(s)
	if(verdict==False):
		return s, l, -C
	else:
		# x = c[0]*t[1]*t[2]*t[3] + c[1]*t[2]*t[3] + c[2]*t[3]+ c[3]
		# print(x, c)
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
