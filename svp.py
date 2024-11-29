import numpy as np
# import cupy as cp
from numba import njit, prange
from numpy.linalg import norm, inv, pinv
import math
# import sys
np.random.seed(1337)



@njit(parallel=True, fastmath=True, cache=True)
def __decision_svp(B, R, sigma, num_batch, batch_size, _seed):
	n, m = B.shape
	s = np.zeros(n, dtype=np.float64)
	l = 2 ** norm(B)
	B_pinv = pinv(B)
	for batch in range(num_batch):
		np.random.seed(_seed+np.random.randint(1,_seed))
		for counter in range(batch_size):
			r = np.random.normal(R, sigma)
			direction = np.random.normal(0,1,n)
			v = r * direction
			x = B @ np.round(B_pinv @ v)
			x_norm = norm(x)
			if(x_norm>1e-5 and x_norm<l):
				l = x_norm
				s = x
			if(l<=R+1e-5):
				return s, l, counter, batch
	return s, l, -1, -1


def decision_svp(B, R, C=0.5, _seed=1337):
	n, m = B.shape
	num_samples = (2**(C*m))#*int(math.log(m))
	batch_size=2**30
	num_batch=int((num_samples+batch_size)//batch_size)
	s, l, c, t = __decision_svp(B.astype(float), float(R), float(R), int(num_batch), int(batch_size), _seed)
	if(t==-1):
		return s, l, -C
	else:
		# print(t*batch_size+c)
		x = math.log2(t*batch_size+c+1)/m
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
