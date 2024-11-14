import numpy as np
# import cupy as cp
from numba import njit
from numpy.linalg import norm, inv, pinv
import math

np.random.seed(1337)



@njit(parallel=True, fastmath=True, cache=True)
def __decision_svp(B, n, R, sigma, C):
    num_samples = (2**(C*n))*int(math.log2(n))
    np.random.seed(1337+np.random.randint(1,10))
    s = np.zeros(n, dtype=np.float64)
    l = 2 ** norm(B)
    B_pinv = inv(B)
    for i in range(num_samples):
        r = np.random.normal(R, sigma)
        direction = np.random.normal(0,1,n)
        v = r * direction
        x = B @ np.round(B_pinv @ v)
        x_norm = norm(x)
        if(x_norm>1e-5 and x_norm<l):
            l = x_norm
            s = x
    return s, l


def __search_svp(B, n, t):
	l, r = 0, min(norm(x) for x in B)
	s, L = -1, r
	# print(l,r)
	for _ in range(int(math.log2(n)+1)):
		m=(l+r)/2
		# print(_, int(m), int(L), '->', int(l), int(r))
		_s, _L = __decision_svp(B, n, m, m, t)
		if(_L<L):
			s, L=_s, _L
		else:
			_s, _L = s, L
		r=_L
		if(_L>m):
			l=m
		else:
			l=m/2
	return s,L
