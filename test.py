import numpy as np
from numpy.linalg import norm
from fpylll import FPLLL, SVP, CVP
from copy import copy
import math
from svp import __decision_svp
from lattice_generator import generate_random_instance, generate_hard_instance, reduced_basis

np.random.seed(1337)
FPLLL.set_random_seed(1337)


def solve_svp(X, n, C):
	A, B = reduced_basis(X, n)

	X=copy(A)
	SVP.shortest_vector(A)
	s, l = A[0], norm(A[0])

	# C = 1	#Working fine, but C~20: 2*e*pi
	num_samples=int(2**(C*n))*int(math.log2(n))
	s, _l = __decision_svp(B, n, l, l, num_samples)

	v0 = CVP.closest_vector(X, tuple([int(x) for x in s]))
	verdict=bool(abs(norm(v0)-_l)<1e-3)
	verdict= verdict and bool(_l<=l+1e-3)
	return verdict


def test_random_instance(n, b, C=1):
	X = generate_random_instance(b, n)
	verdict = solve_svp(X, n, C)
	assert verdict==True


def test_hard_instance(n, p, r, C=1):
	X = generate_hard_instance(n, p , r)
	verdict = solve_svp(X, n, C)
	assert verdict==True

if __name__=='__main__':
	n, b= 10, 32
	test_random_instance(n, b)
	n, p, r= 10, 97, 7
	test_hard_instance(n, p, r)

