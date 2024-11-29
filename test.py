import numpy as np
import pandas as pd
from numpy.linalg import norm
from fpylll import FPLLL, SVP, CVP
from copy import copy
import math
import time

from svp import decision_svp
from lattice_generator import generate_random_instance, generate_hard_instance, reduced_basis, generate_challange

np.random.seed(1337)
FPLLL.set_random_seed(1337)


def solve_svp(X, n, m, C=0.5):
	A, B = reduced_basis(X, n, m)

	X=copy(A)
	SVP.shortest_vector(A)
	s, l = A[0], norm(A[0])

	t1=time.time()
	s, _l, c = decision_svp(B, l, C)
	t2=time.time()
	t=t2-t1

	v0 = CVP.closest_vector(X, tuple([int(x) for x in s]))
	e=0.001
	v1 = bool(abs(norm(v0)-_l)<e)
	v2 = bool(_l<=l+e)
	return c, t, v1, v2, _l/l


def test_random_instance(n, b, _seed):
	X = generate_random_instance(b, n, _seed)
	c, t, v1, v2, ratio = solve_svp(X, n, n)
	verdict = v1 and v2
	assert verdict==True
	return c, t, verdict, ratio
	


def test_hard_instance(n, p, r, _seed):
	X = generate_hard_instance(n, p, r, _seed)
	c, t, v1, v2, ratio = solve_svp(X, n, n)
	verdict = v1 and v2
	assert verdict==True
	return c, t, verdict, ratio


def test_challange(n, _seed):
	X = generate_challange(n, _seed)
	n, m = X.shape
	c, t, v1, v2, ratio = solve_svp(X, n, m)
	verdict = v1 and v2
	assert verdict==True
	return c, t, verdict, ratio


if __name__=='__main__':
	_dict=dict()
	for n in range(40, 42):
		print('Warmup', n)
		c, t, v, r = test_challange(n, 1337+n)

	low, up = 40, 60
	b = 18
	for n in range(low, up):
		print('test_random_instance', n)
		c, t, v, r = test_random_instance(n, b, 1337+n)
		_dict[('test_random_instance', n)]= [c, t, v, r]
		(pd.DataFrame.from_dict(_dict, orient='index')).to_csv('data.csv')
	
	p = int(1e9+7)
	for n in range(low, up):
		print('test_hard_instance', n)
		c, t, v, r = test_hard_instance(n, p, int(n/4), 1337+n)
		_dict[('test_hard_instance', n)]= [c, t, v, r]
		(pd.DataFrame.from_dict(_dict, orient='index')).to_csv('data.csv')
	
	for n in range(low, up):
		print('test_challange', n)
		c, t, v, r = test_challange(n, 1337+n)
		_dict[('test_challange', n)]= [c, t, v, r]
		(pd.DataFrame.from_dict(_dict, orient='index')).to_csv('data.csv')


	

