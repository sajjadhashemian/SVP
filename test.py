import numpy as np
import pandas as pd
from numpy.linalg import norm
from fpylll import FPLLL, SVP, CVP
from copy import copy
import math
import time
import random
from svp import decision_svp
from lattice_generator import reduced_basis, generate_challange, generate_knapsack_instance

np.random.seed(13371)
FPLLL.set_random_seed(13371)
random.seed(1337)
_exp_const=0.35

def solve_svp(X, n, m, C=0.5, _seed=1337):
	A, B = reduced_basis(X, n, m)

	X=copy(A)
	SVP.shortest_vector(A)
	s, l = A[0], norm(A[0])

	t1=time.time()
	s, _l, c = decision_svp(B, l, C, _seed)
	t2=time.time()
	t=t2-t1

	v0 = CVP.closest_vector(X, tuple([round(x) for x in s]))
	v1 = bool(abs(norm(v0)-_l)<1e-3)
	# v2 = bool(_l/l<=1+e)
	v2 = bool(_l<=l+1e-3)
	return c, t, v1, v2, _l/l, _l, l


def test_kanpsack_instance(n, b, _seed):
	X = generate_knapsack_instance(n, b, _seed)
	c, t, v1, v2, ratio, x, y = solve_svp(X, n, n+1, _exp_const, _seed)
	verdict = v1 and v2
	print(v1, v2, ratio, x, y)
	assert verdict==True
	return c, t, verdict, ratio

def test_challange(n, _seed):
	X = generate_challange(n, _seed)
	n, m = X.shape
	c, t, v1, v2, ratio, x, y = solve_svp(X, n, m, _exp_const, _seed)
	verdict = v1 and v2
	print(v1, v2, ratio, x, y)
	assert verdict==True
	return c, t, verdict, ratio


if __name__=='__main__':

	for n in range(40, 42):
		print('Warmup', n)
		c, t, v, r = test_challange(n, 1337+n)

	_dict=dict()
	num_of_test=1
	low, up = 40, 80
	b = 62
	counter=1
	seed_list=[1021, 2221, 3121, 4421, 5021, 6221, 7121, 8221, 9721, 220614721]
	COLUMNS = ['Lattice', 'Dimensions', 'Exp Constant', 'Time (s)', 'Verdict', 'Ratio', 'Seed']
	
	for n in range(low, up, 2):
		print('-------- test dimension', n)
		for i in range(num_of_test):
			print('test number', i)
			_seed = 220614721

			# c, t, v, r = test_challange(n, _seed)
			# _dict[counter] = ['Challange', n, c, t, v, r, _seed]
			# print('Challange')
			# counter+=1

			c, t, v, r = test_kanpsack_instance(n, b, _seed)
			_dict[counter] = ['Knapsack', n, c, t, v, r, _seed]
			print('Knapsack')
			counter+=1

			res = pd.DataFrame.from_dict(_dict, orient='index', columns=COLUMNS)
			res.to_csv('./results/results6.csv')