import numpy as np
from numpy.linalg import norm, inv, pinv
# from fpylll import FPLLL, IntegerMatrix, SVP, LLL, BKZ
# from fpylll.algorithms.bkz import BKZReduction
from scipy.special import gammainc, gammaincinv, gammainccinv
from copy import copy
import math
# from multiprocessing import Pool, cpu_count
# from svp import __svp as __svp__

np.random.seed(1337)

def f_sigma_R(r, sigma, R):
    return np.exp(-np.abs((r**2 - R**2) / sigma))

def sample_direction(n, c):
    vec = np.random.normal(0+c, 1-c, n)
    return vec / np.linalg.norm(vec)

def inverse_cdf(p, sigma, R, n):
    lower_gamma_R = gammainc(n / 2, R**2 / sigma)
    if p <= lower_gamma_R:  # Case 1: r <= R
        r_squared = sigma * gammaincinv(n / 2, p)
        return np.sqrt(r_squared)
    else:  # Case 2: r > R
        r_squared = sigma* gammainccinv(n / 2, p - lower_gamma_R)
        return np.sqrt(r_squared)

def inverse_transform_sampling(R, sigma, n):
    c=0.2
    u = np.random.uniform(0+c, 1-c)
    r = inverse_cdf(u, sigma, R, n)
    direction = sample_direction(n, c)
    sample = r * direction    
    return sample

def random_sampling(R, sigma, n):
    r = np.random.normal(R, sigma)
    direction = np.random.normal(0,1,n)
    return r * direction

def lattice_vector(v, B):
    # return B@[int(x) for x in pinv(B)@v]
    # return B@[int(x) for x in inv(B)@v]
    return B@np.array([int(x) for x in pinv(B)@v])

def __svp(B, n, R, sigma, num_samples):
	np.random.seed(1337+np.random.randint(1,10))
	s, l = -1, 2**n*norm(B)
	for i in range(num_samples):
		# sample = inverse_transform_sampling(R, sigma, n)
		sample = random_sampling(R, sigma, n)
		x=lattice_vector(sample, B)
		if(norm(x)>0 and norm(x)<l):
			l=norm(x)
			s=copy(x)
	return s,l
