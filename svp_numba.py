import numpy as np
# import cupy as cp
from numba import njit
from numpy.linalg import norm
from fpylll import FPLLL, SVP, CVP
from copy import copy

import time
import math

from svp import decision_svp
from lattice_generator import generate_random_instance, generate_hard_instance, reduced_basis

np.random.seed(1337)
FPLLL.set_random_seed(1337)

# 57, 68, 73, 84
n, b = 84, 18
X = generate_random_instance(b, n)

# n, p, r= 17, 97, 3
# X = generate_hard_instance(n, p, r)

A, B = reduced_basis(X, n)

SVP.shortest_vector(A)
s, l = A[0], norm(A[0])
print('------------------------------------')
print('fpylll solution:')
# print([int(x) for x in s],'\n Norm:', l)
# print([int(x) for x in inv(B)@s])
print('Norm:', l)

print('------------------------------------')
print('my solution:')
C = 0.5														#Exponent constant: number of samples, Working fine, but C~20: 2*e*pi
t1=time.time()
s, _l, c = decision_svp(B, n, l, C)
t2=time.time()

# print([-int(x) for x in s],'\n Norm:', _l)
print('Norm:', _l)
print('exponent constant: ', c)

t=int(t2-t1)
ms=int(((t2-t1)-t)*1e3)
print('elapsed time: ',t//60, 'm', t%60, 's', ms, 'ms')


print('------------------------------------')
print('verify solution:')
# print(X)
# print('coefficients vector:  ',[int(x) for x in pinv(B)@s])
# print('reconstructed vector: ' ,list(B@[int(x) for x in pinv(B)@s]))

v0 = CVP.closest_vector(X, tuple([int(x) for x in s]))
# print('Verify with CVP:      ',[int(x) for x in v0],'\n norm:',norm(v0))
e=0.0001
print('Verdict:', bool(abs(norm(v0)-_l)<e) and bool(_l<=l+e), ', ', _l/l)

