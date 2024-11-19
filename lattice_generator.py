import numpy as np
from fpylll import FPLLL, IntegerMatrix, LLL, BKZ
# from hnf import hermite_normal_form
# from sympy import Matrix
# from sympy.matrices.normalforms import hermite_normal_form
from hsnf import row_style_hermite_normal_form as hermite_normal_form

np.random.seed(13371)
FPLLL.set_random_seed(13317)

def generate_random_instance(b, n):
	A = IntegerMatrix(n,n)
	A.randomize("uniform", bits=b)
	# A=[[-1248884424688331397, 217379346648022931, 968695965772720018],
	#    [-252808953422802017, 44003627585902284, 196091014068821624],
	#    [-480021271462343831, 83551935074919420, 372328023281023077]]
	# A = IntegerMatrix.from_matrix(A)
	return A

def generate_hard_instance(n, q, r):
	"""
	Generates a hard SVP instance with a short basis using Ajtai99.
    n: Lattice dimension
    q: Modulus for lattice entries (should be prime or appropriate for lattice properties)
	r: Number of initial random vectors
	"""
	u_vectors = np.random.randint(0, q, (r, n))
	A1 = np.eye(r, dtype=int)
	A2 = np.random.randint(-q//2, q//2, (n - r, n - r))
	A = np.block([
        [A1, np.zeros((r, n - r), dtype=int)], 
        [np.zeros((n - r, r), dtype=int), A2]
    ])
	B_core = np.tril(np.random.randint(-r, r, (n - r, n - r)))
	B = np.hstack([np.zeros((n - r, r), dtype=int), B_core])  # Pad B to shape (n - r, n)
	lattice_basis = (A @ np.vstack([u_vectors, B]) % q)

	norm_bound = np.sqrt(n) * n
	for i in range(lattice_basis.shape[0]):
		vector_norm = np.linalg.norm(lattice_basis[i])
		if vector_norm > norm_bound:
			lattice_basis[i] = (lattice_basis[i] / vector_norm) * norm_bound

	return lattice_basis.astype(int)

def reduced_basis(X, n, m):
	n, m = X.shape
	B=list([])
	for i in range(n):
		B.append([0 for _ in range(m)])
		for j in range(m):
			B[i][j]=int(X[i][j])
	A = IntegerMatrix.from_matrix(B)

	# A=LLL.reduction(A)
	A = BKZ.reduction(A, BKZ.Param(20))
	
	B=list([])
	for i in range(n):
		B.append([0 for _ in range(m)])
		for j in range(m):
			B[i][j]=int(A[i][j])
	B=np.array(B).T
	return A, B

def generate_challange(m):
	c1 = 2.1
	c2 = c1 * np.log(2) - np.log(2) / (50 * np.log(50))
	n = max(50, int(m / (c1 * np.log(m))))
	q = int(np.floor(n ** c2))
	X = np.random.randint(0, q, (n, m))
	Y = np.block([[X.T, q * np.eye(m, dtype=int)]])
	H, L = hermite_normal_form(Y.astype(float))  # Compute Hermite Normal Form
	H = np.array(H).astype(int)
	return H