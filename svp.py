import numpy as np
from numba import njit, prange
from numpy.linalg import norm, pinv
from concurrent.futures import ThreadPoolExecutor
import math
from cpp import svp as cpp_svp

np.random.seed(1337)


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def __decision_svp(B, R, sigma, sample_size, _seed):
    def sample(A, A_, n, R, sigma):
        r = np.random.normal(R, sigma)
        direction = np.random.normal(0, 1, n)
        direction = direction / norm(direction)
        v = r * direction
        z = A @ np.round(A_ @ v)
        return z, norm(z)

    n, m = B.shape
    B_pinv = pinv(B)

    short_vector = np.zeros(n, dtype=np.float64)
    len_vector = 2 ** norm(B)
    for counter in range(sample_size):
        np.random.seed(_seed + np.random.randint(1, _seed))
        z, norm_z = sample(B, B_pinv, n, R, sigma)
        if norm_z > 1e-5 and norm_z < len_vector:
            len_vector = norm_z
            short_vector = z
            if len_vector <= R + 1e-5:
                return short_vector, len_vector, counter, True
    return short_vector, len_vector, 0, False


def __decision_svp__(B, mu, sigma, C=0.5, _seed=1337):
    n, m = B.shape
    sample_size = 2 ** (C * m)
    _B, c, verdict = cpp_svp.basis_reduction(
        B.astype(float), float(mu), float(sigma), int(1000), _seed
    )
    _B = np.array(_B).astype(float)

    if verdict:
        return _B[0], c, verdict

    s, l, c, verdict = cpp_svp.decision_svp(
        B.astype(float), float(mu), float(sigma), int(sample_size), _seed
    )

    # return s, norm(s), c, verdict
    return s, c, verdict


def multi_thread_decision_svp(B, R, C=0.5, _seed=1337, num_threads=8):
    n, m = B.shape
    x = math.log2(num_threads) / m + C
    print(x)

    chunks_seed = [
        _seed + np.random.randint(_seed, 2 * _seed) for i in range(num_threads)
    ]
    short_vector, norm_vector, exp_const, verdict = [], [], [], []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(__decision_svp__, B * 10, R * 10, R, C, chunks_seed[i]): i
            for i in range(num_threads)
        }
        for future in futures:
            try:
                result = future.result()
                v = result[0]
                z = B @ v
                short_vector.append(z)
                norm_vector.append(norm(z))
                exp_const.append(result[1])
                verdict.append(result[2])
                print(f"Thread {futures[future]} is done: {result[2]}, {norm(z)}")
            except Exception as e:
                print(f"Thread {futures[future]} raised an exception: {e}")

    # all_same = True
    # for i in range(1, num_threads):
    # 	s1, s2 = short_vector[i], short_vector[i-1]
    # 	all_same = all_same and ((s1 == s2).all() or (s1 == -1*s2).all())
    # print('Are all vectors equal? ', all_same)

    solution_vector = np.zeros(n, dtype=np.float64)
    solution_norm = 2**R
    for i in range(num_threads):
        if verdict[i]:
            x = math.log2(sum(exp_const)) / m
            return short_vector[i], norm_vector[i], x
        if solution_norm > norm_vector[i] + 1e-3:
            solution_norm = norm_vector[i]
            solution_vector = short_vector[i]

    return solution_vector, solution_norm, -x


def __search_svp(B, n):
    l, r = 0, min(norm(x) for x in B)
    s, L = -1, r
    for _ in range(int(math.log2(n) + 1)):
        m = (l + r) / 2
        _s, _L = multi_thread_decision_svp(B, m)
        if _L < L:
            s, L = _s, _L
        else:
            _s, _L = s, L
        r = _L
        if _L > m:
            l = m
        else:
            l = m / 2
    return s, L
