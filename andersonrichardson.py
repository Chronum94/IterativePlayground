# A very hackey and absolutely not production implementation of the
# Alternating Anderson-Richardson algorithm as described in:
# https://arxiv.org/pdf/1606.08740.pdf

# This is ABSOLUTELY not meant for any production use whatsoever. This was
# purely as a self-amusing example. But it should get an idea of what the method
# is doing, across. I've seen convergence typically for the random matrices
# (strongly diagonally dominant) that are generated.

import numpy as np

def _shift_columns(mat):
    for i in range(1, mat.shape[1]):
        mat[:, i-1] = mat[:, i]
    return mat


def aar(A, b):
    x = np.zeros_like(b)
    n = b.shape[0]
    maxiter = 1000
    resnorm = 1.0
    p = 50 # Number of steps after which to Anderson extrapolate.
    omega = 0.2
    beta = 0.4
    hlen = 3# n//2 # History length. m in the paper.
    eyen = np.eye(b.shape[0])
    bmat =  omega * eyen
    xmatrix = np.zeros([n, hlen])
    fmatrix = np.zeros([n, hlen])

    tol = 1e-11

    f = np.zeros_like(b)
    fold = np.zeros_like(b)
    for k in range(maxiter):
        f = b - A@x # Residual.
        xmatrix = _shift_columns(xmatrix)
        fmatrix = _shift_columns(fmatrix)
        if (k+1)%p != 0:
            xmatrix[:, -1] = omega*f # Scalar multiplied with residual This is just x_k - x_k-1
            fmatrix[:, -1] = f - fold
            x += omega*f
        else:
            # print(fmatrix.T@fmatrix)
            update = beta * f - (xmatrix + beta*fmatrix)@np.linalg.pinv(fmatrix.T@fmatrix)@fmatrix.T@f
            xmatrix[:, -1] = update # Scalar multiplied with residual This is just x_k - x_k-1
            fmatrix[:, -1] = f - fold

            x+= update
        # print(xmatrix, '\n', fmatrix)
        # input()
        fold = f
        resnorm = np.linalg.norm(fold)
        print("Resnorm:", resnorm)
        if resnorm < tol:
            print(k, "iterations.")
            break

    return x

a = np.random.rand(600, 600)*1e-2 + np.eye(600)*2
b = np.random.rand(600)
sol1 = aar(a, b)
sol2 = np.linalg.solve(a, b)
print(np.allclose(sol1, sol2))
# print()
