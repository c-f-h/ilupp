import numpy as np
import scipy.sparse
import ilupp

def laplace_matrix(n):
    h = 1.0 / (n + 1)
    d = np.ones(n) / (h**2)
    return scipy.sparse.diags((-d[:-1], 2*d, -d[:-1]), (-1, 0, 1)).tocsr()

def test_solve():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    x, info = ilupp.solve(A, b, rtol=1e-6, info=True)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    print('Convergence info:', info)
    ##
    A = scipy.sparse.random(50, 50, density=0.1, format='csc') + 10*scipy.sparse.eye(50)
    x_exact = np.ones(50)
    b = A.dot(x_exact)
    x = ilupp.solve(A, b, atol=1e-8)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)

def test_precond():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    P = ilupp.ILUppPreconditioner(A, threshold=1000)
    x = P.dot(b)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    nnz, mem = P.total_nnz, P.memory
    ##
    A = scipy.sparse.random(50, 50, density=0.1, format='csc') + 10*scipy.sparse.eye(50)
    P = ilupp.ILUppPreconditioner(A, threshold=1000)
    x_exact = np.ones(50)
    b = A.dot(x_exact)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)

def test_ILUT():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    P = ilupp.ILUTPreconditioner(A, threshold=0.0)
    x = P.dot(b)
    print(b)
    print(x)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    # each L/u factor has a diagonal and an off-diagonal,
    # but ILU++ reports by n less for unknown reasons
    assert P.total_nnz <= 2 * (n + (n-1))
    ##
    A = scipy.sparse.random(50, 50, density=0.1, random_state=39273, format='csc') + 10*scipy.sparse.eye(50)
    P = ilupp.ILUTPreconditioner(A, fill_in=1000, threshold=0.0)
    x_exact = np.ones(50)
    b = A.dot(x_exact)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)

def test_ILUC():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    P = ilupp.ILUCPreconditioner(A, threshold=0.0)
    x = P.dot(b)
    print(b)
    print(x)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    # each L/u factor has a diagonal and an off-diagonal,
    # but ILU++ reports by n less for unknown reasons
    assert P.total_nnz <= 2 * (n + (n-1))
    ##
    A = scipy.sparse.random(50, 50, density=0.1, random_state=39273, format='csc') + 10*scipy.sparse.eye(50)
    P = ilupp.ILUCPreconditioner(A, fill_in=1000, threshold=0.0)
    x_exact = np.ones(50)
    b = A.dot(x_exact)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)
