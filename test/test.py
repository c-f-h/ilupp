import numpy as np
import scipy.sparse
import ilupp

def laplace_matrix(n):
    h = 1.0 / (n + 1)
    d = np.ones(n) / (h**2)
    return scipy.sparse.diags((-d[:-1], 2*d, -d[:-1]), (-1, 0, 1)).tocsr()

def laplace_Matrix_2d(n):
    A, I = laplace_matrix(n), scipy.sparse.eye(n)
    return (scipy.sparse.kron(A, I) + scipy.sparse.kron(I, A)).tocsr()

def example_2d(n):
    A = laplace_Matrix_2d(n)
    m = A.shape[0]
    x_exact = np.ones(m)
    b = A.dot(x_exact)
    return A, b, x_exact

def random_example(n):
    A = scipy.sparse.random(n, n, density=0.1, random_state=39273, format='csc') + 10*scipy.sparse.eye(n)
    x_exact = np.ones(n)
    b = A.dot(x_exact)
    return A, b, x_exact

########################################

## multilevel preconditioner

def test_ml_solve_laplace_PQ():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_PQ()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace_MWM():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace_MWM_sPQ():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace_SF():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_SPARSE_FIRST()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_random():
    A, b, x_exact = random_example(50)
    x = ilupp.solve(A, b, atol=1e-8)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)

def test_ml_precond_laplace():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    P = ilupp.ILUppPreconditioner(A, threshold=1000)
    x = P.dot(b)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    print('total nnz:', P.total_nnz, 'memory:', P.memory)

def test_ml_precond_random():
    A, b, x_exact = random_example(50)
    P = ilupp.ILUppPreconditioner(A, threshold=1000)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)

## ILUT preconditioner

def test_ILUT_laplace():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    P = ilupp.ILUTPreconditioner(A, threshold=0.0)
    x = P.dot(b)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    # each L/u factor has a diagonal and an off-diagonal,
    # but ILU++ reports by n less for unknown reasons
    print('total nnz:', P.total_nnz)
    assert P.total_nnz <= 2 * (n + (n-1))

def test_ILUT_random():
    A, b, x_exact = random_example(50)
    P = ilupp.ILUTPreconditioner(A, fill_in=1000, threshold=0.0)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)

## ILUTP preconditioner

def test_ILUTP_laplace():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    P = ilupp.ILUTPPreconditioner(A, threshold=0.0)
    x = P.dot(b)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    # each L/u factor has a diagonal and an off-diagonal,
    # but ILU++ reports by n less for unknown reasons
    print('total nnz:', P.total_nnz)
    assert P.total_nnz <= 2 * (n + (n-1))

def test_ILUTP_random():
    A, b, x_exact = random_example(50)
    P = ilupp.ILUTPPreconditioner(A, fill_in=1000, threshold=0.0)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)

## ILUC preconditioner

def test_ILUC_laplace():
    n = 100
    A = laplace_matrix(n)
    b = np.ones(n)
    P = ilupp.ILUCPreconditioner(A, threshold=0.0)
    x = P.dot(b)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    # each L/u factor has a diagonal and an off-diagonal,
    # but ILU++ reports by n less for unknown reasons
    print('total nnz:', P.total_nnz)
    assert P.total_nnz <= 2 * (n + (n-1))

def test_ILUC_random():
    A, b, x_exact = random_example(50)
    P = ilupp.ILUCPreconditioner(A, fill_in=1000, threshold=0.0)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)
