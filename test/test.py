import numpy as np
import scipy.sparse
import ilupp
import unittest

def laplace_matrix(n, format='csr'):
    h = 1.0 / (n + 1)
    d = np.ones(n) / (h**2)
    return scipy.sparse.diags((-d[:-1], 2*d, -d[:-1]), (-1, 0, 1)).asformat(format)

def laplace_Matrix_2d(n, format='csr'):
    A, I = laplace_matrix(n), scipy.sparse.eye(n)
    return (scipy.sparse.kron(A, I) + scipy.sparse.kron(I, A)).asformat(format)

def example_2d(n):
    A = laplace_Matrix_2d(n)
    m = A.shape[0]
    x_exact = np.ones(m)
    b = A.dot(x_exact)
    return A, b, x_exact

def example_laplace(n):
    A = laplace_matrix(n)
    b = np.ones(n)
    X = np.linspace(0, 1, n+2)[1:-1]
    x_exact = X*(1-X)/2
    return A, b, x_exact

def example_random(n, format='csc'):
    A = (scipy.sparse.random(n, n, density=0.1, random_state=39273) + 10*scipy.sparse.eye(n)).asformat(format)
    x_exact = np.ones(n)
    b = A.dot(x_exact)
    return A, b, x_exact

########################################
# auto-generated test cases

def _gen_solve_in_one_step(Precond, params, example_func, example_args):
    def test(self):
        A, b, x_exact = example_func(*example_args)
        P = Precond(A, **params)
        x = b.copy()
        P.apply(x)
        print('Error:', np.linalg.norm(x - x_exact))
        assert np.allclose(x, x_exact)
    return test

def _gen_test_with_predicate(Precond, params, example_func, example_args, pred):
    def test(self):
        A, b, x_exact = example_func(*example_args)
        P = Precond(A, **params)
        assert pred(A, P)
    return test

def _assert_factors_correct(A, P):
    L, U = P.factors()
    LU = L.dot(U)
    return np.allclose(A.A, LU.A)

class TestCases(unittest.TestCase):
    for P in [
            ilupp.ILUTPreconditioner,
            ilupp.ILUTPPreconditioner,
            ilupp.ILUCPreconditioner,
            ilupp.ILUCPPreconditioner,
    ]:
        base_name = 'test_' + P.__name__ + '_'
        case_name = base_name + 'random'
        vars()[case_name] = _gen_solve_in_one_step(P, {'threshold': 0.0}, example_random, (50,))

        case_name = base_name + 'laplace'
        vars()[case_name] = _gen_solve_in_one_step(P, {'threshold': 0.0}, example_laplace, (50,))

        # each L/U factor has a diagonal and an off-diagonal,
        # but ILU++ reports less in some cases because of unit diagonals
        case_name = base_name + 'total_nnz'
        vars()[case_name] = _gen_test_with_predicate(P, {'threshold': 0.0}, example_laplace, (50,),
                lambda A, pr: pr.total_nnz <= 2 * (2*A.shape[0] - 1))

        case_name = base_name + 'laplace_factorscorrect'
        vars()[case_name] = _gen_test_with_predicate(P, {'threshold': 0.0}, example_laplace, (50,), _assert_factors_correct)

        case_name = base_name + 'random_factorscorrect'
        vars()[case_name] = _gen_test_with_predicate(P, {'threshold': 0.0}, example_random, (50,), _assert_factors_correct)


    # ILUCP currently fails for CSR matrices due to a not implemented permuted triangular solve
    del vars()['test_ILUCPPreconditioner_laplace']

    # the correct evaluation of the factors needs a right permutation for pivoted ILUC (not implemented)
    del vars()['test_ILUCPPreconditioner_laplace_factorscorrect']


########################################

## multilevel preconditioner

def test_ml_solve_laplace2d_PQ():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_PQ()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_MWM():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_MWM_sPQ():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_SF():
    A, b, x_exact = example_2d(30)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_SPARSE_FIRST()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_random():
    A, b, x_exact = example_random(50)
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
    A, b, x_exact = example_random(50)
    P = ilupp.ILUppPreconditioner(A, threshold=1000)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)
