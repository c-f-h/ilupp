import numpy as np
import scipy.sparse
import ilupp
import unittest
import math

################################################################################
## test problems

def laplace_matrix(n, format='csr'):
    h = 1.0 / (n + 1)
    d = np.ones(n) / (h**2)
    return scipy.sparse.diags((-d[:-1], 2*d, -d[:-1]), (-1, 0, 1)).asformat(format)

def laplace2d_matrix(n_total, format='csr'):
    n = int(math.sqrt(n_total))
    A, I = laplace_matrix(n), scipy.sparse.eye(n)
    return (scipy.sparse.kron(A, I) + scipy.sparse.kron(I, A)).asformat(format)

def example_laplace2d(n_total):
    A = laplace2d_matrix(n_total)
    m = A.shape[0]
    x_exact = np.ones(m)
    b = A.dot(x_exact)
    return A, b, x_exact

def example_laplace(n, format='csr'):
    A = laplace_matrix(n, format=format)
    b = np.ones(n)
    X = np.linspace(0, 1, n+2)[1:-1]
    x_exact = X*(1-X)/2
    return A, b, x_exact

def random_matrix(n, format='csr'):
    avg_nnz_per_row = 5
    density = min(1.0, avg_nnz_per_row / n)
    return (scipy.sparse.random(n, n, density=density, random_state=39273) + 10*scipy.sparse.eye(n)).asformat(format)

def example_random(n, format='csc'):
    A = random_matrix(n, format=format)
    x_exact = np.ones(n)
    b = A.dot(x_exact)
    return A, b, x_exact

_test_problems = {
    'laplace': (laplace_matrix, example_laplace),
    'laplace2d': (laplace2d_matrix, example_laplace2d),
    'random':  (random_matrix, example_random),
}

def example_matrix(name, args):
    return _test_problems[name][0](*args)
def example_problem(name, args):
    return _test_problems[name][1](*args)

################################################################################
## reference implementations

def ichol_dense(A):
    A = A.toarray()
    # simple reference implementation for ichol based on dense matrices
    L = np.tril(A)
    n = A.shape[0]

    for k in range(n):
        L[k,k] = np.sqrt(L[k,k])
        for i in range(k+1, n):
            if L[i,k] != 0:
                L[i,k] /= L[k,k]
        for j in range(k+1, n):
            for i in range(j, n):
                if L[i,j] != 0:
                    L[i,j] -= L[i,k] * L[j,k]
    return L

def ilu0_dense(A):
    # sparse implementation depends on orientation, so we have to check it
    is_csc = isinstance(A, scipy.sparse.csc_matrix)
    if is_csc:
        A = A.T
    A = A.toarray()
    n = A.shape[0]
    for i in range(n):
        for k in range(i):
            if A[i,k] != 0:
                l_ik = A[i,k] / A[k,k]
                A[i,k] = l_ik
                for j in range(k+1, n):
                    if A[i,j] != 0:
                        A[i,j] -= l_ik * A[k,j]
    L, U = np.tril(A, -1), np.triu(A)
    np.fill_diagonal(L, 1.0)
    if is_csc:
        return (U.T, L.T)
    else:
        return (L, U)

def ilut_dense(A, fill_in=None, threshold=0.1):
    # fill_in parameter currently not supported and is ignored

    # sparse implementation depends on orientation, so we have to check it
    is_csc = isinstance(A, scipy.sparse.csc_matrix)
    if is_csc:
        A = A.T

    A = A.toarray()
    n = A.shape[0]
    for i in range(n):
        norm_Li = np.linalg.norm(A[i, :i])
        for k in range(i):
            if A[i,k] != 0:
                # do dropping on single element L[i,k]
                if abs(A[i,k]) < threshold * norm_Li:
                    A[i,k] = 0
                else:
                    l_ik = A[i,k] / A[k,k]
                    A[i,k] = l_ik
                    for j in range(k+1, n):
                        A[i,j] -= l_ik * A[k,j]

        # do dropping on i-th row
        L = A[i, :i]
        norm_L = np.linalg.norm(L)
        L[abs(L) < threshold*norm_L] = 0

        U = A[i, i+1:]
        norm_U = np.linalg.norm(U)
        U[abs(U) < threshold*norm_U] = 0

    L, U = np.tril(A, -1), np.triu(A)
    np.fill_diagonal(L, 1.0)
    if is_csc:
        return (U.T, L.T)
    else:
        return (L, U)

########################################
# auto-generated test cases

def _gen_solve_in_one_step(Precond, params, problem, example_args):
    def test(self):
        A, b, x_exact = example_problem(problem, example_args)
        P = Precond(A, **params)
        x = b.copy()
        P.apply(x)
        print('Error:', np.linalg.norm(x - x_exact))
        assert np.allclose(x, x_exact)
    return test

def _gen_test_with_predicate(Precond, params, problem, example_args, pred):
    def test(self):
        A, b, x_exact = example_problem(problem, example_args)
        P = Precond(A, **params)
        assert pred(A, P)
    return test

def _gen_test_factorization(func, args, sym, problem, matrix_args, reference_impl):
    def test(self):
        A = example_matrix(problem, matrix_args)
        if sym:
            A = (A + A.T) / 2       # symmetrize
        X = func(A, **args)
        X_ref = reference_impl(A, **args)
        if isinstance(X, tuple):
            assert all(np.allclose(Xi.A, Xi_ref) for (Xi,Xi_ref) in zip(X, X_ref))
        else:
            assert np.allclose(X.A, X_ref)
    return test

def is_lower_triangular(A):
    return all(i >= j for (i,j) in zip(*A.nonzero()))
def is_upper_triangular(A):
    return all(i <= j for (i,j) in zip(*A.nonzero()))

def _assert_factors_correct(A, P):
    L, U = P.factors()
    assert is_lower_triangular(L)
    assert is_upper_triangular(U)
    LU = L.dot(U)
    assert np.allclose(A.A, LU.A)
    return True

class TestCases(unittest.TestCase):
    # generate tests for preconditioner classes
    for P in [
            ilupp.ILUTPreconditioner,
            ilupp.ILUTPPreconditioner,
            ilupp.ILUCPreconditioner,
            ilupp.ILUCPPreconditioner,
    ]:
        base_name = 'test_' + P.__name__[:-14] + '_'

        for problem in ('laplace', 'random'):
            for format in ('csr', 'csc'):
                case_name = base_name + problem + '_' + format
                vars()[case_name] = _gen_solve_in_one_step(P, {'threshold': 0.0}, problem, (50,format))

                case_name = base_name + problem + '_factorscorrect_' + format
                vars()[case_name] = _gen_test_with_predicate(P, {'threshold': 0.0}, problem, (50,format), _assert_factors_correct)

        # each L/U factor has a diagonal and an off-diagonal,
        # but ILU++ reports less in some cases because of unit diagonals
        case_name = base_name + 'total_nnz'
        vars()[case_name] = _gen_test_with_predicate(P, {'threshold': 0.0}, 'laplace', (50,),
                lambda A, pr: pr.total_nnz <= 2 * (2*A.shape[0] - 1))

    # generate tests for stand-alone factorization functions
    for (F, ref, args, sym) in [
            # function to test, reference implementation, symmetric
            (ilupp.ichol, ichol_dense, {}, True),
            (ilupp.ilu0, ilu0_dense, {}, False),
            (ilupp.ilut, ilut_dense, {'fill_in': 10000, 'threshold': 0.1}, False),
    ]:
        base_name = 'test_' + F.__name__ + '_'

        for problem in ('laplace', 'laplace2d', 'random'):
            for format in ('csr', 'csc'):
                case_name = base_name + problem + '_' + format
                vars()[case_name] = _gen_test_factorization(F, args, sym, problem, (50,format), ref)

########################################

## multilevel preconditioner

def test_ml_solve_laplace2d_PQ():
    A, b, x_exact = example_laplace2d(900)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_PQ()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_MWM():
    A, b, x_exact = example_laplace2d(900)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_MWM_sPQ():
    A, b, x_exact = example_laplace2d(900)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ()
    param.threshold = 2.0
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_SF():
    A, b, x_exact = example_laplace2d(900)
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
