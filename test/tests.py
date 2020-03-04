import numpy as np
import scipy.sparse
import ilupp
import unittest

################################################################################
## test problems

def laplace_matrix(n, format='csr'):
    h = 1.0 / (n + 1)
    d = np.ones(n) / (h**2)
    return scipy.sparse.diags((-d[:-1], 2*d, -d[:-1]), (-1, 0, 1)).asformat(format)

def laplace2d_matrix(n_total, format='csr'):
    n = int(np.sqrt(n_total))
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

def random_matrix(n, format='csr', eye_factor=10.0):
    avg_nnz_per_row = 5
    density = min(1.0, avg_nnz_per_row / n)
    return (scipy.sparse.random(n, n, density=density, random_state=39273) + eye_factor*scipy.sparse.eye(n)).asformat(format)

def example_random(n, format='csc', eye_factor=10.0):
    A = random_matrix(n, format=format, eye_factor=eye_factor)
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

def do_dropping(X, max_entries=None, threshold=None):
    if threshold is not None:
        nrm = np.linalg.norm(X)
        X[abs(X) < threshold*nrm] = 0
    if max_entries is not None:
        indices = np.argsort(abs(X))    # largest last
        max_entries = min(max_entries, X.shape[0])
        X[indices[:-max_entries]] = 0

def ichol0_dense(A):
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

def icholt_dense(A, add_fill_in=0, threshold=0.0):
    n = A.shape[0]
    A = A.toarray()
    L = np.zeros_like(A)
    D = np.zeros(n)
    for j in range(n):
        w = A[:,j].copy()
        w[:j] = 0
        D[j] += w[j]
        w[j] = np.sqrt(D[j])
        col_len = np.count_nonzero(w[j:])
        for k in range(j):
            if L[j,k] != 0:
                for i in range(j+1, n):
                    if L[i,k] != 0:
                        w[i] -= L[i,k] * L[j,k]
        for i in range(j+1, n):
            if w[i] != 0:
                w[i] /= w[j]
                D[i] -= w[i]**2
        do_dropping(w, max_entries=col_len+add_fill_in, threshold=threshold)
        L[:,j] = w
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
    if fill_in is not None:
        fill_in -= 1        # match the ILU++ convention
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
        do_dropping(A[i, :i], max_entries=fill_in, threshold=threshold)
        do_dropping(A[i, i+1:], max_entries=fill_in, threshold=threshold)

    L, U = np.tril(A, -1), np.triu(A)
    np.fill_diagonal(L, 1.0)
    if is_csc:
        return (U.T, L.T)
    else:
        return (L, U)

def iluc_dense(A, fill_in=1000, threshold=0.1):
    # see (Li, Saad, Chow 2002), Crout versions of ILU for general sparse matrices
    is_csc = isinstance(A, scipy.sparse.csc_matrix)
    if is_csc:
        A = A.T
    A = A.toarray()
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for k in range(n):
        z = A[k, :].copy()     # get upper row into z
        z[:k] = 0
        for i in range(k):
            if L[k,i] != 0:
                z[k:] -= L[k,i] * U[i,k:]
        w = A[:, k].copy()     # get lower column into w
        w[:k+1] = 0
        for i in range(k):
            if U[i,k] != 0:
                w[k+1:] -= U[i,k] * L[k+1:,i]
        do_dropping(z[k+1:], max_entries=fill_in-1, threshold=threshold)
        do_dropping(w[k+1:], max_entries=fill_in-1, threshold=threshold)
        U[k,:] = z
        L[:,k] = w / U[k,k]
        L[k,k] = 1
    if is_csc:
        return U.T, L.T
    else:
        return L, U

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
    factors = P.factors()
    if len(factors) == 2:
        # LU case
        L, U = factors
        if not hasattr(P, 'permutation'):
            # for preconditioners with pivoting, we only have
            # that L[perm,:] or U[:,perm] is triangular, however
            # A = L.U is still correct
            assert is_lower_triangular(L)
            assert is_upper_triangular(U)
        LU = L.dot(U)
        assert np.allclose(A.A, LU.A)
        return True
    else:
        # LLT case
        L = factors[0]
        LLT = L.dot(L.T)
        assert np.allclose(A.A, LLT.A)
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

    # generate tests for zero fill-in preconditioner classes
    for P in [
            ilupp.IChol0Preconditioner,
            ilupp.ICholTPreconditioner,
    ]:
        base_name = 'test_' + P.__name__[:-14] + '_'

        for problem in ('laplace',):
            for format in ('csr', 'csc'):
                case_name = base_name + problem + '_' + format
                vars()[case_name] = _gen_solve_in_one_step(P, {}, problem, (50,format))

                case_name = base_name + problem + '_factorscorrect_' + format
                vars()[case_name] = _gen_test_with_predicate(P, {}, problem, (50,format), _assert_factors_correct)

        case_name = base_name + 'total_nnz'
        vars()[case_name] = _gen_test_with_predicate(P, {}, 'laplace', (50,),
                lambda A, pr: pr.total_nnz == (2*A.shape[0] - 1))   # one diagonal, one off-diagonal

    # test the pivoting preconditioners on pseudo-random matrices without diagonal dominance
    for P in [
            ilupp.ILUTPPreconditioner,
            ilupp.ILUCPPreconditioner,
    ]:
        base_name = 'test_' + P.__name__[:-14] + '_pivot_'

        for problem in ('random',):
            for format in ('csr', 'csc'):
                case_name = base_name + problem + '_' + format
                # we set eye_factor=0 so that the matrix has some 0 diagonal entries
                vars()[case_name] = _gen_solve_in_one_step(P, {'threshold': 0.0, 'piv_tol':1}, problem, (50,format,0.0))

                case_name = base_name + problem + '_factorscorrect_' + format
                vars()[case_name] = _gen_test_with_predicate(P, {'threshold': 0.0, 'piv_tol':0.5}, problem, (50,format,0.0), _assert_factors_correct)

    # generate tests for stand-alone factorization functions
    for (F, ref, args, sym) in [
            # function to test, reference implementation, symmetric
            (ilupp.ichol0, ichol0_dense, {}, True),
            (ilupp.icholt, icholt_dense, {}, True),
            (ilupp.ilu0, ilu0_dense, {}, False),
            (ilupp.ilut, ilut_dense, {'fill_in': 5, 'threshold': 0.1}, False),
            (ilupp.iluc, iluc_dense, {'fill_in': 5, 'threshold': 0.1}, False),
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
    param.threshold = 1e-2
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_MWM():
    A, b, x_exact = example_laplace2d(900)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING()
    param.threshold = 1e-2
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_MWM_sPQ():
    A, b, x_exact = example_laplace2d(900)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ()
    param.threshold = 1e-2
    x, info = ilupp.solve(A, b, atol=1e-8, rtol=1e-8, params=param, info=True)
    print('Convergence info:', info)
    assert np.allclose(x_exact, x)

def test_ml_solve_laplace2d_SF():
    A, b, x_exact = example_laplace2d(900)
    param = ilupp.iluplusplus_precond_parameter()
    param.PREPROCESSING.set_SPARSE_FIRST()
    param.threshold = 1e-2
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
    P = ilupp.ILUppPreconditioner(A, threshold=0)
    x = P.dot(b)
    X = np.linspace(0, 1, n+2)[1:-1]
    assert np.allclose(x, X*(1-X)/2)
    print('total nnz:', P.total_nnz, 'memory:', P.memory)

def test_ml_precond_random():
    A, b, x_exact = example_random(50)
    P = ilupp.ILUppPreconditioner(A, threshold=0)
    x = b.copy()
    P.apply(x)
    print('Error:', np.linalg.norm(x - x_exact))
    assert np.allclose(x, x_exact)
