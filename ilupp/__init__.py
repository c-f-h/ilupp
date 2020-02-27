"""ilupp

Python bindings for ILU++
"""

__version__ = '0.0.1'

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from . import _ilupp
from ._ilupp import iluplusplus_precond_parameter

def _matrix_fields(A):
    if isinstance(A, scipy.sparse.csr_matrix):
        is_csr = True
    elif isinstance(A, scipy.sparse.csc_matrix):
        is_csr = False
    else:
        raise TypeError("A must be a csr_matrix or a csc_matrix")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix!")

    A.sort_indices()    # most ILU algorithms require the indices to be ascending
    return A.data, A.indices, A.indptr, is_csr

def _matrix_from_info(data, indices, indptr, is_csr, rows, cols):
    if is_csr:
        A = scipy.sparse.csr_matrix((data, indices, indptr), shape=(rows, cols), copy=False)
    else:
        A = scipy.sparse.csc_matrix((data, indices, indptr), shape=(rows, cols), copy=False)
    A.has_sorted_indices = True     # all ILU++ algorithms output sorted indices
    return A


def solve(A, b, rtol=1e-4, atol=1e-4, max_iter=500, threshold=1.0, fill_in=None, params=None, info=False):
    """Solve the linear system Ax=b using a multilevel ILU++ preconditioner and BiCGStab.

    Args:
        A: a sparse matrix in CSR or CSC format
        b: the right-hand side vector
        rtol: target relative reduction in the residual
        atol: target absolute magnitude of the residual
        max_iter: maximum number of iterations
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
        params: an instance of iluplusplus_precond_parameter; if passed, overrides fill_in and threshold
        info: if True, a tuple (nr_of_iterations, achieved_relative_reduction, residual_magnitude) is returned
            along the solution

    Returns:
        a vector containing the solution x
    """
    if params is None:
        params = iluplusplus_precond_parameter()
        params.threshold = threshold
        if fill_in is not None:
            params.fill_in = fill_in

    Ad, Ai, Ap, Ao = _matrix_fields(A)
    b = np.ascontiguousarray(b, dtype=np.float_)

    sol, iter, rtol_out, atol_out = _ilupp.solve(
            Ad, Ai, Ap, Ao, b, rtol, atol, max_iter, params)

    if info:
        return sol, (iter, rtol_out, atol_out)
    else:
        return sol


class _BaseWrapper(scipy.sparse.linalg.LinearOperator):
    """Wrapper base class which supports methods and properties common to all preconditioners."""
    def _matvec(self, x):
        if x.ndim != 1:
            raise ValueError('only implemented for 1D vectors')
        y = x.copy()
        self.pr.apply(y)
        return y

    def apply(self, x):
        """Apply the preconditioner to `x` in-place."""
        if x.ndim != 1:
            raise ValueError('only implemented for 1D vectors')
        self.pr.apply(x)

    @property
    def total_nnz(self):
        return self.pr.total_nnz

    def factors(self):
        """Return all matrix factors (usually (L,U)) as a list of sparse matrices."""
        return [_matrix_from_info(*info) for info in self.pr.factors_info()]


class ILUppPreconditioner(_BaseWrapper):
    """A multilevel ILU++ preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
        params: an instance of iluplusplus_precond_parameter; if passed, overrides fill_in and threshold
    """
    def __init__(self, A, threshold=1.0, fill_in=None, params=None):
        if params is None:
            params = iluplusplus_precond_parameter()
            params.threshold = threshold
            if fill_in is not None:
                params.fill_in = fill_in

        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.MultilevelILUCDPPreconditioner(Ad, Ai, Ap, Ao, params)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

    @property
    def memory(self):
        return self.pr.memory

    @property
    def memory_used_calculations(self):
        return self.pr.memory_used_calculations

    @property
    def memory_allocated_calculations(self):
        return self.pr.memory_allocated_calculations


class ILUTPreconditioner(_BaseWrapper):
    """An ILUT preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
    """
    def __init__(self, A, threshold=1.0, fill_in=10000):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUTPreconditioner(Ad, Ai, Ap, Ao, fill_in, threshold)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

class ILUTPPreconditioner(_BaseWrapper):
    """An ILUTP preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
    """
    def __init__(self, A, threshold=1.0, fill_in=10000, perm_tol=0.0, row_pos=-1, mem_factor=10.0):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUTPPreconditioner(Ad, Ai, Ap, Ao,
                fill_in, threshold, perm_tol, row_pos, mem_factor)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

    @property
    def permutation(self):
        return self.pr.permutation

class ILUCPreconditioner(_BaseWrapper):
    """An ILUC (Crout ILU) preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
    """
    def __init__(self, A, threshold=1.0, fill_in=10000):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUCPreconditioner(Ad, Ai, Ap, Ao, fill_in, threshold)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

class ILUCPPreconditioner(_BaseWrapper):
    """An ILUCP preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
    """
    def __init__(self, A, threshold=1.0, fill_in=10000, perm_tol=0.0, row_pos=-1, mem_factor=10.0):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUCPPreconditioner(Ad, Ai, Ap, Ao,
                fill_in, threshold, perm_tol, row_pos, mem_factor)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

    @property
    def permutation(self):
        return self.pr.permutation

########################################

def ichol(A):
    """Compute the L factor of an incomplete Cholesky decomposition without fill-in for the symmetric matrix A."""
    return _matrix_from_info(*_ilupp.ichol(*_matrix_fields(A)))
