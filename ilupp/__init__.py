"""ilupp

Python bindings for ILU++
"""

__version__ = '0.0.1'

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from . import _ilupp
from ._ilupp import iluplusplus_precond_parameter

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

    if isinstance(A, scipy.sparse.csr_matrix):
        is_csr = True
    elif isinstance(A, scipy.sparse.csc_matrix):
        is_csr = False
    else:
        raise TypeError("A must be a csr_matrix or a csc_matrix")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix!")

    b = np.ascontiguousarray(b, dtype=np.float_)

    sol, iter, rtol_out, atol_out = _ilupp.solve(
            A.data, A.indices, A.indptr, is_csr, b, rtol, atol, max_iter, params)

    if info:
        return sol, (iter, rtol_out, atol_out)
    else:
        return sol


class ILUppPreconditioner(scipy.sparse.linalg.LinearOperator):
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

        if isinstance(A, scipy.sparse.csr_matrix):
            is_csr = True
        elif isinstance(A, scipy.sparse.csc_matrix):
            is_csr = False
        else:
            raise TypeError("A must be a csr_matrix or a csc_matrix")

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix!")

        self.pr = _ilupp.multilevel_preconditioner()
        self.pr.setup(A.data, A.indices, A.indptr, is_csr, params)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

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

    @property
    def memory(self):
        return self.pr.memory

    @property
    def memory_used_calculations(self):
        return self.pr.memory_used_calculations

    @property
    def memory_allocated_calculations(self):
        return self.pr.memory_allocated_calculations


class ILUTPreconditioner(scipy.sparse.linalg.LinearOperator):
    """An ILUT preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
    """
    def __init__(self, A, threshold=1.0, fill_in=10000):
        if isinstance(A, scipy.sparse.csr_matrix):
            is_csr = True
        elif isinstance(A, scipy.sparse.csc_matrix):
            is_csr = False
        else:
            raise TypeError("A must be a csr_matrix or a csc_matrix")

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix!")

        self.pr = _ilupp.ILUTPreconditioner(A.data, A.indices, A.indptr, is_csr, fill_in, threshold)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

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

class ILUTPPreconditioner(scipy.sparse.linalg.LinearOperator):
    """An ILUTP preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
    """
    def __init__(self, A, threshold=1.0, fill_in=10000, perm_tol=0.0, row_pos=-1, mem_factor=20.0):
        if isinstance(A, scipy.sparse.csr_matrix):
            is_csr = True
        elif isinstance(A, scipy.sparse.csc_matrix):
            is_csr = False
        else:
            raise TypeError("A must be a csr_matrix or a csc_matrix")

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix!")

        self.pr = _ilupp.ILUTPPreconditioner(A.data, A.indices, A.indptr, is_csr,
                fill_in, threshold, perm_tol, row_pos, mem_factor)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

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

class ILUCPreconditioner(scipy.sparse.linalg.LinearOperator):
    """An ILUC (Crout ILU) preconditioner. Implements the scipy LinearOperator protocol.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries smaller than 10^-threshold are dropped
    """
    def __init__(self, A, threshold=1.0, fill_in=10000):
        if isinstance(A, scipy.sparse.csr_matrix):
            is_csr = True
        elif isinstance(A, scipy.sparse.csc_matrix):
            is_csr = False
        else:
            raise TypeError("A must be a csr_matrix or a csc_matrix")

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix!")

        self.pr = _ilupp.ILUCPreconditioner(A.data, A.indices, A.indptr, is_csr, fill_in, threshold)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

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
