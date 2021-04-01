# ilupp -- ILU algorithms for C++ and Python
# Copyright (C) 2020 Clemens Hofreither
# ILU++ is Copyright (C) 2006 by Jan Mayer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""ilupp

ILU algorithms for C++ and Python
"""

__version__ = '1.0.2'

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from . import _ilupp
from ._ilupp import iluplusplus_precond_parameter, preprocessing_sequence

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


def solve(A, b, rtol=1e-4, atol=1e-4, max_iter=500, threshold=0.1, fill_in=None, params=None, info=False):
    """Solve the linear system Ax=b using a multilevel ILU++ preconditioner and BiCGStab.

    Args:
        A: a sparse matrix in CSR or CSC format
        b: the right-hand side vector
        rtol: target relative reduction in the residual
        atol: target absolute magnitude of the residual
        max_iter: maximum number of iterations
        threshold: the threshold parameter for ILU++; entries with relative
            magnitude less than this are dropped
        fill_in: the fill_in parameter for the ILU++ preconditioner
        params: an instance of :class:`iluplusplus_precond_parameter`; if passed, overrides fill_in and threshold
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
    """Wrapper base class which supports methods and properties common to all preconditioners.

    Implements the scipy.sparse.linalg.LinearOperator protocol, which means that it has a
    :code:`.shape` property and can be applied to a vector using :code:`.dot()` or simply the
    multiplication operator :code:`*`.

    To apply the preconditioner to a vector in place, avoiding a copy, use the
    :func:`apply` method.
    """
    def _matvec(self, x):
        if x.ndim != 1:
            raise ValueError('only implemented for 1D vectors')
        y = x.copy()
        self.pr.apply(y)
        return y

    def apply(self, x):
        """Apply the preconditioner to the vector `x` in-place."""
        if x.ndim != 1:
            raise ValueError('only implemented for 1D vectors')
        self.pr.apply(x)

    @property
    def total_nnz(self):
        """The total number of nonzeros stored in the factor matrices of the preconditioner."""
        return self.pr.total_nnz

    def factors(self):
        """Return all matrix factors (usually (L,U) or just (L,)) as a list of sparse matrices."""
        return [_matrix_from_info(*info) for info in self.pr.factors_info()]

    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with nnz=%d, %s>' % (M, N, self.__class__.__name__, self.total_nnz, dt)


class ILUppPreconditioner(_BaseWrapper):
    """A multilevel ILU++ preconditioner.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the fill_in parameter for the ILU++ preconditioner
        threshold: the threshold parameter for ILU++; entries with relative
            magnitude less than this are dropped
        params: an instance of :class:`iluplusplus_precond_parameter`; if passed, overrides fill_in and threshold
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
    """An ILUT (incomplete LU with thresholding) preconditioner.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the number of nonzeros to allow per row of L/U
        threshold: entries with relative magnitude less than this are dropped
    """
    def __init__(self, A, fill_in=100, threshold=0.1):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUTPreconditioner(Ad, Ai, Ap, Ao, fill_in, threshold)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

class ILUTPPreconditioner(_BaseWrapper):
    """An ILUTP (incomplete LU with thresholding and column pivoting) preconditioner.

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the number of nonzeros to allow per row of L/U
        threshold: entries with relative magnitude less than this are dropped
        piv_tol: pivoting tolerance; 0=only pivot when 0 encountered, 1=always pivot
            to the largest entry, inbetween: pivot depending on relative magnitude
    """
    def __init__(self, A, fill_in=100, threshold=0.1, piv_tol=0.1, mem_factor=10.0):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUTPPreconditioner(Ad, Ai, Ap, Ao,
                fill_in, threshold, piv_tol, -1, mem_factor)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

    def permutations(self):
        """Return a pair (L,R) of permutation arrays to be applied from the left or right due to pivoting."""
        return self.pr.permutations()

class ILUCPreconditioner(_BaseWrapper):
    """An ILUC (Crout ILU) preconditioner. Similar to ILUT, but tends to be faster
    for matrices with symmetric structure. See (Li, Saad, Chow 2003).

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the number of nonzeros to allow per column/row of L/U
        threshold: entries with relative magnitude less than this are dropped
    """
    def __init__(self, A, fill_in=100, threshold=0.1):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUCPreconditioner(Ad, Ai, Ap, Ao, fill_in, threshold)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

class ILUCPPreconditioner(_BaseWrapper):
    """An ILUCP (ILUC with pivoting) preconditioner. See (Mayer 2005).

    Args:
        A: a sparse matrix in CSR or CSC format
        fill_in: the number of nonzeros to allow per column/row of L/U
        threshold: entries with relative magnitude less than this are dropped
        piv_tol: pivoting tolerance; 0=only pivot when 0 encountered, 1=always pivot
            to the largest entry, inbetween: pivot depending on relative magnitude
    """
    def __init__(self, A, fill_in=100, threshold=0.1, piv_tol=0.1, mem_factor=10.0):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILUCPPreconditioner(Ad, Ai, Ap, Ao,
                fill_in, threshold, piv_tol, -1, mem_factor)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

    def permutations(self):
        """Return a pair (L,R) of permutation arrays to be applied from the left or right due to pivoting."""
        return self.pr.permutations()

class ILU0Preconditioner(_BaseWrapper):
    """An ILU(0) preconditioner (no fill-in, same sparsity pattern as A).

    Args:
        A: a sparse matrix in CSR or CSC format
    """
    def __init__(self, A):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ILU0Preconditioner(Ad, Ai, Ap, Ao)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

class IChol0Preconditioner(_BaseWrapper):
    """An IChol(0) preconditioner (no fill-in, same sparsity pattern as A) for a
    symmetric positive definite matrix.

    Args:
        A: a symmetric sparse matrix in CSR or CSC format
    """
    def __init__(self, A):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.IChol0Preconditioner(Ad, Ai, Ap, Ao)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

class ICholTPreconditioner(_BaseWrapper):
    """An incomplete Cholesky preconditioner with user-specifiable additional fill-in and threshold.
    With threshold=0, this is identical to the method described in (Lin, Moré 1999).

    Args:
        A: a symmetric sparse matrix in CSR or CSC format
        add_fill_in: the number of additional nonzeros to allow per column. By default (0),
            the factorization keeps the number (but not necessarily the positions) of the nonzeros
            identical to the original matrix.
        threshold: entries with a relative magnitude less than this are dropped. By default (0.0),
            dropping is only performed based on the number of nonzeros.
    """
    def __init__(self, A, add_fill_in=0, threshold=0.0):
        Ad, Ai, Ap, Ao = _matrix_fields(A)
        self.pr = _ilupp.ICholTPreconditioner(Ad, Ai, Ap, Ao, add_fill_in, threshold)
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype)

########################################

def ichol0(A):
    """Compute the L factor of an incomplete Cholesky decomposition without fill-in for the symmetric matrix A."""
    return _matrix_from_info(*_ilupp.ichol0(*_matrix_fields(A)))

def icholt(A, add_fill_in=0, threshold=0.0):
    """Compute the L factor of an incomplete Cholesky decomposition with thresholding for the symmetric matrix A."""
    return _matrix_from_info(*_ilupp.icholt(*_matrix_fields(A), add_fill_in, threshold))

def ilu0(A):
    """Compute the (L,U) factors of an incomplete LU decomposition without fill-in."""
    return tuple(_matrix_from_info(*mtx) for mtx in _ilupp.ilu0(*_matrix_fields(A)))

def ilut(A, fill_in=100, threshold=0.1):
    """Compute the (L,U) factors of an incomplete LU decomposition with thresholding."""
    return tuple(_matrix_from_info(*mtx) for mtx in _ilupp.ilut(*_matrix_fields(A), fill_in, threshold))

def iluc(A, fill_in=100, threshold=0.1):
    """Compute the (L,U) factors of an incomplete Crout LU decomposition with thresholding."""
    return tuple(_matrix_from_info(*mtx) for mtx in _ilupp.iluc(*_matrix_fields(A), fill_in, threshold))
