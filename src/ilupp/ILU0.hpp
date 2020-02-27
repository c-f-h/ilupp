#pragma once

#include "declarations.h"

namespace iluplusplus {


template <class T>
void sparse_vec_update(Integer l1, Integer u1, Integer l2, Integer u2, Integer k, T l_ik, const Integer* indices, T* data_U)
{
    while (l1 < u1 && l2 < u2) {
        if (indices[l1] == indices[l2]) {   // matching column?
            if (indices[l1] > k)            // above the diagonal?
                data_U[l1] -= l_ik * data_U[l2];
            l1++;
            l2++;
        }
        else if (indices[l1] < indices[l2])       // else proceed until we find matching columns
            l1++;
        else
            l2++;
    }
}

// data_U should have size A.actual_non_zeroes() (same as A.data)
template <class T>
void compute_ilu0(const matrix_sparse<T>& A, T* data_U, Integer& nnzL, Integer& nnzU)
{
    const Integer n = A.dim_major();
    std::vector<T> diag_U(n);
    Integer diag_idx = 0;
    nnzL = nnzU = 0;

    for (Integer i = 0; i < n; ++i) {
        // initialize U[i,:] = A[i,:]
        for (Integer kk = A.pointer[i]; kk < A.pointer[i+1]; ++kk) {
            data_U[kk] = A.data[kk];

            if (A.indices[kk] == i)
                diag_idx = kk;

            // count nnz needed for L/U parts
            if (A.indices[kk] <= i) ++nnzL;
            if (A.indices[kk] >= i) ++nnzU;
        }

        for (Integer kk = A.pointer[i]; kk < A.pointer[i+1]; ++kk) {
            const Integer k = A.indices[kk];     // column
            if (k >= i)
                continue;

            const T L_ik = data_U[kk] / diag_U[k];

            // U[i, k+1:] -= L_ik * U[k, k+1:]
            sparse_vec_update(
                    A.pointer[i], A.pointer[i+1],       // i-th row
                    A.pointer[k], A.pointer[k+1],       // k-th row
                    k, L_ik,
                    A.indices, data_U);

            data_U[kk] = L_ik;
        }
        // store diagonal for later iterations
        diag_U[i] = data_U[diag_idx];
    }
}

// compute ILU factorization without fill-in (retains structure of A)
template <class T>
void ILU0(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U)
{
    std::vector<T> LU_data(A.actual_non_zeroes());
    Integer nnzL, nnzU;

    // compute the data (same structure as A)
    compute_ilu0(A, &LU_data[0], nnzL, nnzU);

    // split the single matrix into its lower and upper components
    L.reformat(A.dim_major(), A.dim_minor(), nnzL, ROW);
    U.reformat(A.dim_major(), A.dim_minor(), nnzU, ROW);

    Integer *iL = &L.indices[0], *iU = &U.indices[0];
    T *dL = &L.data[0], *dU = &U.data[0];

    for (Integer i = 0; i < A.dim_major(); ++i) {
        for (Integer k = A.pointer[i]; k < A.pointer[i+1]; ++k) {
            const Integer j = A.indices[k];
            if (j < i) { *iL++ = j; *dL++ = LU_data[k]; }
            else       { *iU++ = j; *dU++ = LU_data[k]; }
        }

        // set diagonal of L to 1
        *iL++ = i; *dL++ = static_cast<T>(1.0);

        // finished i-th row - set pointer to start of next row
        L.pointer[i + 1] = static_cast<Integer>(iL - &L.indices[0]);
        U.pointer[i + 1] = static_cast<Integer>(iU - &U.indices[0]);
    }

    if (A.orientation == COLUMN) {
        // compute_ilu0 assumes CSR, so we actually decomposed A^T
        L.interchange(U);
        L.transpose_in_place();
        U.transpose_in_place();
    }
}

} // end namespace iluplusplus
