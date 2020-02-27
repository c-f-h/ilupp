#pragma once

#include "declarations.h"
#include <cmath>

namespace iluplusplus {

/*
   ichol code is loosely based on
   https://github.com/f-t-s/spelliptic/blob/master/Factorization.jl
   (Florian Schaefer, 2018, MIT licensed)
*/

template <class T>
T sparse_dot_product(Integer l1, Integer u1, Integer l2, Integer u2, const Integer* indices, const T* data)
{
    T result = 0.0;
    while (l1 < u1 && l2 < u2) {
        if (indices[l1] == indices[l2])     // matching column?
            result += data[l1++] * data[l2++];
        else if (indices[l1] < indices[l2])       // else proceed until we find matching columns
            l1++;
        else
            l2++;
    }
    return result;
}

// input A should be in major triangular form (j <= i), the diagonal must have no zeros
// returns an array of new data representing the L factor (same sparsity structure as A)
template <class T>
std::vector<T> compute_ichol(const matrix_sparse<T>& A)
{
    std::vector<T> new_data(A.actual_non_zeroes());

    const Integer n = A.get_pointer_size() - 1;
    for (Integer i = 0; i < n; ++i) {
        for (Integer k = A.pointer[i]; k < A.pointer[i+1]; ++k) {
            const Integer j = A.indices[k];     // column

            const T dp = sparse_dot_product(
                    A.pointer[i], A.pointer[i+1] - 1,       // i-th row minus diagonal
                    A.pointer[j], A.pointer[j+1] - 1,       // j-th row minus diagonal
                    A.indices, &new_data[0]);

            const T A_ij = A.data[k];

            if (j < i) {        // below diagonal?
                const T L_jj = new_data[A.pointer[j+1] - 1];    // diagonal is last entry of j-th row
                new_data[k] = (A_ij - dp) / L_jj;
            } else if (j == i)    // on the diagonal?
                new_data[k] = std::sqrt(A_ij - dp);
            else                // above diagonal -- input should be triangular!
                throw std::logic_error("Matrix passed in the wrong format - should be triangular");
        }
    }
    return new_data;
}

// compute the L factor in a L.L^T incomplete Cholesky decomposition
template <class T>
matrix_sparse<T> IChol(const matrix_sparse<T>& A)
{
    matrix_sparse<T> L = A.natural_triangular_part();
    // ensure we have the lower triangular part
    if (L.orientation == COLUMN)
        L.transpose_in_place();

    std::vector<T> new_data = compute_ichol(L);
    std::copy(new_data.begin(), new_data.end(), &L.data[0]);
    return L;
}

} // end namespace iluplusplus
