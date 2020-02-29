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
    matrix_sparse<T> L = A.natural_triangular_part(true);
    // ensure we have the lower triangular part
    if (L.orientation == COLUMN)
        L.transpose_in_place();

    std::vector<T> new_data = compute_ichol(L);
    std::copy(new_data.begin(), new_data.end(), &L.data[0]);
    return L;
}


// ICholT, assumes A is already in CSC lower triangular form
template <class T>
matrix_sparse<T> ICholT_tri(const matrix_sparse<T>& A, Integer add_fill_in, Real threshold, Real mem_factor=10.0)
{
    const Integer m = A.dim_major();
    if (A.dim_minor() != m)
        throw std::runtime_error("ICholT: A must be square");

    // estimate maximal size needed for L
    Integer reserved_memory = std::min(
            A.actual_non_zeroes() + std::max(add_fill_in, 0) * m,   // can add at most add_fill_in entries per column
            (Integer)(A.actual_non_zeroes() * mem_factor));         // safety in case fillin is large but threshold is large too

    // firstL[j]: index of the first element in the j-th column having a row index >= j
    // listL[j]: start of linked list (listL[j], listL[listL[j]], ...) of nonzero columns in row j
    std::vector<Integer> firstL(m), listL(m);
    initialize_triangular_fields(m, listL);

    // initialize L to the proper size
    matrix_sparse<T> L(A.orientation, m, m, reserved_memory);
    vector_sparse_dynamic<T> w(m);

    std::vector<T> D(m, 0.0);       // storage for the diagonal of L
    index_list listw;               // storage for selected entries to keep

    // refer to icholt_dense in tests.py for the algorithm

    // Iterate over the columns of A
    for (Integer j = 0; j < m; ++j) {
        if (A.indices[A.pointer[j]] != j) {
            throw std::logic_error("ICholT: A must be in triangular form with no zeros on the diagonal");
        }

        // initialize w to lower part of A (A only contains the lower part)
        w.zero_reset();
        for (Integer x = A.pointer[j]; x < A.pointer[j+1]; ++x)
            w[A.indices[x]] = A.data[x];

        // compute the diagonal L[j,j]
        D[j] += A.data[A.pointer[j]];       // add A_jj
        const T L_jj = std::sqrt(D[j]);
        w[j] = L_jj;

        // for all nonzero L[j,*]:
        for (Integer k = listL[j]; k != -1; k = listL[k]) {
            // L[j,k] is nonzero and has the following value:
            const T L_jk = L.data[firstL[k]];

            // iterate over i in range(j+1, n) with L[i,k] != 0
            // firstL[k] points to first nonzero L[h,k] with h >= j, so advance by one if h==j
            Integer x = firstL[k];
            if (L.indices[x] == j)
                ++x;
            for (; x < L.pointer[k+1]; ++x) {
                w[L.indices[x]] -= L.data[x] * L_jk;
            }
        }

        // iterate over all nonzeros in w below the diagonal
        for (Integer x = 0; x < w.non_zeroes(); ++x) {
            const Integer i = w.get_pointer(x);
            if (i > j) {
                w.get_data(x) /= L_jj;
                D[i] -= sqr(w.get_data(x));     // D[i] -= w[i]^2
            }
        }

        // apply dropping to w
        const Integer col_len = A.pointer[j+1] - A.pointer[j];  // nnz in the original column A[:,j]
        w.take_largest_elements_by_abs_value_with_threshold(listw, col_len + add_fill_in, threshold, j, m);   // TODO: limiting to [j,m) is unnecessary

        // copy w into L[:,j]

        // ensure we reserved enough space
        if (L.pointer[j] + listw.dimension() > reserved_memory) {
            throw std::runtime_error("ICholT: memory reserved was insufficient. Increase mem_factor!");
        }

        for(Integer x = 0; x < listw.dimension(); ++x) {
            L.data   [L.pointer[j]+x] = w[listw[x]];
            L.indices[L.pointer[j]+x] = listw[x];
        }
        L.pointer[j+1] = L.pointer[j] + listw.dimension();

        update_triangular_fields(j, L.pointer, L.indices, listL, firstL);
    }

    L.compress(-1.0);
    return L;
}

template <class T>
matrix_sparse<T> ICholT(const matrix_sparse<T>& A, Integer add_fill_in=0, Real threshold=0.0, Real mem_factor=10.0)
{
    matrix_sparse<T> A_tri = A.natural_triangular_part(false);
    if (A.orientation == ROW)
        A_tri.transpose_in_place();
    return ICholT_tri(A_tri, add_fill_in, threshold, mem_factor);
}

} // end namespace iluplusplus
