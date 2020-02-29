#pragma once

#include "declarations.h"

namespace iluplusplus {

// calculates the (incomplete) LU factorization;
// returns false in case of break-down, true in case of success.
// although this algorithm works for A being both a ROW and COLUMN matrix,
// if A is a COLUMN matrix, L and U will be reversed, even though L still has 1's on the diagonal.
// In this case, it still works as a preconditioner

template<class T>
void ILUT(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, Integer max_fill_in, Real threshold, Real& time_self)
{
    const clock_t time_begin=clock();
    // the notation will be for A being a ROW matrix, i.e. U also a ROW matrix and L a ROW matrix.
    if (non_fatal_error(!A.square_check(), "ILUT: argument matrix must be square"))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    const Integer m = A.rows(), n = A.columns();

    vector_sparse_dynamic<T> w(m);
    index_list list_L, list_U;

    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;

    Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (n-max_fill_in)*max_fill_in;

    U.reformat(m, m, reserved_memory, ROW);
    L.reformat(m, m, reserved_memory, ROW);

    // (1.) begin for i -- loop over rows
    for (Integer i = 0; i < n; ++i) {
        // (2.) initialize w
        Real norm_wL = 0.0;     // norm of the L part of w
        for (Integer k = A.pointer[i]; k < A.pointer[i+1]; ++k) {
            w[A.indices[k]] = A.data[k];

            // take the norm only of the L part to be consistent with the dropping rule applied in (10.)
            if (A.indices[k] < i)
                norm_wL += absvalue_squared(A.data[k]);
        }     // end for k
        // (3.) begin for k
        norm_wL = std::sqrt(norm_wL);

        // for all k < i with w(k) != 0:
        for (Integer k = 0; k < i; ++k) {
            if (w.non_zero_check(k)) {
                // (5.) Apply dropping to w.
                // NB: we do this BEFORE dividing by the diagonal, otherwise the scaling is wrong
                if (std::abs(w[k]) < threshold*norm_wL) {
                    w.zero_set(k);
                } else {
                    // (4.) w[k] = w[k] / U[k,k]
                    // NB: the first entry of the k-th row of U is the diagonal entry
                    const T wk = (w[k] /= U.data[U.pointer[k]]);

                    // (6./7./8.) w = w - w[k] * u[k,*] (w[k] scalar, u[k,*] a row of U)
                    for (Integer j = U.pointer[k] + 1; j < U.pointer[k+1]; j++)
                        w[U.indices[j]] -= wk * U.data[j];
                }
            } // end if
        }  // (9.) end for k

        // (10.) Do dropping in w.
        // keep one space free for the diagonal; begin with 0 and go upto the diagonal, but not including it.
        w.take_largest_elements_by_abs_value_with_threshold(list_L, max_fill_in-1, threshold, 0, i);
        // keep one space free for the diagonal; begin with the element after the diagonal and go to the end.
        w.take_largest_elements_by_abs_value_with_threshold(list_U, max_fill_in-1, threshold, i+1, n);

        // (11.) Copy values to L:
        for (Integer j = 0; j < list_L.dimension(); ++j) {
            L.data[L.pointer[i]+j] = w[list_L[j]];
            L.indices[L.pointer[i]+j] = list_L[j];
        }
        L.data[L.pointer[i]+list_L.dimension()] = 1.0;      // diagonal element
        L.indices[L.pointer[i]+list_L.dimension()] = i;     // diagonal element
        L.pointer[i+1] = L.pointer[i]+list_L.dimension()+1;

        // (12.) Copy values to U:
        U.data[U.pointer[i]] = w[i]; // diagonal element
        U.indices[U.pointer[i]] = i; // diagonal element
        for (Integer j = 0; j < list_U.dimension(); ++j) {
            U.data[U.pointer[i]+j+1] = w[list_U[j]];
            U.indices[U.pointer[i]+j+1] = list_U[j];
        }
        U.pointer[i+1] = U.pointer[i]+list_U.dimension()+1;

        if (U.data[U.pointer[i]] == 0)
            throw std::runtime_error("matrix_sparse::ILUT: encountered zero pivot in row " + std::to_string(i));

        // (13.) w:=0
        w.zero_reset();
    }  // (14.) end for i
    L.compress();
    U.compress();
    time_self = ((Real)clock() - (Real)time_begin) / (Real)CLOCKS_PER_SEC;
}

// Like ILUT(), but uses a linear search through the w-nonzeros to find k rather than a linear search through [0,i).
// Should be better for large/very sparse matries, worse for smaller/more dense matrices?
template<class T>
void ILUT_wsearch(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, Integer max_fill_in, Real threshold, Real& time_self)
{
    const clock_t time_begin=clock();
    // the notation will be for A being a ROW matrix, i.e. U also a ROW matrix and L a ROW matrix.
    if (non_fatal_error(!A.square_check(), "ILUT: argument matrix must be square"))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    const Integer m = A.rows(), n = A.columns();

    vector_sparse_dynamic<T> w(m);
    index_list list_L, list_U;

    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;

    Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (n-max_fill_in)*max_fill_in;

    U.reformat(m, m, reserved_memory, ROW);
    L.reformat(m, m, reserved_memory, ROW);

    // (1.) begin for i -- loop over rows
    for (Integer i = 0; i < n; ++i) {
        // (2.) initialize w
        Real norm_wL = 0.0;     // norm of the L part of w
        for (Integer k = A.pointer[i]; k < A.pointer[i+1]; ++k) {
            w[A.indices[k]] = A.data[k];

            // take the norm only of the L part to be consistent with the dropping rule applied in (10.)
            if (A.indices[k] < i)
                norm_wL += absvalue_squared(A.data[k]);
        }     // end for k
        // (3.) begin for k
        norm_wL = std::sqrt(norm_wL);

        // for all k < i with w(k) != 0 -- use linear search through the nonzeros in w
        Integer x = w.find_next_index(-1);
        while (x >= 0) {
            const Integer k = w.get_pointer(x);
            if (k >= i)     // stay below the diagonal
                break;
            T& wk = w.get_data(x);

            // (5.) Apply dropping to w.
            // NB: we do this BEFORE dividing by the diagonal, otherwise the scaling is wrong
            if (std::abs(wk) < threshold*norm_wL) {
                w.zero_set(k);
            } else {
                // (4.) w[k] = w[k] / U[k,k]
                // NB: the first entry of the k-th row of U is the diagonal entry
                wk /= U.data[U.pointer[k]];

                // (6./7./8.) w = w - w[k] * u[k,*] (w[k] scalar, u[k,*] a row of U)
                for (Integer j = U.pointer[k] + 1; j < U.pointer[k+1]; j++)
                    w[U.indices[j]] -= wk * U.data[j];
            }
            x = w.find_next_index(k);
        }  // (9.) end for k

        // (10.) Do dropping in w.
        // keep one space free for the diagonal; begin with 0 and go upto the diagonal, but not including it.
        w.take_largest_elements_by_abs_value_with_threshold(list_L, max_fill_in-1, threshold, 0, i);
        // keep one space free for the diagonal; begin with the element after the diagonal and go to the end.
        w.take_largest_elements_by_abs_value_with_threshold(list_U, max_fill_in-1, threshold, i+1, n);

        // (11.) Copy values to L:
        for (Integer j = 0; j < list_L.dimension(); ++j) {
            L.data[L.pointer[i]+j] = w[list_L[j]];
            L.indices[L.pointer[i]+j] = list_L[j];
        }
        L.data[L.pointer[i]+list_L.dimension()] = 1.0;      // diagonal element
        L.indices[L.pointer[i]+list_L.dimension()] = i;     // diagonal element
        L.pointer[i+1] = L.pointer[i]+list_L.dimension()+1;

        // (12.) Copy values to U:
        U.data[U.pointer[i]] = w[i]; // diagonal element
        U.indices[U.pointer[i]] = i; // diagonal element
        for (Integer j = 0; j < list_U.dimension(); ++j) {
            U.data[U.pointer[i]+j+1] = w[list_U[j]];
            U.indices[U.pointer[i]+j+1] = list_U[j];
        }
        U.pointer[i+1] = U.pointer[i]+list_U.dimension()+1;

        if (U.data[U.pointer[i]] == 0)
            throw std::runtime_error("matrix_sparse::ILUT: encountered zero pivot in row " + std::to_string(i));

        // (13.) w:=0
        w.zero_reset();
    }  // (14.) end for i
    L.compress();
    U.compress();
    time_self = ((Real)clock() - (Real)time_begin) / (Real)CLOCKS_PER_SEC;
}

// Like ILUT(), but uses a heap to find k rather than a linear search through [0,i).
template<class T>
void ILUT_heap(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, Integer max_fill_in, Real threshold, Real& time_self)
{
    const clock_t time_begin=clock();
    // the notation will be for A being a ROW matrix, i.e. U also a ROW matrix and L a ROW matrix.
    if (non_fatal_error(!A.square_check(), "ILUT: argument matrix must be square"))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    const Integer m = A.rows(), n = A.columns();

    // sparse vector which allows sequential access to indices via a heap
    vector_sparse_ordered<T> w(m);
    index_list list_L, list_U;

    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;

    Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (n-max_fill_in)*max_fill_in;

    U.reformat(m, m, reserved_memory, ROW);
    L.reformat(m, m, reserved_memory, ROW);

    // (1.) begin for i -- loop over rows
    for (Integer i = 0; i < n; ++i) {
        // (2.) initialize w
        Real norm_wL = 0.0;     // norm of the L part of w
        for (Integer k = A.pointer[i]; k < A.pointer[i+1]; ++k) {
            w[A.indices[k]] = A.data[k];

            // take the norm only of the L part to be consistent with the dropping rule applied in (10.)
            if (A.indices[k] < i)
                norm_wL += absvalue_squared(A.data[k]);
        }     // end for k
        // (3.) begin for k
        norm_wL = std::sqrt(norm_wL);

        // for all k < i with w(k) != 0:
        for (Integer x = w.pop_next_index(); x >= 0; x = w.pop_next_index()) {
            const Integer k = w.get_pointer(x);
            if (k >= i)
                break;
            T& wk = w.get_data(x);
            if (wk == static_cast<T>(0))        // numeric zeros may crop up as stale entries in the pqueue
                continue;

            // (5.) Apply dropping to w.
            // NB: we do this BEFORE dividing by the diagonal, otherwise the scaling is wrong
            if (std::abs(wk) < threshold*norm_wL) {
                w.zero_set(k);
            } else {
                // (4.) w[k] = w[k] / U[k,k]
                // NB: the first entry of the k-th row of U is the diagonal entry
                wk /= U.data[U.pointer[k]];

                // (6./7./8.) w = w - w[k] * u[k,*] (w[k] scalar, u[k,*] a row of U)
                for (Integer j = U.pointer[k] + 1; j < U.pointer[k+1]; j++)
                    w[U.indices[j]] -= wk * U.data[j];
            }
        }  // (9.) end for k

        // (10.) Do dropping in w.
        // keep one space free for the diagonal; begin with 0 and go upto the diagonal, but not including it.
        w.take_largest_elements_by_abs_value_with_threshold(list_L, max_fill_in-1, threshold, 0, i);
        // keep one space free for the diagonal; begin with the element after the diagonal and go to the end.
        w.take_largest_elements_by_abs_value_with_threshold(list_U, max_fill_in-1, threshold, i+1, n);

        // (11.) Copy values to L:
        for (Integer j = 0; j < list_L.dimension(); ++j) {
            L.data[L.pointer[i]+j] = w[list_L[j]];
            L.indices[L.pointer[i]+j] = list_L[j];
        }
        L.data[L.pointer[i]+list_L.dimension()] = 1.0;      // diagonal element
        L.indices[L.pointer[i]+list_L.dimension()] = i;     // diagonal element
        L.pointer[i+1] = L.pointer[i]+list_L.dimension()+1;

        // (12.) Copy values to U:
        U.data[U.pointer[i]] = w[i]; // diagonal element
        U.indices[U.pointer[i]] = i; // diagonal element
        for (Integer j = 0; j < list_U.dimension(); ++j) {
            U.data[U.pointer[i]+j+1] = w[list_U[j]];
            U.indices[U.pointer[i]+j+1] = list_U[j];
        }
        U.pointer[i+1] = U.pointer[i]+list_U.dimension()+1;

        if (U.data[U.pointer[i]] == 0)
            throw std::runtime_error("matrix_sparse::ILUT: encountered zero pivot in row " + std::to_string(i));

        // (13.) w:=0
        w.zero_reset();
    }  // (14.) end for i
    L.compress();
    U.compress();
    time_self = ((Real)clock() - (Real)time_begin) / (Real)CLOCKS_PER_SEC;
}


// not yet functional. does it wrong.

template<class T>
void ILUT2(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, Integer max_fill_in, Real threshold, Real& time_self)
{
    if(non_fatal_error(!A.square_check(), "ILUT2: argument matrix must be square."))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);

    const clock_t time_begin = clock();
    // the notation will be for A being a ROW matrix, i.e. U also a ROW matrix and L a ROW matrix.
    const Integer m = A.rows(), n = A.columns();
    Integer k,i,j;
    vector_sparse_dynamic_enhanced<T> w(m);
    index_list list_L, list_U;

    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;

    Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (n-max_fill_in)*max_fill_in;

    U.reformat(m,m,reserved_memory,ROW);
    L.reformat(m,m,reserved_memory,ROW);

    // (1.) begin for i
    for (i=0;i<n;i++) {
        // (2.) initialize w
        for(k=A.pointer[i]; k<A.pointer[i+1]; k++)
            w(A.indices[k], A.indices[k]) = A.data[k];

        // (3.) begin for k
        const Real norm_w = w.norm2();
        while(w.current_sorting_index()<i && !w.at_end()){
            w.current_element() /= U.data[U.pointer[w.current_sorting_index()]];
            if(std::abs(w.current_element())<threshold*norm_w){
                w.current_zero_set();
                // taking a step forward is not necessary, because the iterator jumps automatically ahead if current element is erased.
            } else {
                for(j=U.pointer[w.current_sorting_index()]+1; j<U.pointer[w.current_sorting_index()+1]; j++){
                    w(U.indices[j],U.indices[j]) -= w.current_element()*U.data[j];
                } // end for
                w.take_step_forward();
            }   // end if
        } // end while
        // (10.) Do dropping in w.
        std::cout<<"w"<<std::endl<<w.expand();
        w.take_largest_elements_by_abs_value_with_threshold(list_L,max_fill_in-1,threshold,0,i);      // keep one space free for the diagonal; begin with 0 and go upto the diagonal, but not including it.
        w.take_largest_elements_by_abs_value_with_threshold(list_U,max_fill_in-1,threshold,i+1,n);    // keep one space free for the diagonal; begin with the element after the diagonal and go to the end.
        std::cout<<"ListL"<<std::endl<<list_L<<std::endl;
        // (11.) Copy values to L:
        for(j=0;j<list_L.dimension();j++){
            L.data[L.pointer[i]+j] = w[list_L[j]];
            L.indices[L.pointer[i]+j] = list_L[j];
            std::cout<<"copied "<<w[list_L[j]]<<" to position "<<L.indices[L.pointer[i]+j]<<std::endl;
        } // end for j
        L.data[L.pointer[i]+list_L.dimension()]=1.0;
        L.indices[L.pointer[i]+list_L.dimension()]=i;
        L.pointer[i+1]=L.pointer[i]+list_L.dimension()+1;
        // (12.) Copy values to U:
        U.data[U.pointer[i]]=w[i]; // diagonal element
        U.indices[U.pointer[i]]=i; // diagonal element
        for(j=0;j<list_U.dimension();j++){
            U.data[U.pointer[i]+j+1]=w[list_U[j]];
            U.indices[U.pointer[i]+j+1] = list_U[j];
        }  // end j
        U.pointer[i+1]=U.pointer[i]+list_U.dimension()+1;

        if (U.data[U.pointer[i]] == 0)
            throw std::runtime_error("ILUT2: encountered zero pivot in row " + std::to_string(i));

        // (13.) w:=0
        w.zero_reset();
    }  // (14.) end for i
    L.compress();
    U.compress();
    time_self=((Real)clock() - (Real)time_begin)/(Real)CLOCKS_PER_SEC;
}

} // end namespace iluplusplus
