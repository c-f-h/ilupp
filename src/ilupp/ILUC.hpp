#pragma once

#include "declarations.h"

namespace iluplusplus {

/*********************************************************************/
// Declarations for ILUC-type preconditioners
/*********************************************************************/

template <class IntArr>
void initialize_triangular_fields(Integer n, IntArr& list);

template <class IntArr>
void update_triangular_fields(Integer k, Integer *pointer, Integer *indices, IntArr& list, IntArr& first);

template <class IntArr>
void insert(IntArr& list, IntArr& head, Integer i, Integer j);

template <class IntArr>
void initialize_sparse_matrix_fields(Integer n, Integer *pointer, Integer *indices, IntArr& list, IntArr& head, IntArr& first);

template <class IntArr>
void update_sparse_matrix_fields(Integer k, Integer *pointer, Integer *indices, IntArr& list, IntArr& head, IntArr& first);

/**************************************************************************************************************************************/
//             functions for ILUC-type factorizations
/**************************************************************************************************************************************/

template <class IntArr>
void initialize_triangular_fields(Integer n, IntArr& list){
    for (int k=0;k<n;k++)
        list[k]=-1;
}

template <class IntArr>
void update_triangular_fields(Integer k, Integer *pointer, Integer *indices, IntArr& list, IntArr& first){
    Integer h,i,j;

    // update first:
    for (h = list[k]; h != -1; h = list[h])
        first[h] += 1;

    first[k]=pointer[k]+1; // can grow too large, i.e. move erroneously to the next row.
    // update list; insert k  at appropriate position
    h=list[k];
    if(pointer[k]+1<pointer[k+1]){
        j=indices[pointer[k]+1];
        list[k]=list[j];
        list[j]=k;
    }
    // distribute P_k
    while(h!=-1){
        i=h;
        h=list[i];
        if(first[i]<pointer[i+1]){
            j=indices[first[i]];
            list[i]=list[j];
            list[j]=i;
        }
    }
}

// insert i in P_j for the functions below

template <class IntArr>
void insert(IntArr& list, IntArr& head, Integer i, Integer j) {
    list[i]=head[j];
    head[j]=i;
}

template <class IntArr>
void initialize_sparse_matrix_fields(Integer n, Integer *pointer, Integer *indices, IntArr& list, IntArr& head, IntArr& first) {
    for(int k=0; k<n; k++)
        head[k] = -1;

    for(int k=0; k<n; k++) {
        first[k] = pointer[k];
        if (pointer[k] < pointer[k+1])
            insert(list, head, k, indices[pointer[k]]); // inserting k into P_indices[pointer[k]]
    }
}

template <class IntArr>
void update_sparse_matrix_fields(Integer k, Integer *pointer, Integer *indices, IntArr& list, IntArr& head, IntArr& first) {
    Integer h,i;

    // update first:
    for (h = head[k]; h != -1; h = list[h])
        first[h] += 1; // can grow too large, i.e. move erroneously to the next row.

    // update list, distribute P_k
    h=head[k];
    while (h != -1) {
        i=h;
        h=list[i];
        if(first[i] < pointer[i+1])
            insert(list, head, i, indices[first[i]]);
    }
}

// calculates the (incomplete) LU factorization with Crout;
// *this will be U such that A = LU;
// see: Saad: "Crout Versions of ILU for general sparse matrices".
// returns false in case of break-down, true in case of success.
// although this algorithm works for A being both a ROW and COLUMN matrix,
// if A is a COLUMN matrix, L and U will be reversed, even though L still has 1's on the diagonal.
// In this case, it still works as a preconditioner
// ILUC2 employs the linked lists for data structures and ist MUCH faster.
template <class T>
bool ILUC2(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, Integer max_fill_in, Real threshold, Real mem_factor = 10.0)
{
    Integer m = A.rows();
    Integer n = A.columns();
    Integer k,j,h;
    vector_sparse_dynamic<T> z,w;
    Real norm_w=0.0, norm_z=0.0;
    index_list listw, listz;
    // calculate maximal size needed for L and U:
    Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*A.non_zeroes());

    // the following field will store the index of the first element in the i-th row having a column index >= k (i=0...m-1).
    // listU field will contain the information needed to retrieve a column of U.
    // listL: same as above for L, but orientation reversed:
    std::vector<Integer> firstU(m), firstL(m), listA(m), headA(m), firstA(m), listU(m), listL(m);

    // reformat L and U and initialize the pointers:
    L.reformat(m, m, reserved_memory, other_orientation(A.orientation));
    U.reformat(m, n, reserved_memory, A.orientation);
    z.resize(n);
    w.resize(m);
    if (max_fill_in<1) max_fill_in = 1;

    //initialize fields.
    initialize_sparse_matrix_fields(m, A.pointer, A.indices, listA, headA, firstA);
    initialize_triangular_fields(m, listL);
    initialize_triangular_fields(m, listU);

    // Iterate over the rows of A or U respectively.
    for(k=0; k<min(m,n); k++){  // (1.) in algorithm of Saad.
        // initialize z to upper part of row k - (2.) in algorithm of Saad.
        z.zero_reset();
        for(j=firstA[k]; j<A.pointer[k+1]; ++j)
            z[A.indices[j]] = A.data[j];

        // subtract multiples of the various rows of U from z, the new row of U
        // (3.) in the algorithm of Saad.
        for (h = listL[k]; h != -1; h = listL[h]) {
            // h is current column index of k-th row of L
            const T L_kh = L.data[firstL[h]];
            for (j=firstU[h]; j<U.pointer[h+1]; ++j) {
                // (4.) in the algorithm of Saad.
                z[U.indices[j]] -= L_kh * U.data[j];
            }  // end for j
        } // end while (5.) in algorithm of Saad.

        // initialize w to lower part of column k - (6.) in algorithm of Saad.
        w.zero_reset();
        for (h = headA[k]; h != -1; h = listA[h]) {
            if(h > k)
                w[h] = A.data[firstA[h]];
        }

        // (7.) in the algorithm of Saad.
        for (h = listU[k]; h != -1; h = listU[h]) {
            // h is current row index of k-th column of U
            const T U_hk = U.data[firstU[h]];
            for (j=firstL[h]; j<L.pointer[h+1]; ++j) {
                // (8.) in the algorithm of Saad.
                w[L.indices[j]] -= U_hk * L.data[j];
            }  // end for j
        } // end while (9.) in algorithm of Saad.

        if (z.zero_check(k)) {
            L.reformat(0,0,0,other_orientation(A.orientation));
            U.reformat(0,0,0,A.orientation);
            return false;
        }

        // apply dropping to w - (10.) in the algorithm of Saad.
        w.take_largest_elements_by_abs_value_with_threshold(norm_w, listw, max_fill_in-1, threshold, k+1, m);

        // apply dropping to z - (11.) in the algorithm of Saad.
        z.take_largest_elements_by_abs_value_with_threshold(norm_z, listz, max_fill_in-1, threshold, k+1, n);

        // copy z to U - (12.) in the algorithm of Saad.
        U.indices[U.pointer[k]] = k;
        U.data[U.pointer[k]] = z[k];
        if (U.pointer[k]+listz.dimension() > reserved_memory
                || L.pointer[k]+listw.dimension()+1 > reserved_memory) {
            std::cerr<<"matrix_sparse::ILUC2: memory reserved was insufficient. Returning 0x0 matrix."<<std::endl;
            L.reformat(0,0,0,other_orientation(A.orientation));
            U.reformat(0,0,0,A.orientation);
            return false;
        }
        for(j=0; j<listz.dimension(); ++j) {
            U.data[U.pointer[k]+j+1] = z[listz[j]];
            U.indices[U.pointer[k]+j+1] = listz[j];
        }
        U.pointer[k+1] = U.pointer[k]+listz.dimension()+1;

        // copy w to L - (13.) in the algorithm of Saad.
        L.indices[L.pointer[k]] = k;
        L.data[L.pointer[k]] = 1.0;
        const T U_kk = U.data[U.pointer[k]];  // diagonal entry of U
        for(j=0; j<listw.dimension(); ++j) {
            L.data[L.pointer[k]+j+1] = w[listw[j]] / U_kk;
            L.indices[L.pointer[k]+j+1] = listw[j];
        }
        L.pointer[k+1] = L.pointer[k]+listw.dimension()+1;

        update_sparse_matrix_fields(k, A.pointer, A.indices, listA, headA, firstA);
        update_triangular_fields(k, L.pointer, L.indices, listL, firstL);
        update_triangular_fields(k, U.pointer, U.indices, listU, firstU);
    }  // end for k - (14.) in the algorithm of Saad.

    // fill out remaining rows, if any (only for tall matrices)
    for(k=n+1; k<=m; ++k) {
        U.pointer[k] = U.pointer[n];
        L.indices[L.pointer[k-1]] = k - 1;
        L.data[L.pointer[k-1]] = 1.0;
        L.pointer[k] = L.pointer[k-1] + 1;
    }
    L.compress(-1.0);
    U.compress(-1.0);
    return true;
}

} // end namespace iluplusplus
