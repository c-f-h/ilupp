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
    const Integer m = A.dim_along_orientation(), n = A.dim_against_orientation();   // for csr: rows, columns
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
    for(k=0; k < m; k++){  // (1.) in algorithm of Saad.
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

        if (z.zero_check(k))
            throw std::runtime_error("ILUC2: zero pivot on diagonal, k=" + std::to_string(k));

        // apply dropping to w - (10.) in the algorithm of Saad.
        w.take_largest_elements_by_abs_value_with_threshold(norm_w, listw, max_fill_in-1, threshold, k+1, m);

        // apply dropping to z - (11.) in the algorithm of Saad.
        z.take_largest_elements_by_abs_value_with_threshold(norm_z, listz, max_fill_in-1, threshold, k+1, n);

        // copy z to U - (12.) in the algorithm of Saad.

        // ensure we reserved enough space
        if (U.pointer[k]+listz.dimension() > reserved_memory
                || L.pointer[k]+listw.dimension()+1 > reserved_memory) {
            throw std::runtime_error("ILUC2: memory reserved was insufficient. Increase mem_factor!");
        }

        // set U[k,k] = z[k] = A[k,k]
        U.indices[U.pointer[k]] = k;
        U.data[U.pointer[k]] = z[k];

        // copy z[:] to U[k,:]
        for(j=0; j<listz.dimension(); ++j) {
            U.data   [U.pointer[k]+j+1] = z[listz[j]];
            U.indices[U.pointer[k]+j+1] = listz[j];
        }
        U.pointer[k+1] = U.pointer[k]+listz.dimension()+1;

        // copy w to L - (13.) in the algorithm of Saad.
        // set L[k,k] = 1
        L.indices[L.pointer[k]] = k;
        L.data[L.pointer[k]] = 1.0;

        // copy w[:] / U_kk to L[:,k]
        const T U_kk = U.data[U.pointer[k]];  // diagonal entry of U
        for(j=0; j<listw.dimension(); ++j) {
            L.data   [L.pointer[k]+j+1] = w[listw[j]] / U_kk;
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

// we use linked list for L and A, A is only needed in column format and we perform column pivoting
// U is stored with a companion structure
template<class T> bool ILUCP4(const matrix_sparse<T>& Acol,
        matrix_sparse<T>& L, matrix_sparse<T>& U, index_list& perm, Integer max_fill_in,
        Real threshold, Real perm_tol, Integer rp, Integer& zero_pivots, Real& time_self, Real mem_factor=10.0)
{
    const clock_t time_begin = clock();

    if (perm_tol > 500.0) perm_tol=0.0;
    else perm_tol=std::exp(-perm_tol*std::log(10.0));

    if (non_fatal_error(!Acol.square_check(), "matrix_sparse::ILUCP4: argument matrix must be square."))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    const Integer n = Acol.columns();
    Integer k, i, j, p, current_row_col_U, h, pos;
    zero_pivots=0;
    Real norm_L, norm_U; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    vector_sparse_dynamic<T> w, z;
    vector_dense<bool> non_pivot;
    index_list list_L, list_U, inverse_perm;

    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*Acol.non_zeroes());

    std::vector<Integer> firstL(n), listL(n), firstA(n), listA(n), headA(n),
        linkU(reserved_memory), rowU(reserved_memory), startU;

    U.reformat(n,n,reserved_memory,ROW);
    L.reformat(n,n,reserved_memory,COLUMN);
    perm.resize(n);
    w.resize(n);
    z.resize(n);
    non_pivot.resize(n,true);
    inverse_perm.resize(n);
    initialize_triangular_fields(n,listL);
    initialize_sparse_matrix_fields(n,Acol.pointer,Acol.indices,listA,headA,firstA);

    startU.resize(n, -1);

    // (1.) begin for k
    for(k=0;k<n;k++){
        if (k == rp) perm_tol = 1.0;  // permute always
        // (2.) initialize z
        z.zero_reset();

        // read row of A
        h=headA[k];
        while(h!=-1){
            if(non_pivot[h]) z[h]=Acol.data[firstA[h]];
            h=listA[h];
        }

        // (3.) begin while
        h=listL[k];
        while(h!=-1){
            // h is current column index of k-th row of L
            for(j=U.pointer[h];j<U.pointer[h+1];j++){
                if(non_pivot[U.indices[j]]){
                    z[U.indices[j]] -= L.data[firstL[h]]*U.data[j];
                } // end if
            } // end for
            h=listL[h];
        } // end while (5.)

        // (6.) sort and copy data to U; update information for accessing columns of U
        z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,threshold,perm[k],perm_tol);
        // dropping too stringent?
        if(list_U.dimension()==0){
            if(threshold>0.0)
                //std::cout<<"Dropping too stringent, selecting elements without threshold."<<std::endl;
                z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,0.0,perm[k],perm_tol);
        }
        // still no non-zero elements?
        if(list_U.dimension()==0){
            zero_pivots++;
            z[perm[k]]=1.0;
            list_U.resize(1);
            list_U[0]=perm[k];
        } // end if
        if(U.pointer[k]+list_U.dimension()>reserved_memory){
            throw std::runtime_error("ILUCP4: Insufficient memory reserved. Increase mem_factor");
        }
        // copy data, update access information.
        // copy pivot
        U.data[U.pointer[k]] = z[list_U[list_U.dimension()-1]];
        U.indices[U.pointer[k]] = list_U[list_U.dimension()-1];
        for(j=1;j<list_U.dimension();j++){
            pos = U.pointer[k]+j;
            U.data[pos] = z[list_U[list_U.dimension()-1-j]];
            U.indices[pos] = list_U[list_U.dimension()-1-j];
            h = startU[U.indices[pos]];
            startU[U.indices[pos]] = pos;
            linkU[pos] = h;
            rowU[pos] = k;
        }
        U.pointer[k+1] = U.pointer[k]+list_U.dimension();
        if(U.data[U.pointer[k]]==0){
            std::cerr<<"dim list_U "<<list_U.dimension()<<std::endl;
            std::cerr<<"last element corresponding to pivot: "<<z[perm[k]]<<std::endl;
            throw std::runtime_error("ILUCP4: Pivot is zero, because pivoting was not permitted.");
        }
        // store positions of columns of U, but without pivot
        // update non-pivots.
        // (7.) update permutations
        p = inverse_perm[U.indices[U.pointer[k]]];
        inverse_perm.switch_index(perm[k],U.indices[U.pointer[k]]);
        perm.switch_index(k,p);
        non_pivot[U.indices[U.pointer[k]]] = false;

        // (8.) read w
        w.zero_reset();
        // read column of A
        /* // works fine as alternative, but not really faster
           for(i=firstA[perm[k]];i<Acol.pointer[perm[k]+1];i++)
           w[Acol.indices[i]] = Acol.data[i];
           */
        for(i = Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
            if(Acol.indices[i]>k)
                w[Acol.indices[i]] = Acol.data[i];
        }     // end for i

        // (9.) begin while
        h = startU[perm[k]];
        while(h!=-1){
            current_row_col_U = rowU[h];
            const T current_data_col_U = U.data[h];
            h = linkU[h];
            // (10.) w = w - U(i,perm(k))*l_i
            for(j = L.pointer[current_row_col_U];j<L.pointer[current_row_col_U+1];j++){
                w[L.indices[j]] -= current_data_col_U * L.data[j];
            } // end for
        }   // (11.) end while

        // (12.) sort and copy data to L
        // sort
        w.take_largest_elements_by_abs_value_with_threshold(norm_L,list_L,max_fill_in-1,threshold,k+1,n);
        if(L.pointer[k]+list_L.dimension()+1>reserved_memory){
            throw std::runtime_error("ILUCP4: Insufficient memory reserved. Increase mem_factor");
        }
        // copy data
        L.data[L.pointer[k]] = 1.0;
        L.indices[L.pointer[k]] = k;
        for(j = 0;j<list_L.dimension();j++){
            L.data[L.pointer[k]+j+1] = w[list_L[j]]/U.data[U.pointer[k]];
            L.indices[L.pointer[k]+j+1] = list_L[j];
        } // end for j
        L.pointer[k+1] = L.pointer[k]+list_L.dimension()+1;
        update_sparse_matrix_fields(k, Acol.pointer,Acol.indices,listA,headA,firstA);
        update_triangular_fields(k, L.pointer,L.indices,listL,firstL);
    }  // (13.) end for k

    L.compress();
    U.compress();

    time_self = ((Real)clock() - (Real)time_begin) / (Real)CLOCKS_PER_SEC;
    return true;
}


} // end namespace iluplusplus
