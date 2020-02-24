#pragma once

#include "declarations.h"

namespace iluplusplus {

// calculates the (incomplete) LU factorization;
// returns false in case of break-down, true in case of success.
// although this algorithm works for A being both a ROW and COLUMN matrix,
// if A is a COLUMN matrix, L and U will be reversed, even though L still has 1's on the diagonal.
// In this case, it still works as a preconditioner

template<class T>
bool ILUT(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, Integer max_fill_in, Real threshold, Real& time_self)
{
    try {
        const clock_t time_begin=clock();
        // the notation will be for A being a ROW matrix, i.e. U also a ROW matrix and L a ROW matrix.
        if (non_fatal_error(!A.square_check(), "matrix_sparse::ILUT: argument matrix must be square."))
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        Real norm=0.0;
        const Integer m = A.rows(), n = A.columns();

        vector_sparse_dynamic<T> w;
        index_list list_L, list_U;

        if(max_fill_in<1) max_fill_in = 1;
        if(max_fill_in>n) max_fill_in = n;

        Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (n-max_fill_in)*max_fill_in;
        w.resize(m);
        U.reformat(m,m,reserved_memory,ROW);
        L.reformat(m,m,reserved_memory,ROW);
        U.pointer[0]=0;
        L.pointer[0]=0;

        // (1.) begin for i -- loop over rows
        for (Integer i = 0; i < n; ++i) {
            // (2.) initialize w
            for (Integer k = A.pointer[i]; k < A.pointer[i+1]; ++k) {
                w[A.indices[k]] = A.data[k];
                //if(A.indices[k]==i) diag_A[i]=A.data[k];
            }     // end for k
            // (3.) begin for k
            const Real norm_w = w.norm2();
            /*
             * cfh:
             *
             * It would be nice to iterate only over the non-zeros in w here, but
             * the modification to w below creates non-sorted nonzeros which
             * breaks the algorithm.
             *
             * It would be possible to re-sort w in each iteration; worth it?
             */
            for (Integer k = 0; k < i; ++k) {
                if (w.non_zero_check(k)) {
                    // (4.) w[k]= w[k] / U[k,k]
                    // NB: the first entry of the k-th row of U is the diagonal entry
                    const T U_kk = U.data[U.pointer[k]];
                    w[k] /= U_kk;
                    // (5.) Apply dropping to w.
                    // (6./7./8.) w = w -w[k] * u[k,*] (w[k] scalar, u[k,*] a row of U)
                    // no need to check if w[k] != 0; this has already been done.
                    // BUG: the scaling seems off here since we already divided by the diagonal?
                    if (std::abs(w[k]) < threshold*norm_w) {
                        w.zero_set(k);
                    } else {
                        for (Integer j = U.pointer[k] + 1; j < U.pointer[k+1]; j++)
                            w[U.indices[j]] -= w[k]*U.data[j];
                    }
                } // end if
            }  // (9.) end for k

            // (10.) Do dropping in w. (Needs to be implemented)
            // keep one space free for the diagonal; begin with 0 and go upto the diagonal, but not including it.
            w.take_largest_elements_by_abs_value_with_threshold(norm,list_L,max_fill_in-1,threshold,0,i);
            // keep one space free for the diagonal; begin with the element after the diagonal and go to the end.
            w.take_largest_elements_by_abs_value_with_threshold(norm,list_U,max_fill_in-1,threshold,i+1,n);

            // (11.) Copy values to L:
            for (Integer j = 0; j < list_L.dimension(); ++j) {
                L.data[L.pointer[i]+j] = w[list_L[j]];
                L.indices[L.pointer[i]+j] = list_L[j];
            }
            L.data[L.pointer[i]+list_L.dimension()]=1.0;
            L.indices[L.pointer[i]+list_L.dimension()]=i;
            L.pointer[i+1]=L.pointer[i]+list_L.dimension()+1;
            // (12.) Copy values to U:
            U.data[U.pointer[i]]=w[i]; // diagonal element
            U.indices[U.pointer[i]]=i; // diagonal element
            for (Integer j = 0; j < list_U.dimension(); ++j) {
                U.data[U.pointer[i]+j+1]=w[list_U[j]];
                U.indices[U.pointer[i]+j+1] = list_U[j];
            }  // end j
            U.pointer[i+1] = U.pointer[i]+list_U.dimension()+1;
            if (U.data[U.pointer[i]] == 0) {
#ifdef VERBOSE
                std::cerr<<"matrix_sparse::ILUT: encountered zero pivot in row "<<k<<std::endl;
#endif
                L.reformat(0,0,0,A.orientation);
                U.reformat(0,0,0,A.orientation);
                return false;
            }
            // (13.) w:=0
            w.zero_reset();
        }  // (14.) end for i
        L.compress();
        U.compress();
        time_self = ((Real)clock() - (Real)time_begin) / (Real)CLOCKS_PER_SEC;
        return true;
    }  // end try (code not indented)
    catch(iluplusplus_error ippe){
        std::cerr<<"matrix_sparse::ILUT: "<<ippe.error_message()<<" Returning 0x0 matrices."<<std::endl<<std::flush;
        U.reformat(0,0,0,ROW);
        L.reformat(0,0,0,ROW);
        return false;
    }
}


// not yet functional. does it wrong.

template<class T>
bool ILUT2(const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, Integer max_fill_in, Real threshold, Real& time_self)
{
    clock_t time_begin, time_end;
    time_begin=clock();
    if(threshold > 500) threshold = 0.0;
    else threshold=std::exp(-threshold*std::log(10.0));
    // the notation will be for A being a ROW matrix, i.e. U also a ROW matrix and L a ROW matrix.
    if(non_fatal_error(!A.square_check(),"matrix_sparse::ILUT: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Real norm=0.0;
    Real norm_w;
    Integer m = A.rows();
    Integer n = A.columns();
    Integer k,i,j;
    vector_sparse_dynamic_enhanced<T> w(m);
    index_list list_L;
    index_list list_U;
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (n-max_fill_in)*max_fill_in;
    U.reformat(m,m,reserved_memory,ROW);
    L.reformat(m,m,reserved_memory,ROW);
    // (1.) begin for i
    for(i=0;i<n;i++){
        // (2.) initialize w
        for(k=A.pointer[i];k<A.pointer[i+1];k++){
            w(A.indices[k],A.indices[k]) = A.data[k];
        }     // end for k
        // (3.) begin for k
        norm_w=w.norm2();
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
        w.take_largest_elements_by_abs_value_with_threshold(norm,list_L,max_fill_in-1,threshold,0,i);      // keep one space free for the diagonal; begin with 0 and go upto the diagonal, but not including it.
        w.take_largest_elements_by_abs_value_with_threshold(norm,list_U,max_fill_in-1,threshold,i+1,n);    // keep one space free for the diagonal; begin with the element after the diagonal and go to the end.
        std::cout<<"ListL"<<std::endl<<list_L<<std::endl;
        // (11.) Copy values to L:
        for(j=0;j<list_L.dimension();j++){
            L.data[L.pointer[i]+j] = w.read(list_L[j]);
            L.indices[L.pointer[i]+j] = list_L[j];
            std::cout<<"copied "<<w.read(list_L[j])<<" to position "<<L.indices[L.pointer[i]+j]<<std::endl;
        } // end for j
        L.data[L.pointer[i]+list_L.dimension()]=1.0;
        L.indices[L.pointer[i]+list_L.dimension()]=i;
        L.pointer[i+1]=L.pointer[i]+list_L.dimension()+1;
        // (12.) Copy values to U:
        U.data[U.pointer[i]]=w.read(i); // diagonal element
        U.indices[U.pointer[i]]=i; // diagonal element
        for(j=0;j<list_U.dimension();j++){
            U.data[U.pointer[i]+j+1]=w.read(list_U[j]);
            U.indices[U.pointer[i]+j+1] = list_U[j];
        }  // end j
        U.pointer[i+1]=U.pointer[i]+list_U.dimension()+1;
        if(U.data[U.pointer[i]]==0) {
#ifdef VERBOSE
            std::cerr<<"matrix_sparse::ILUT2: encountered zero pivot in row "<<k<<std::endl;
#endif
            L.reformat(0,0,0,A.orientation);
            U.reformat(0,0,0,A.orientation);
            return false;
        }
        // (13.) w:=0
        w.zero_reset();
    }  // (14.) end for i
    L.compress();
    U.compress();
    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    return true;
}

} // end namespace iluplusplus
