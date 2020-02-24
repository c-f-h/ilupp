#pragma once

#include "declarations.h"

namespace iluplusplus {

//bool ILUTP_new(const matrix_sparse<T>& A, matrix_sparse<T>& U, index_list& perm, Integer max_fill_in, Real threshold, Integer& zero_pivots);

// ILUTP is the standard implemenation. Accessing elements of w in increasing
// order is slow. This is improved in ILUTP2 using vector_sparse_dynamic_enhanced

template<class T>
bool ILUTP2(
        const matrix_sparse<T>& A, matrix_sparse<T>& L, matrix_sparse<T>& U, index_list& perm,
        Integer max_fill_in, Real threshold, Real perm_tol, Integer bp,
        Integer& zero_pivots, Real& time_self, Real mem_factor)
{
    clock_t time_begin, time_end;
    time_begin=clock();
    // the notation will be for A being a ROW matrix, i.e. U also a ROW matrix and L a ROW matrix.
    if (perm_tol > 500) perm_tol=0.0;
    else perm_tol=std::exp(-perm_tol*std::log(10.0));

    if(non_fatal_error(!A.square_check(),"matrix_sparse::ILUTP2: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer m = A.rows();
    Integer n = A.columns();
    Integer k,i,j,p;
    zero_pivots=0;
    Real norm_L,norm_U, norm_w; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    vector_sparse_dynamic_enhanced<T> w;
    index_list list_L;
    index_list list_U;
    index_list inverse_perm;
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*A.non_zeroes());
    U.reformat(m,m,reserved_memory,ROW);
    L.reformat(m,m,reserved_memory,ROW);
    perm.resize(n);
    w.resize(m);
    inverse_perm.resize(n);
    // (1.) begin for i

    for(i=0;i<n;i++){
        if (i == bp) perm_tol = 1.0;

        // (2.) initialize w
        for(k=A.pointer[i];k<A.pointer[i+1];k++){
            w(A.indices[k],inverse_perm[A.indices[k]]) = A.data[k];
        }     // end for k

        norm_w=w.norm2();
        w.move_to_beginning();
        while(w.current_sorting_index()<i && !w.at_end()){
            w.current_element() /= U.data[U.pointer[w.current_sorting_index()]];
            if(abs(w.current_element())<threshold*norm_w){
                w.current_zero_set();
                // taking a step forward is not necessary, because the iterator jumps automatically ahead if current element is erased.
            } else {
                for(j=U.pointer[w.current_sorting_index()]+1; j<U.pointer[w.current_sorting_index()+1]; j++){
                    w(U.indices[j],inverse_perm[U.indices[j]]) -= w.current_element()*U.data[j];
                } // end for
                w.take_step_forward();
            }   // end if
        } // end while

        // (10.) Do dropping in w.
        w.take_largest_elements_by_abs_value_with_threshold(norm_L,norm_U,list_L,list_U,inverse_perm,max_fill_in-1,max_fill_in,threshold,threshold,i,perm_tol); // we need one element less for L, as the diagonal will always be 1.
        if(list_U.dimension()==0){
            if(threshold>0.0) w.take_largest_elements_by_abs_value_with_threshold(norm_L,norm_U,list_L,list_U,inverse_perm,max_fill_in-1,max_fill_in,threshold,0.0,i); // we need one element less for L, as the diagonal will always be 1.
            if(list_U.dimension()==0){
                zero_pivots++;
                w(perm[i],i)=1.0;
                list_U.resize(1);
                list_U[0]=perm[i];
            }
        }

        if(L.pointer[i]+list_L.dimension()+1>reserved_memory){
            std::cerr<<"matrix_sparse::ILUTP2: memory reserved was insufficient."<<std::endl;
            return false;
        }
        for(j=0;j<list_L.dimension();j++){
            L.data[L.pointer[i]+j] = w.read(list_L[list_L.dimension()-1-j]);
            L.indices[L.pointer[i]+j] = inverse_perm[list_L[list_L.dimension()-1-j]];
        } // end for j
        L.data[L.pointer[i]+list_L.dimension()]=1.0;
        L.indices[L.pointer[i]+list_L.dimension()]=i;
        L.pointer[i+1]=L.pointer[i]+list_L.dimension()+1;
        // (12.) Copy values to U:
        if(U.pointer[i]+list_U.dimension()>reserved_memory){
            throw std::runtime_error("matrix_sparse::ILUTP2: memory reserved was insufficient.");
        }
        for(j=0;j<list_U.dimension();j++){
            U.data[U.pointer[i]+j] = w.read(list_U[list_U.dimension()-1-j]);
            U.indices[U.pointer[i]+j] = list_U[list_U.dimension()-1-j];
        }  // end j
        U.pointer[i+1]=U.pointer[i]+list_U.dimension();
        p=inverse_perm[U.indices[U.pointer[i]]];
        inverse_perm.switch_index(perm[i],U.indices[U.pointer[i]]);
        perm.switch_index(i,p);
        if(U.data[U.pointer[i]]==0) {
            throw std::runtime_error("matrix_sparse::ILUTP2: encountered zero pivot in row ");
        }

        // (13.) w:=0
        w.zero_reset();
    }  // (14.) end for i

    L.compress();
    U.compress();

    U.reorder(inverse_perm);
    L.normal_order();

    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    return true;
}




} // end namespace iluplusplus
