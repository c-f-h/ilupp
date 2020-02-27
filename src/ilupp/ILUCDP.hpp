#pragma once

#include "declarations.h"

namespace iluplusplus {

template<class T> bool matrix_sparse<T>::ILUCDP(const matrix_sparse<T>& Arow, const matrix_sparse<T>& Acol, matrix_sparse<T>& U, index_list& perm, index_list& permrows, Integer max_fill_in, Real threshold, Real perm_tol,  Integer bpr, Integer& zero_pivots, Real& time_self, Real mem_factor){
    clock_t time_begin, time_end;
    time_begin=clock();
    if (threshold > 500.0) threshold=0.0;
    else threshold=std::exp(-threshold*std::log(10.0));
    if (perm_tol > 500.0) perm_tol=0.0;
    else perm_tol=std::exp(-perm_tol*std::log(10.0));
#ifdef VERBOSE
    clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_8,time_9;
    Real time_init=0.0;
    Real time_read=0.0;
    Real time_calc_L=0.0;
    Real time_scu_L=0.0;  // sorting, copying, updating access information
    Real time_calc_U=0.0;
    Real time_scu_U=0.0;
    Real time_zeroset=0.0;
    Real time_compress=0.0;
    Real time_resort=0.0;
    time_0 = clock();
#endif
    if(non_fatal_error(!Arow.square_check(),"matrix_sparse::ILUCDP: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(!Acol.square_check(),"matrix_sparse::ILUCDP: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(Acol.rows()!=Arow.rows(),"matrix_sparse::ILUCDP: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = Acol.columns();
    Integer a,b,k,i,j,p,current_row_col_U,current_col_row_L;
    Integer h,pos, selected_row;
    T current_data_row_L,current_data_col_U;
    zero_pivots=0;
    Real norm_L,norm_U; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    vector_sparse_dynamic<T> w, z;
    vector_dense<bool> non_pivot, unused_rows;
    vector_dense<Integer> numb_el_row_L, pointer_num_el_row_L;
    index_list list_L, list_U;
    index_list inverse_perm, inverse_permrows;
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*Acol.non_zeroes());

    std::vector<Integer> linkU(reserved_memory), rowU(reserved_memory), startU(n);
    std::vector<Integer> linkL(reserved_memory), colL(reserved_memory), startL(n);

    U.reformat(n,n,reserved_memory,ROW);
    reformat(n,n,reserved_memory,COLUMN);
    perm.resize(n);
    permrows.resize(n);
    inverse_perm.resize(n);
    inverse_permrows.resize(n);
    non_pivot.resize(n,true); 
    unused_rows.resize(n,true);
    numb_el_row_L.resize(n,0);
    pointer_num_el_row_L.resize(n+2,n);
    w.resize(n);
    z.resize(n);
    pointer_num_el_row_L[0]=0;
    for(k=0;k<n;k++) startU[k]=-1;
    for(k=0;k<n;k++) startL[k]=-1;
    // (1.) begin for k
#ifdef VERBOSE
    time_1 = clock();
    time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
#endif
    for(k=0;k<n;k++){
        if (k == bpr) perm_tol = 1.0;  // permute always
        //if (k == bpr) threshold = 0.0;
        //if (k == bpr) threshold *= (0.1>threshold) ? 0.1 : threshold;
#ifdef VERBOSE
        time_2=clock();
#endif
        // (2.) initialize z
        selected_row = permrows[k];
        unused_rows[selected_row]=false;
        z.zero_reset();
#ifdef VERBOSE
        time_3=clock();
        time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
        // read row of A
        for(i=Arow.pointer[selected_row];i<Arow.pointer[selected_row+1];i++){
            if(non_pivot[Arow.indices[i]]) z[Arow.indices[i]] = Arow.data[i];
        }     // end for i
#ifdef VERBOSE
        time_4=clock();
        time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
#endif
        // (3.) begin while
        h=startL[selected_row]; // h=startL[permrows[k]];
        while(h!=-1){
            current_col_row_L=colL[h];
            current_data_row_L=data[h];
            h=linkL[h];
            for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
            } // end for
        }   // (5.) end while
#ifdef VERBOSE
        time_5=clock();
        time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif
        // (6.) sort and copy data to U; update information for accessing columns of U
        z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,threshold,perm[k],perm_tol);
        // dropping too stringent?
        if(list_U.dimension()==0){
            if(threshold>0.0)
#ifdef VERBOSE
                std::cout<<"Dropping too stringent, selecting elements without threshold."<<std::endl;
#endif
            z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,0.0,perm[k],perm_tol);
        }
        // still no non-zero elements?
        if(list_U.dimension()==0){
#ifdef VERBOSE
            std::cout<<"Obtained a zero row, setting an arbitrary element to 1."<<std::endl;
#endif
            zero_pivots++;
            z[perm[k]]=1.0;
            list_U.resize(1);
            list_U[0]=perm[k];
        } // end if
        if(U.pointer[k]+list_U.dimension()>reserved_memory){
            std::cerr<<"matrix_sparse::ILUCDP: memory reserved was insufficient. Returning 0x0 matrices and permutations of dimension 0."<<std::endl;
            perm.resize(0);
            permrows.resize(0);
            reformat(0,0,0,COLUMN);
            U.reformat(0,0,0,ROW);
            return false;
        }
        // copy data, update access information.
        // copy pivot
        U.data[U.pointer[k]]=z[list_U[list_U.dimension()-1]];
        U.indices[U.pointer[k]]=list_U[list_U.dimension()-1];
        for(j=1;j<list_U.dimension();j++){
            pos=U.pointer[k]+j;
            U.data[pos]=z[list_U[list_U.dimension()-1-j]];
            U.indices[pos]=list_U[list_U.dimension()-1-j];
            h=startU[U.indices[pos]];
            startU[U.indices[pos]]=pos;
            linkU[pos]=h;
            rowU[pos]=k;
        }
        U.pointer[k+1]=U.pointer[k]+list_U.dimension();
        if(U.data[U.pointer[k]]==0){
            std::cerr<<"matrix_sparse::ILUCDP: Pivot is zero, because pivoting was not permitted. Preconditioner does not exist.Returning 0x0 matrices and permutations of dimension 0. "<<std::endl;
            std::cout<<"dim list_U "<<list_U.dimension()<<std::endl;
            std::cout<<"last element corresponding to pivot: "<<z[perm[k]]<<std::endl;
            perm.resize(0);
            permrows.resize(0);
            reformat(0,0,0,COLUMN);
            U.reformat(0,0,0,ROW);
            return false;
        }
        // store positions of columns of U, but without pivot
        // update non-pivots.
        // (7.) update permutations
        p=inverse_perm[U.indices[U.pointer[k]]];
        inverse_perm.switch_index(perm[k],U.indices[U.pointer[k]]);
        perm.switch_index(k,p);
        non_pivot[U.indices[U.pointer[k]]]=false;
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
#endif
        // (8.) read w
        w.zero_reset();
#ifdef VERBOSE
        time_7=clock();
        time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
#endif
        // read column of A
        for(i=Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
            if(unused_rows[Acol.indices[i]])
                w[Acol.indices[i]] = Acol.data[i];
        }     // end for i
#ifdef VERBOSE
        time_8=clock();
        time_read += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
#endif
        // (9.) begin while
        h=startU[perm[k]];
        while(h!=-1){
            current_row_col_U=rowU[h];
            current_data_col_U=U.data[h];
            h=linkU[h];
            // (10.) w = w - U(i,perm(k))*l_i
            for(j=pointer[current_row_col_U];j<pointer[current_row_col_U+1];j++){
                if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U*data[j];
            } // end for
        }   // (11.) end while
#ifdef VERBOSE
        time_9=clock();
        time_calc_L += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
#endif
        // (12.) sort and copy data to L
        // sort
        w.take_largest_elements_by_abs_value_with_threshold(norm_L,list_L,max_fill_in-1,threshold,0,n);
        if(pointer[k]+list_L.dimension()+1>reserved_memory){
            std::cerr<<"matrix_sparse::ILUCDP: memory reserved was insufficient. Returning 0x0 matrices and permutations of dimension 0."<<std::endl;
            perm.resize(0);
            permrows.resize(0);
            reformat(0,0,0,COLUMN);
            U.reformat(0,0,0,ROW);
            return false;
        }
        // copy data
        data[pointer[k]]=1.0;
        indices[pointer[k]]=selected_row;
        for(j=0;j<list_L.dimension();j++){
            pos = pointer[k]+j+1;
            data[pos] = w[list_L[j]]/U.data[U.pointer[k]];
            b = indices[pos] = list_L[j];
            h=startL[b];
            startL[b]=pos;
            linkL[pos]=h;
            colL[pos]=k;
            // begin updating fields for number elements of row of L
            if (b > bpr) {
                b = inverse_permrows[b];
                a = --pointer_num_el_row_L[++numb_el_row_L[b]];
                inverse_permrows.switch_index(permrows[a],permrows[b]);
                permrows.switch_index(a,b);
                numb_el_row_L.switch_entry(a,b);
            }
            // end updating fields
        } // end for j
        // sort permrows if necessary, i.e. if num_el_row_L increases at next iteration.
        if(pointer_num_el_row_L[numb_el_row_L[k]+1] == k+1){ 
            permrows.quicksort_with_inverse(inverse_permrows,pointer_num_el_row_L[numb_el_row_L[k]+1],pointer_num_el_row_L[numb_el_row_L[k]+2]-1);}
        // end sorting
        pointer[k+1]=pointer[k]+list_L.dimension()+1;
#ifdef VERBOSE
        time_0=clock();
        time_scu_L += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
#endif
    }  // (13.) end for k
#ifdef VERBOSE
    time_2 = clock();
#endif
    compress();
    U.compress();
#ifdef VERBOSE
    time_3=clock();
    time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef VERBOSE
    time_4=clock();
    time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
    std::cout<<"    ILUCDP-Times: "<<std::endl;
    std::cout<<"        initialization:                           "<<time_init<<std::endl;
    std::cout<<"        reading matrix:                           "<<time_read<<std::endl;
    std::cout<<"        sparse zero set:                          "<<time_zeroset<<std::endl;
    std::cout<<"        calculating L:                            "<<time_calc_L<<std::endl;
    std::cout<<"        calculating U:                            "<<time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info L: "<<time_scu_L<<std::endl;
    std::cout<<"        sorting, copying, updating access info U: "<<time_scu_U<<std::endl;
    std::cout<<"        compressing:                              "<<time_compress<<std::endl;
    std::cout<<"        resorting:                                "<<time_resort<<std::endl;
    std::cout<<"      Total times:"<<std::endl;
    std::cout<<"        calculations:                             "<<time_calc_L+time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info:   "<<time_scu_L+time_scu_U<<std::endl;
    std::cout<<"        other administration:                     "<<time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Grand total                                 "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
#endif
    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    return true;
}


template<class T>
bool matrix_sparse<T>::partialILUCDP(
        const matrix_sparse<T>& Arow, const matrix_sparse<T>& Acol,
        matrix_sparse<T>& Anew, const iluplusplus_precond_parameter& IP, bool force_finish,
        matrix_sparse<T>& U, vector_dense<T>& Dinv, index_list& perm, index_list& permrows,
        index_list& inverse_perm, index_list& inverse_permrows,Integer last_row_to_eliminate,
        Real threshold, Integer bp, Integer bpr, Integer epr, Integer& zero_pivots, Real& time_self,
        Real mem_factor, Real& total_memory_allocated, Real& total_memory_used)
{
    time_self = 0.0;
    total_memory_allocated = 0.0;
    Integer n = Acol.columns();
    if(!Arow.square_check()){
        std::cerr<<"matrix_sparse::partialILUCDP: argument matrix must be square. Returning 0x0 matrices."<<std::endl<<std::flush;
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        perm.resize(0);
        permrows.resize(0);
        inverse_perm.resize(0);
        inverse_permrows.resize(0);
        return false;
    }
    if(n==0){
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        perm.resize(0);
        permrows.resize(0);
        inverse_perm.resize(0);
        inverse_permrows.resize(0);
        return true;
    }
    clock_t time_begin, time_end;
    time_begin=clock();
    Integer max_fill_in;
    Integer bandwidth, bandwidth_L, bandwidth_U;
    Integer a,b,k,i,j,p,current_row_col_U,current_col_row_L;//help;
    Integer h,pos, selected_row;
    Integer pos_pivot=-1; // is set later
    T current_data_row_L,current_data_col_U;
    T val_larg_el = 0.0;
    zero_pivots=0;
    Real norm_U,norm; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    Real max_inv_piv=0.0;
    Real threshold_Schur_factor = std::exp(-IP.get_THRESHOLD_SHIFT_SCHUR()*std::log(10.0));
    Real post_fact_threshold;
    Real perm_tol = IP.get_perm_tol();
    bool end_level_now = false;  // indicates if next iteration in k-loop starts a new level, i.e. calculations of Schur complement begin.
    bool eliminate = true;       // indicates if standard elimination is being performed or Schur complement being calculated
    //bool pivoting = true;        // indicates if columns are pivoted in a particular step
    T pivot = 0.0;
    Integer k_Anew,n_Anew=0; // set later
    Integer reserved_memory_Anew=0; // will be set later
    T  xplus, xminus, yplus, yminus,vi;
    Real nuplus,numinus;
    Integer nplus, nminus,pk;
    bool use_improved_SCHUR = (IP.get_SCHUR_COMPLEMENT()>0);
    bool use_weightsLU = IP.get_USE_WEIGHTED_DROPPING() || IP.get_USE_WEIGHTED_DROPPING2();
    bool use_norm_row_U=false;
    Real weightL, weightU;
    Real move_level_parameter=0;
    Integer reserved_memory_L;
    Integer reserved_memory_U;
    Integer reserved_memory_droppedU;
#ifdef VERBOSE
    clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_8,time_9;
    Real time_init=0.0;
    Real time_read=0.0;
    Real time_calc_L=0.0;
    Real time_scu_L=0.0;  // sorting, copying, updating access information
    Real time_calc_U=0.0;
    Real time_scu_U=0.0;
    Real time_calc_Anew=0.0;
    Real time_scu_Anew=0.0;
    Real time_zeroset=0.0;
    Real time_compress=0.0;
    Real time_resort=0.0;
    Real time_dropping=0.0;
    time_0 = clock();
#endif
    if(IP.get_MAX_FILLIN_IS_INF())  max_fill_in = n;
    else max_fill_in = IP.get_fill_in();
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    if(IP.get_DROP_TYPE_L()==4||IP.get_DROP_TYPE_U()==4) bandwidth=Arow.bandwidth(); else bandwidth=0;
    switch (IP.get_DROP_TYPE_L()){
        case 3: bandwidth_L = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_L = bandwidth; break;
        default: bandwidth_L = 0;
    }
    switch (IP.get_DROP_TYPE_U()){
        case 3: bandwidth_U = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_U = bandwidth; break;
        default: bandwidth_U = 0;
    }
    if (threshold > 500.0) threshold=0.0;
    else threshold=std::exp(-threshold*std::log(10.0));
    if (perm_tol > 500.0) perm_tol=0.0;
    else perm_tol=std::exp(-perm_tol*std::log(10.0));
    if  (IP.get_POST_FACT_THRESHOLD() > 500.0) post_fact_threshold = 0.0; 
    else post_fact_threshold = threshold*std::exp(-IP.get_POST_FACT_THRESHOLD()*std::log(10.0));
    if(last_row_to_eliminate+1>n) last_row_to_eliminate = n-1;
    if(last_row_to_eliminate<0) last_row_to_eliminate = 0;
    if(epr<0)  epr=0;
    if(epr>=n) epr=n-1;
    if(bpr<0)  bpr=0;
    if(bpr>=n) bpr=n-1;
    reserved_memory_L = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Acol.non_zeroes()));
    reserved_memory_U = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Acol.non_zeroes()));
    reserved_memory_droppedU = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Acol.non_zeroes()));
#ifdef STATISTICS
    vector_dense<Integer> L_total,L_kept,U_total,U_kept;
    Real average_total,average_kept,average_prop, min_prop, max_prop, stand_dev_kept, stand_dev_prop, stand_dev_total;
    Integer min_total, max_total, min_kept, max_kept,help;
    Real sum1, sum2, sum3, prop;
#endif
    vector_sparse_dynamic<T> w,z;
    vector_dense<bool> non_pivot, unused_rows;
    vector_dense<Integer> numb_el_row_L, pointer_num_el_row_L;
    vector_dense<Real> norm_row_U;
    sorted_vector row_reorder_weight;
    index_list list_L, list_U;
    index_list rejected_L, rejected_U;
    matrix_sparse<T> droppedU;  // matrix containing dropped elements of U. Used to calculate an improved Schur complement.
    std::vector< std::queue<T> > droppedL_data;
    std::vector< std::queue<Integer> > droppedL_colindex;
    Real droppedL_data_memory = 0.0;
    Real droppedL_colindex_memory = 0.0;
    vector_dense<T> vxL,vyL,vxU,vyU,xL,yL,xU,yU;
    vector_dense<Real> weightsL,weightsU;

    // h=link[startU[i]]] points to second 2nd element, link[h] to next, etc.
    // rowU: row indices of elements of U.data.
    // startU[i] points to start of points to an index of data belonging to column i
    // h=link[startL[i]]] points to second 2nd element, link[h] to next, etc.
    // colL: column indices of elements of data.
    // startL[i] points to start of points to an index of data belonging to row i
    std::vector<Integer> linkU(reserved_memory_U), rowU(reserved_memory_U), startU(n);
    std::vector<Integer> linkL(reserved_memory_L), colL(reserved_memory_L), startL(n);

    Dinv.resize(n,1.0);
    perm.resize(n);
    permrows.resize(n);
    inverse_perm.resize(n);
    inverse_permrows.resize(n);
    w.resize(n); z.resize(n);
    non_pivot.resize(n,true);
    unused_rows.resize(n,true);
    numb_el_row_L.resize(n,0);
    pointer_num_el_row_L.resize(n+2,epr+1);
    U.reformat(n,n,reserved_memory_U,ROW);
    reformat(n,n,reserved_memory_L,COLUMN);
    if(IP.get_FINAL_ROW_CRIT() <= -1){ row_reorder_weight.resize(n); if(n>0) row_reorder_weight.remove(0);}
    if(IP.get_FINAL_ROW_CRIT() == -3 || IP.get_FINAL_ROW_CRIT() == -4) {use_norm_row_U=true; norm_row_U.resize(n,0.0);}
    if(IP.get_USE_INVERSE_DROPPING()){
        xL.resize(n,0); yL.resize(n,0); vxL.resize(n,0);vyL.resize(n,0);xU.resize(n,0);yU.resize(n,0);vxU.resize(n,0);vyU.resize(n,0);
    }
    if(use_weightsLU){
        weightsL.resize(n,IP.get_INIT_WEIGHTS_LU()); weightsU.resize(n,IP.get_INIT_WEIGHTS_LU());  // set equal to 1 for diagonal element
    }
    if(use_improved_SCHUR){
        droppedU.reformat(n,n,reserved_memory_droppedU,ROW);
        droppedL_data.resize(n);
        droppedL_colindex.resize(n);
    }
#ifdef STATISTICS
    L_total.resize(n,0); L_kept.resize(n,0); U_total.resize(n,0); U_kept.resize(n,0);
#endif
    pointer_num_el_row_L[0]=0;
    for(k=0;k<n;k++) startU[k]=-1;
    for(k=0;k<n;k++) startL[k]=-1;
    // stores dropped elements of L by rows
    // (1.) begin for k
#ifdef VERBOSE
    time_1 = clock();
    time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
#endif
    for(k=0;k<n;k++){
        if (IP.get_BEGIN_TOTAL_PIV() && k == bp){ perm_tol = 1.0;}// permute always
#ifdef VERBOSE
        time_2=clock();
#endif
        // (2.) initialize z
        selected_row = permrows[k];
        unused_rows[selected_row]=false;
        z.zero_reset();
#ifdef VERBOSE
        time_3=clock();
        time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
        // read row of A
        for(i=Arow.pointer[selected_row];i<Arow.pointer[selected_row+1];i++){
            if(non_pivot[Arow.indices[i]]) z[Arow.indices[i]] = Arow.data[i];
        }     // end for i
#ifdef VERBOSE
        time_4=clock();
        time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
#endif
        // (3.) begin while
        h=startL[selected_row]; // h=startL[permrows[k]];
        // do standard elimination
        while(h!=-1){
            current_col_row_L=colL[h];
            current_data_row_L=data[h];
            h=linkL[h];
            for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L/Dinv[current_col_row_L]*U.data[j];
                // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
            } // end for
            // use improved Schur?
            if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of L, small of U):
                for(j=droppedU.pointer[current_col_row_L];j<droppedU.pointer[current_col_row_L+1];j++){
                    if(non_pivot[droppedU.indices[j]]) z[droppedU.indices[j]] -= current_data_row_L/Dinv[current_col_row_L]*droppedU.data[j];
                    // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
                } // end for
            } //end if
        }   // (5.) end while
        if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of U, small of L):
            while(droppedL_data[k].size()>0){
                current_col_row_L = droppedL_colindex[k].front();  // read
                current_data_row_L = droppedL_data[k].front();
                for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                    if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L/Dinv[current_col_row_L]*U.data[j];
                    // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
                } // end for
                droppedL_colindex[k].pop();  // remove
                droppedL_data[k].pop();
            }  // end while
        } // end if
#ifdef VERBOSE
        time_5=clock();
        if(eliminate) time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
        else time_calc_Anew += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif

        /*
           if(eliminate){  // select potential pivot
           val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
           if (std::abs(val_larg_el*perm_tol)>std::abs(z[perm[k]]) && pos_pivot>=0){ 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           pivoting = false;
           pivot = z[perm[k]];
           }
           }
           */

        /*
           if(eliminate){  // select potential pivot
           val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
           if(non_pivot[selected_row]){  // not pivoting is with respect to the diagonal element of the selected row (if possible)
           if (std::abs(val_larg_el*perm_tol)>std::abs(z[selected_row]) && pos_pivot>=0){
           std::cout<<"*1 "; 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           std::cout<<"*2 "; 
           pivoting = false;
           pos_pivot = selected_row;
           pivot = z[pos_pivot];
           }
           } else {   // not pivoting is with respect to the perm(k)-th element if diagonal element has already been eliminated
           if (std::abs(val_larg_el*perm_tol)>std::abs(z[perm[k]]) && pos_pivot>=0){ 
           std::cout<<"*3 "; 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           std::cout<<"*4 "; 
           pivoting = false;
           pos_pivot = perm[k];
           pivot = z[pos_pivot];
           }
           }
           }
           */
        if(eliminate){  // select potential pivot
            val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
            if(non_pivot[selected_row]){  // not pivoting is with respect to the diagonal element of the selected row (if possible)
                if (std::abs(val_larg_el*perm_tol)>std::abs(z[selected_row]) && pos_pivot>=0 && IP.get_perm_tol() <= 500.0){
                    //pivoting = true;
                    pivot = val_larg_el;
                } else {
                    //pivoting = false;
                    pos_pivot = selected_row;
                    pivot = z[pos_pivot];
                }
            } else {   // pivot if possible... only if nothing else works, use corresponding column
                if ( (std::abs(val_larg_el)>0.0) && pos_pivot>=0){ 
                    //pivoting = true;
                    pivot = val_larg_el;
                } else {
                    //pivoting = false;
                    pos_pivot = perm[k];
                    pivot = z[pos_pivot];
                }
            }
        }

        /*

           if(eliminate){  // select potential pivot
           val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
           if(non_pivot[selected_row]){  // not pivoting is with respect to the diagonal element of the selected row (if possible)
           if (std::abs(val_larg_el*perm_tol)>std::abs(z[selected_row]) && pos_pivot>=0 && IP.get_perm_tol() <= 500.0){
           pivoting = true;
           pivot = val_larg_el;
           } else {
           pivoting = false;
           pos_pivot = selected_row;
           pivot = z[pos_pivot];
           }
           } else {   // not pivoting is with respect to the perm(k)-th element if diagonal element has already been eliminated
           if (std::abs(val_larg_el*perm_tol)>std::abs(z[perm[k]]) && pos_pivot>=0 && IP.get_perm_tol() <= 500.0){ 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           pivoting = false;
           pos_pivot = perm[k];
           pivot = z[pos_pivot];
           }
           }
           }

*/

        if(eliminate && !force_finish && !IP.get_EXTERNAL_FINAL_ROW() && k > IP.get_MIN_ELIM_FACTOR()*n && IP.get_SMALL_PIVOT_TERMINATES() && std::abs(pivot) < IP.get_MIN_PIVOT()){  // terminate level because pivot is too small.
            eliminate = false;
            end_level_now = true;
            threshold *= threshold_Schur_factor;
            last_row_to_eliminate = k-1;  // the current row will already be the first row of Anew
            n_Anew = n-k;
            reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Acol.non_zeroes()));
            Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
            if(use_improved_SCHUR){ 
                for(size_t p = 0; p < droppedL_data.size(); p++)
                    droppedL_data_memory += droppedL_data[p].size();
                for(size_t p = 0; p < droppedL_colindex.size(); p++)
                    droppedL_colindex_memory += droppedL_colindex[p].size();
                droppedL_data_memory *= sizeof(T);
                droppedL_colindex_memory *= sizeof(Integer);
            }
        }
        if(eliminate){  // select pivot scale z/U
            Dinv[k]=1.0/pivot;
            /*
               if (pivoting) {
               Dinv[k]=1.0/val_larg_el;
               } else {
               Dinv[k]=1.0/z[perm[k]]; pos_pivot=perm[k];
               }
               */
            z.scale(Dinv[k]);
            z[pos_pivot]=0.0; // eliminate pivot for sorting
            // update permutations
            p=inverse_perm[pos_pivot];
            inverse_perm.switch_index(perm[k],pos_pivot);
            perm.switch_index(k,p);
            non_pivot[pos_pivot]=false;
        }
        if(use_weightsLU){
            for(j=0;j<z.non_zeroes();j++) weightsU[z.get_pointer(j)] += std::abs(z.get_data(j));
        }
        // (8.) read w
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
#endif
        w.zero_reset();
#ifdef VERBOSE
        time_7 = clock();
        time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
#endif
        // read column of A
        if(eliminate){
            for(i=Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
                if(unused_rows[Acol.indices[i]])
                    w[Acol.indices[i]] = Acol.data[i];
            }     // end for i
#ifdef VERBOSE
            time_8=clock();
            time_read += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
#endif
            // (9.) begin while
            h=startU[perm[k]];
            while(h!=-1){
                current_row_col_U=rowU[h];
                current_data_col_U=U.data[h];
                h=linkU[h];
                // (10.) w = w - U(i,perm(k))*l_i
                for(j=pointer[current_row_col_U];j<pointer[current_row_col_U+1];j++){
                    //if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U*data[j];
                    if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U/Dinv[current_row_col_U]*data[j];
                } // end for
            }   // (11.) end while
#ifdef VERBOSE
            time_7 = time_9=clock();
            time_calc_L += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
#endif
        } // end if 
        w.scale(Dinv[k]);
        if(use_weightsLU){
            for(j=0;j<w.non_zeroes();j++){ 
                weightsL[w.get_pointer(j)] += std::abs(w.get_data(j));
            }
        }
#ifdef VERBOSE
        time_8 = clock();
        time_scu_L += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
#endif
        pk = perm[k];
        if(IP.get_USE_INVERSE_DROPPING() && eliminate){
            if(k==0){
                xU[pk]=1.0; yU[pk]=1.0;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)]=vxU[z.get_pointer(j)]=z.get_data(j);
            } else {
                // initialise
                xplus  =  1.0 - vxU[pk];
                xminus = -1.0 - vxU[pk];
                nplus  = 0;
                nminus = 0;
                yplus  =  1.0 - vyU[pk];
                yminus = -1.0 - vyU[pk];
                nuplus  = 0.0;
                numinus = 0.0;
                // do x_k
                for(j=0;j<z.non_zeroes();j++) nuplus  += std::abs(vxU[z.get_pointer(j)]+z.get_data(j)*xplus);
                for(j=0;j<z.non_zeroes();j++) numinus += std::abs(vxU[z.get_pointer(j)]+z.get_data(j)*xminus);
                if(nuplus > numinus) xU[pk] = xplus;
                else xU[pk] = xminus;
                for(j=0;j<z.non_zeroes();j++) vxU[z.get_pointer(j)] +=  z.get_data(j)*xU[pk];
                xU[pk]=max(std::abs(xplus),std::abs(xminus));
                // do y_k
                for(j=0;j<z.non_zeroes();j++){
                    vi=vyU[z.get_pointer(j)];
                    if(std::abs(vi+z.get_data(j)*yplus) > max(2.0*std::abs(vi),(Real)0.5)) nplus++;
                    if(max(2.0*std::abs(vi+z.get_data(j)*yplus),(Real) 0.5)<std::abs(vi)) nplus--;
                    if(std::abs(vi+z.get_data(j)*yminus) > max(2.0*std::abs(vi),(Real) 0.5)) nminus++;
                    if(max(2.0*std::abs(vi+z.get_data(j)*yminus),(Real) 0.5)<std::abs(vi)) nminus--;
                }
                if(nplus > nminus) yU[pk]=yplus;
                else yU[pk]= yminus;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)] += z.get_data(j)*yU[pk];
                yU[pk]=max(std::abs(yplus),std::abs(yminus));
            }
        }   // values for dropping are now in xU[pk],yU[pk]
#ifdef STATISTICS
        L_total[k]= w.non_zeroes(); 
        U_total[k]= z.non_zeroes();
#endif
        if(!eliminate){  // drop in Schur complement
            z.take_largest_elements_by_abs_value_with_threshold(norm_U,list_U,max_fill_in,threshold,0,n);
        } else { // drop in U
            weightU=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()){norm = z.norm2(); if(norm==0.0) norm=1e-16; weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP()/norm);} 
            if(IP.get_USE_STANDARD_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP2());  // drop if |w_i|<tau
            if(IP.get_USE_INVERSE_DROPPING())  weightU = IP.combine(weightU,IP.get_WEIGHT_INVERSE_DROP()*max(std::abs(xU[pk]),std::abs(yU[pk])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_WEIGHTED_DROP()*weightsU[pk]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP()*w.norm1());
            //if(USE_ERR_PROP_DROPPING) weightU = combine(weightU,WEIGHT_ERR_PROP_DROP*(w.norm_max()));
            //if(USE_ERR_PROP_DROPPING) weightU = combine(weightU,WEIGHT_ERR_PROP_DROP*(1.0+w.norm_max()));
            if(IP.get_USE_ERR_PROP_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP2()*w.norm1()/std::abs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_PIVOT_DROP()*std::abs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG()) weightU *= std::abs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG()){max_inv_piv = max(max_inv_piv,std::abs(Dinv[k])); weightU *= max_inv_piv;}
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,0,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightsU,weightU,max_fill_in-1,threshold,0,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    default: throw std::runtime_error("matrix_sparse::partialILUCDP: DROP_TYPE_U does not have permissible value.");
                }
            } else {
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,0,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,weightsU,weightU,max_fill_in-1,threshold,0,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    default: throw std::runtime_error("matrix_sparse::partialILUCDP: DROP_TYPE_U does not have permissible value.");
                }
            }
        }

#ifdef VERBOSE
        time_9 = clock();
        time_dropping += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef STATISTICS
        U_kept[k]= list_U.dimension();
        //if (U_total[k] != U_kept[k]){ std::cout<<"k = "<<k<<" U_kept"<<U_kept[k]<<" U_total "<<U_total[k]<<std::endl<<"Vector"<<std::endl; z.print_non_zeroes();}
#endif
        // update U or Anew
        if(eliminate){
            if(U.pointer[k]+list_U.dimension()+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+list_U.dimension()+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                linkU.resize(reserved_memory_U);
                rowU.resize(reserved_memory_U);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for U at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<U.pointer[k]+list_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            U.indices[U.pointer[k]]=pos_pivot;
            for(j=0;j<list_U.dimension();j++){
                pos=U.pointer[k]+j+1;
                U.data[pos]=z[list_U[list_U.dimension()-1-j]];
                U.indices[pos]=list_U[list_U.dimension()-1-j];
                if(use_norm_row_U) norm_row_U[k] += std::abs(U.data[pos]);
                h=startU[U.indices[pos]];
                startU[U.indices[pos]]=pos;
                linkU[pos]=h;
                rowU[pos]=k;
            }
            U.pointer[k+1]=U.pointer[k]+list_U.dimension()+1;
            if(pivot==0.0){
                zero_pivots++;
                Dinv[k]=1.0;
#ifdef VERBOSE
                std::cerr<<"matrix_sparse::partialILUCDP: Preconditioner does not exist (zero pivot). Setting diagonal to 1."<<std::endl;
#endif
            }
            if(use_improved_SCHUR){ // update droppedU
                if(droppedU.pointer[k]+rejected_U.dimension()>reserved_memory_droppedU){
                    reserved_memory_droppedU = 2*(droppedU.pointer[k]+rejected_U.dimension());
                    droppedU.enlarge_fields_keep_data(reserved_memory_droppedU);
                    // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for droppedU at position 1"<<std::endl;
                    // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<droppedU.pointer[k]+rejected_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                    // reformat(0,0,0,COLUMN);
                    // U.reformat(0,0,0,ROW);
                    // Dinv.resize_without_initialization(0);
                    // Anew.reformat(0,0,0,ROW);
                    // perm.resize(0);
                    // permrows.resize(0);
                    // inverse_perm.resize(0);
                    // inverse_permrows.resize(0);
                    // return false;
                }
                for(j=0;j<rejected_U.dimension();j++){
                    pos=droppedU.pointer[k]+j;
                    droppedU.data[pos]=z[rejected_U[j]];
                    droppedU.indices[pos]=rejected_U[j];
                }
                droppedU.pointer[k+1]=droppedU.pointer[k]+rejected_U.dimension();
            }  // end updating droppedU
        } else {
            k_Anew = k -last_row_to_eliminate-1;
            if(U.pointer[k]+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                linkU.resize(reserved_memory_U);
                rowU.resize(reserved_memory_U);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For U:    Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<U.pointer[k]+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            if(Anew.pointer[k_Anew]+list_U.dimension()>reserved_memory_Anew){
                reserved_memory_Anew = 2*(Anew.pointer[k_Anew]+list_U.dimension());
                Anew.enlarge_fields_keep_data(reserved_memory_Anew);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For Anew: Reserved memory for non-zero elements: "<<reserved_memory_Anew<<" Memory needed: "<<Anew.pointer[k_Anew]+list_U.dimension()<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            Dinv[k]=1.0;
            U.indices[U.pointer[k]]=perm[k];
            U.pointer[k+1]=U.pointer[k]+1;
            for(j=0;j<list_U.dimension();j++){
                pos=Anew.pointer[k_Anew]+j;
                Anew.data[pos]=z[list_U[list_U.dimension()-1-j]];
                Anew.indices[pos]=list_U[list_U.dimension()-1-j];
            }
            Anew.pointer[k_Anew+1]=Anew.pointer[k_Anew]+list_U.dimension();
        }
        // store positions of columns of U, but without pivot
        // update non-pivots.
#ifdef VERBOSE
        time_0=clock();
        if(eliminate) time_scu_U += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
        else time_scu_Anew += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
#endif
        // (12.) sort and copy data to L
        // sort
        pk = permrows[k];
        if(eliminate){

            if(IP.get_USE_INVERSE_DROPPING()){
                if(k==0){
                    xL[pk]=1.0; yL[pk]=1.0;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)]=vxL[w.get_pointer(j)]=w.get_data(j);
                } else {
                    // initialise
                    xplus  =  1.0 - vxL[pk];
                    xminus = -1.0 - vxL[pk];
                    nplus  = 0;
                    nminus = 0;
                    yplus  =  1.0 - vyL[pk];
                    yminus = -1.0 - vyL[pk];
                    nuplus  = 0.0;
                    numinus = 0.0;
                    // do x_k
                    for(j=0;j<w.non_zeroes();j++) nuplus  += std::abs(vxL[w.get_pointer(j)]+w.get_data(j)*xplus);
                    for(j=0;j<w.non_zeroes();j++) numinus += std::abs(vxL[w.get_pointer(j)]+w.get_data(j)*xminus);
                    if(nuplus > numinus) xL[pk] = xplus;
                    else xL[pk] = xminus;
                    for(j=0;j<w.non_zeroes();j++) vxL[w.get_pointer(j)] +=  w.get_data(j)*xL[pk];
                    xL[pk]=max(std::abs(xplus),std::abs(xminus));
                    // do y_k
                    for(j=0;j<w.non_zeroes();j++){
                        vi=vyL[w.get_pointer(j)];
                        if(std::abs(vi+w.get_data(j)*yplus) > max(2.0*std::abs(vi),(Real) 0.5)) nplus++;
                        if(max(2.0*std::abs(vi+w.get_data(j)*yplus),(Real) 0.5)<std::abs(vi)) nplus--;
                        if(std::abs(vi+w.get_data(j)*yminus) > max(2.0*std::abs(vi),(Real) 0.5)) nminus++;
                        if(max(2.0*std::abs(vi+w.get_data(j)*yminus),(Real) 0.5)<std::abs(vi)) nminus--;
                    }
                    if(nplus > nminus) yL[pk]=yplus;
                    else yL[pk]= yminus;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)] += w.get_data(j)*yL[pk];
                    yL[pk]=max(std::abs(yplus),std::abs(yminus));
                }  // values for dropping are now in xL[pk],yL[pk]
            }
            weightL=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()) {norm = w.norm2(); if(norm==0.0) norm=1e-16; weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP()/norm);}
            if(IP.get_USE_STANDARD_DROPPING2()) weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP2());
            if(IP.get_USE_INVERSE_DROPPING())  weightL = IP.combine(weightL,IP.get_WEIGHT_INVERSE_DROP()*max(std::abs(xL[pk]),std::abs(yL[pk])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_WEIGHTED_DROP()*weightsL[pk]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP()*z.norm1());
            if(IP.get_USE_ERR_PROP_DROPPING2())weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP2()*z.norm1()/std::abs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING())weightL = IP.combine(weightL,IP.get_WEIGHT_PIVOT_DROP()*std::abs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG())  weightL *= std::abs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG())  weightL *= max_inv_piv;
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in,threshold,0,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightsL,weightL,max_fill_in-1,threshold,0,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default: throw std::runtime_error("matrix_sparse::partialILUCDP: DROP_TYPE_L does not have permissible value.");
                }
            } else {
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in,threshold,0,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,weightsL,weightL,max_fill_in-1,threshold,0,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default: throw std::runtime_error("matrix_sparse::partialILUCDP: DROP_TYPE_L does not have permissible value.");
                }
            }
#ifdef VERBOSE
            time_1 = clock();
            time_dropping += (Real)(time_1-time_0)/(Real)CLOCKS_PER_SEC;
            time_0 = time_1;
#endif
#ifdef STATISTICS
            L_kept[k]= list_L.dimension();
            //if (L_total[k] != L_kept[k]){ std::cout<<"k = "<<k<<" L_kept"<<L_kept[k]<<" L_total "<<L_total[k]<<std::endl<<"Vector"<<std::endl; w.print_non_zeroes();}
#endif
            if(pointer[k]+list_L.dimension()+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+list_L.dimension()+1);
                enlarge_fields_keep_data(reserved_memory_L);
                linkL.resize(reserved_memory_L);
                colL.resize(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for L at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=selected_row;
            for(j=0;j<list_L.dimension();j++){
                pos = pointer[k]+j+1;
                //data[pos] = w[list_L[j]]/U.data[U.pointer[k]];
                //data[pos] = w[list_L[j]]*Dinv[k];
                data[pos] = w[list_L[j]]; // scaling has already been performed previously
                b = indices[pos] = list_L[j];
                h=startL[b];
                startL[b]=pos;
                linkL[pos]=h;
                colL[pos]=k;
                // begin updating fields for number elements of row of L
                if (b >= bpr && b <=  epr) {
                    if(IP.get_FINAL_ROW_CRIT() >= -1 && IP.get_FINAL_ROW_CRIT() < 11){  // resorting by the number of elements in row of L. Eliminating in increasing order.
                        b = inverse_permrows[b];
                        a = --pointer_num_el_row_L[++numb_el_row_L[b]];
                        inverse_permrows.switch_index(permrows[a],permrows[b]);
                        permrows.switch_index(a,b);
                        numb_el_row_L.switch_entry(a,b);
                    } else {   // resorting by 1-norm of number of elements in row of L. Eliminating in increasing order.
                        switch(IP.get_FINAL_ROW_CRIT()){
                            case -2: row_reorder_weight.add(b,std::abs(data[pos])); break;
                            case -3: row_reorder_weight.add(b,std::abs(data[pos])*norm_row_U[b]); break;
                            case -4: row_reorder_weight.add(b,std::abs(data[pos])/std::abs(Dinv[b])*norm_row_U[b]); break;
                            default: throw std::runtime_error("matrix_sparse::partialILUCDP: FINAL_ROW_CRIT has undefined value. Please set to correct value.");
                        }
                    }
                }
                // end updating fields
            } // end for j
            // sort permrows if necessary, i.e. if num_el_row_L increases at next iteration.
            if(IP.get_FINAL_ROW_CRIT() >= -1 && IP.get_FINAL_ROW_CRIT() < 11 && pointer_num_el_row_L[numb_el_row_L[k]+1] == k+1) 
                permrows.quicksort_with_inverse(inverse_permrows,pointer_num_el_row_L[numb_el_row_L[k]+1],pointer_num_el_row_L[numb_el_row_L[k]+2]-1);
            // end sorting
            if(IP.get_FINAL_ROW_CRIT() < -1 && k<n-1){  // still need to update permutations and inverse permutations for rows in this case
                b = row_reorder_weight.index_min();
                if(IP.get_USE_MAX_AS_MOVE()) move_level_parameter=row_reorder_weight.read_max(); else move_level_parameter=row_reorder_weight.read_min();
                row_reorder_weight.remove_min();
                p=inverse_permrows[b];
                inverse_permrows.switch_index(permrows[k+1],b); // k+1 the next loop
                permrows.switch_index(k+1,p);
            }
            pointer[k+1]=pointer[k]+list_L.dimension()+1;
            if(use_improved_SCHUR){ // update droppedU
                for(j=0;j<rejected_L.dimension();j++){
                    pos = rejected_L[j]; // row index of current element
                    droppedL_colindex[pos].push(k);  // store corresponding column index = k
                    droppedL_data[pos].push(w[pos]);  // store corresponding data element.

                }
            }  // end updating droppedU
        } else {  //  else branch of if(eliminate)
            if(pointer[k]+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+1);
                enlarge_fields_keep_data(reserved_memory_L);
                linkL.resize(reserved_memory_L);
                colL.resize(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for L at position 2"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=selected_row;
            pointer[k+1]=pointer[k]+1;
        }  //  end:  if(eliminate)
#ifdef VERBOSE
        time_1=clock();
        time_scu_L += (Real)(time_1-time_0)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate && IP.get_FINAL_ROW_CRIT() < 11 && !force_finish){
            if(IP.get_EXTERNAL_FINAL_ROW()){
                if (k >= last_row_to_eliminate && k >= IP.get_EXT_MIN_ELIM_FACTOR()*n){
                    end_level_now = true;
                }  // end if (last_row_to_eliminate == k)
            } else {
                if(k > IP.get_MIN_ELIM_FACTOR()*n)
                    switch(IP.get_FINAL_ROW_CRIT()){
                        case -3:  if(move_level_parameter > IP.get_MOVE_LEVEL_THRESHOLD()) end_level_now = true; break;
                        case -2:  if(move_level_parameter > IP.get_MOVE_LEVEL_THRESHOLD()) end_level_now = true; break;
                        case -1:  if( numb_el_row_L[k] > ((Real) IP.get_MOVE_LEVEL_FACTOR()*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  0:  if( numb_el_row_L[k] > ((Real) 0.5*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  1:  if( numb_el_row_L[k] > ((Real)     Acol.non_zeroes())/n) end_level_now = true; break;
                        case  2:  if( numb_el_row_L[k] > ((Real) 2.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  3:  if( numb_el_row_L[k] > ((Real) 4.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  4:  if( numb_el_row_L[k] > ((Real) 6.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  5:  if( numb_el_row_L[k] > 10) end_level_now = true; break;
                        case  6:  if( numb_el_row_L[k] > ((Real) 1.5*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  7:  if( z.norm2() > IP.get_ROW_U_MAX()) end_level_now = true; break;
                        case  8:  if( numb_el_row_L[k] > ((Real) 3.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  9:  if( numb_el_row_L[k] > ((Real) 1.2*Acol.non_zeroes())/n) end_level_now = true; break;
                                      //case 10:  end_level_now = true; break;
                        default:
                                      std::cerr<<"Please set FINAL_ROW_CRIT to a permissible value. Returning empty preconditioner."<<std::endl;
                                      reformat(0,0,0,COLUMN);
                                      U.reformat(0,0,0,ROW);
                                      Dinv.resize_without_initialization(0);
                                      Anew.reformat(0,0,0,ROW);
                                      perm.resize(0);
                                      permrows.resize(0);
                                      inverse_perm.resize(0);
                                      inverse_permrows.resize(0);
                                      return false;
                                      break;
                    }  // end switch
            }   // end if(EXTERNAL_ROW)
            if(end_level_now){
                eliminate = false;
                threshold *= threshold_Schur_factor;
                last_row_to_eliminate = k;
                n_Anew = n-k-1;
                //reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(mem_factor*Acol.non_zeroes()));
                reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Acol.non_zeroes()));
                Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
                Anew.pointer[0]=0;
                if(use_improved_SCHUR){ 
                    for(size_t p = 0; p < droppedL_data.size(); p++)
                        droppedL_data_memory += droppedL_data[p].size();
                    for(size_t p = 0; p < droppedL_colindex.size(); p++)
                        droppedL_colindex_memory += droppedL_colindex[p].size();
                    droppedL_data_memory *= sizeof(T);
                    droppedL_colindex_memory *= sizeof(Integer);
                }
            } // end if(end_level_now)
        } // end if (eliminate)
        if (eliminate && IP.get_REQUIRE_ZERO_SCHUR() && IP.get_REQ_ZERO_SCHUR_SIZE()>= n-k-1){
            //std::cout<<"setting schur complement: matrix has dimension "<<n<<" IP.get_REQ_ZERO_SCHUR_SIZE() = "<<IP.get_REQ_ZERO_SCHUR_SIZE()<<" k = "<<k<<std::endl; 
            eliminate = false;
            if(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                enlarge_fields_keep_data(reserved_memory_L);
                linkL.resize(reserved_memory_L);
                colL.resize(reserved_memory_L);
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            } 
            if(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                U.enlarge_fields_keep_data(reserved_memory_U);
                linkU.resize(reserved_memory_U);
                rowU.resize(reserved_memory_U);
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            } 
            n_Anew = IP.get_REQ_ZERO_SCHUR_SIZE();
            Anew.reformat(n_Anew,n_Anew,0,ROW);
            //std::cout<<"reformated Anew to dimension = "<<Anew.rows()<<std::endl; 
            for(j=k+1;j<n;j++){
                //std::cout<<"doing row/column of L/U j=  "<<j<<std::endl; 
                data[pointer[j]]=1.0;
                indices[pointer[j]]=permrows[j];
                pointer[j+1]=pointer[j]+1;
                U.data[U.pointer[j]]=1.0;
                Dinv[j]=1.0;
                U.indices[U.pointer[j]]=perm[j];
                U.pointer[j+1]=U.pointer[j]+1;
            }
            break;
        }  // end if
    }  // (13.) end for k
#ifdef VERBOSE
    time_2 = clock();
#endif
    Real memory_L_allocated = memory();
    Real memory_U_allocated = U.memory();
    Real memory_Anew_allocated = Anew.memory();
    compress();
    U.compress();
    if(eliminate) Anew.reformat(0,0,0,ROW); // if eliminated till end, then Anew is a 0x0 matrix.
    else {
        if(Anew.nnz>0){
            Anew.compress();
            // abuse linkU to store data
            linkU.clear();
            linkU.resize(Anew.nnz);
            // resort and shift indices to standard
            for(j=0;j<Anew.nnz;j++) linkU[j]=Anew.indices[j];
            for (i=0; i<Anew.rows(); i++)
                for(j=Anew.pointer[i]; j<Anew.pointer[i+1]; j++)
                    Anew.indices[j] = inverse_perm[linkU[j]]-last_row_to_eliminate-1;
            Anew.normal_order();
            Anew.number_columns=n_Anew; // originally, Anew has n columns
        } else {
            Anew.reformat(n_Anew,n_Anew,0,ROW);
        }
    }
    permute(permrows,ROW);
    U.permute(perm,COLUMN);
    if(IP.get_USE_POS_COMPRESS()){
        positional_compress(IP,post_fact_threshold);
        U.positional_compress(IP,post_fact_threshold);
    }
#ifdef VERBOSE
    time_3=clock();
    time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef VERBOSE
    time_4=clock();
    time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
    std::cout<<"    partialILUCDP-Times: "<<std::endl;
    std::cout<<"        initialization:                              "<<time_init<<std::endl;
    std::cout<<"        reading matrix:                              "<<time_read<<std::endl;
    std::cout<<"        sparse zero set:                             "<<time_zeroset<<std::endl;
    std::cout<<"        calculating L:                               "<<time_calc_L<<std::endl;
    std::cout<<"        calculating U:                               "<<time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info L:    "<<time_scu_L<<std::endl;
    std::cout<<"        sorting, copying, updating access info U:    "<<time_scu_U<<std::endl;
    std::cout<<"        calculating Anew:                            "<<time_calc_Anew<<std::endl;
    std::cout<<"        sorting, copying, updating access info Anew: "<<time_scu_Anew<<std::endl;
    std::cout<<"        dropping:                                    "<<time_dropping<<std::endl;
    std::cout<<"        compressing:                                 "<<time_compress<<std::endl;
    std::cout<<"        resorting:                                   "<<time_resort<<std::endl;
    std::cout<<"      Total times:"<<std::endl;
    std::cout<<"        calculations:                                "<<time_calc_L+time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info:      "<<time_scu_L+time_scu_U<<std::endl;
    std::cout<<"        other administration:                        "<<time_init+time_read+time_zeroset+time_compress+time_resort+time_dropping<<std::endl;
    std::cout<<"      Grand total                                    "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
#endif
    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    Real allocated_mem_lists_L = ((Real) (reserved_memory_L+1)) * sizeof(Integer);  // for linkL, colL
    Real allocated_mem_lists_U = ((Real) (reserved_memory_U+1)) * sizeof(Integer);  // for linkU, rowU
    Real used_mem_lists_L = ((Real) (nnz+1)) * sizeof(Integer);  // for linkL, colL
    Real used_mem_lists_U = ((Real) (U.nnz+1)) * sizeof(Integer);  // for linkU, rowU
    total_memory_allocated = w.memory() + z.memory() + non_pivot.memory() + unused_rows.memory() + numb_el_row_L.memory()+
        pointer_num_el_row_L.memory() + norm_row_U.memory() + row_reorder_weight.memory() + list_L.memory()+
        list_U.memory() + rejected_L.memory() + rejected_U.memory() + droppedU.memory() + vxL.memory() +
        vyL.memory() + vxU.memory() + vyU.memory() + xL.memory() + yL.memory() + xU.memory() + yU.memory() +
        memsize(startU) + memsize(startL) + Dinv.memory()+ perm.memory() + permrows.memory() + inverse_perm.memory()
        + inverse_permrows.memory() + droppedL_data_memory + droppedL_colindex_memory;
    total_memory_used = total_memory_allocated;
    total_memory_allocated += 2.0*allocated_mem_lists_U + 2.0* allocated_mem_lists_L + memory_U_allocated + memory_L_allocated + memory_Anew_allocated;
    total_memory_used += 2.0*used_mem_lists_U + 2.0* used_mem_lists_L + U.memory() + memory() + Anew.memory();
    //       Real mem_mat = Arow.memory();
    //       std::cout<<std::endl;
    //       std::cout<<"Relative Memory for a matrix of dimension "<<n<<":"<<std::endl;
    //       std::cout<<"w                    "<< w.memory()/mem_mat<<std::endl;
    //       std::cout<<"z                    "<< z.memory()/mem_mat<<std::endl;
    //       std::cout<<"non_pivot            "<< non_pivot.memory()/mem_mat<<std::endl;
    //       std::cout<<"unused_rows          "<< unused_rows.memory()/mem_mat<<std::endl;
    //       std::cout<<"numb_el_row_L        "<< numb_el_row_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"pointer_num_el_row_L "<< pointer_num_el_row_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"norm_row_U           "<< norm_row_U.memory()/mem_mat<<std::endl;
    //       std::cout<<"row_reorder_weight   "<< row_reorder_weight.memory()/mem_mat<<std::endl;
    //       std::cout<<"list                 "<< list_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"list_U               "<< list_U.memory()/mem_mat<<std::endl;
    //       std::cout<<"rejected_L           "<< rejected_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"rejected_U           "<< rejected_U.memory()/mem_mat<<std::endl;
    //       std::cout<<"droppedU             "<< droppedU.memory()/mem_mat<<std::endl;
    //       std::cout<<"vxL                  "<< vxL.memory()/mem_mat<<std::endl;
    //       std::cout<<"vyL                  "<< vyL.memory()/mem_mat<<std::endl;
    //       std::cout<<"vxU                  "<< vxU.memory()/mem_mat<<std::endl;
    //       std::cout<<"vyU                  "<< vyU.memory()/mem_mat<<std::endl;
    //       std::cout<<"xL                   "<< xL.memory()/mem_mat<<std::endl;
    //       std::cout<<"yL                   "<< yL.memory()/mem_mat<<std::endl;
    //       std::cout<<"xU                   "<< xU.memory()/mem_mat<<std::endl;
    //       std::cout<<"yU                   "<< yU.memory()/mem_mat<<std::endl;
    //       std::cout<<"linkU                "<< allocated_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"rowU                 "<< allocated_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"startU               "<< startU.memory()/mem_mat<<std::endl;
    //       std::cout<<"linkL                "<< allocated_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"colL                 "<< allocated_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"startL               "<< startL.memory()/mem_mat<<std::endl;
    //       std::cout<<"Dinv                 "<< Dinv.memory()/mem_mat<<std::endl;
    //       std::cout<<"perm                 "<< perm.memory()/mem_mat<<std::endl;
    //       std::cout<<"permrows             "<< permrows.memory()/mem_mat<<std::endl;
    //       std::cout<<"inverse_perm         "<< inverse_perm.memory()/mem_mat<<std::endl;
    //       std::cout<<"inverse_permrows     "<< inverse_permrows.memory()/mem_mat<<std::endl;
    //       std::cout<<"U                    "<< memory_U_allocated/mem_mat<<std::endl;
    //       std::cout<<"L                    "<< memory_L_allocated/mem_mat<<std::endl;
    //       std::cout<<"Anew                 "<< memory_Anew_allocated/mem_mat<<std::endl;
    //       std::cout<<"droppedL_data        "<< droppedL_data_memory/mem_mat<<std::endl;
    //       std::cout<<"droppedL_colindex    "<< droppedL_colindex_memory/mem_mat<<std::endl;
    //       std::cout<<"linkU (used)         "<< used_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"rowU  (used)         "<< used_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"linkL (used)         "<< used_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"colL  (used)         "<< used_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"U     (used)         "<< U.memory()/mem_mat<<std::endl;
    //       std::cout<<"L     (used)         "<< memory()/mem_mat<<std::endl;
    //       std::cout<<"Anew  (used)         "<< Anew.memory()/mem_mat<<std::endl;
    //       std::cout<<"total_memory         "<< total_memory_allocated/mem_mat<<std::endl;
    //       std::cout<<"total_memory (used)  "<< total_memory_used/mem_mat<<std::endl;
    //       std::cout<<std::endl;
#ifdef STATISTICS
    // Statistics for A
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Arow.row_density();
    for(k=0;k<n;k++){
        help =  Arow.pointer[k+1]-Arow.pointer[k];
        if (max_total < help) max_total = help;
        if (min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for A"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number: "<<average_total<<std::endl;
    std::cout<<"       Minimum Number     in Row:    "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Row:    "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Row:    "<<stand_dev_total<<std::endl;
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Acol.column_density();
    for(k=0;k<n;k++){
        help =  Acol.pointer[k+1]-Acol.pointer[k];
        if(max_total < help) max_total = help;
        if(min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<"       Minimum Number     in Column: "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Column: "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Column: "<<stand_dev_total<<std::endl;
    // Statistics for L
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total[k] == 0) prop = 1.0;
        else prop = ((Real) L_kept[k])/((Real) L_total[k]);
        if(max_total < L_total[k]) max_total =L_total[k];
        if(min_total > L_total[k]) min_total =L_total[k];
        if(max_kept < L_kept[k]) max_kept =L_kept[k];
        if(min_kept > L_kept[k]) min_kept =L_kept[k];
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += L_total[k];
        sum2 += L_kept[k];
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total[k] == 0) prop = 1.0;
        else prop = ((Real) L_kept[k])/((Real) L_total[k]);
        sum1 += (L_total[k]-average_total)*(L_total[k]-average_total);
        sum2 += (L_kept[k]-average_total)*(L_kept[k]-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for L"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;
    // Statistics for U
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total[k] == 0) prop = 1.0;
        else prop = ((Real) U_kept[k])/((Real) U_total[k]);
        if(max_total < U_total[k]) max_total =U_total[k];
        if(min_total > U_total[k]) min_total =U_total[k];
        if(max_kept < U_kept[k]) max_kept =U_kept[k];
        if(min_kept > U_kept[k]) min_kept =U_kept[k];
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += U_total[k];
        sum2 += U_kept[k];
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total[k] == 0) prop = 1.0;
        else prop = ((Real) U_kept[k])/((Real) U_total[k]);
        sum1 += (U_total[k]-average_total)*(U_total[k]-average_total);
        sum2 += (U_kept[k]-average_total)*(U_kept[k]-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for U"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;

#endif
    //std::cout<<"L"<<std::endl<<expand()<<std::endl;
    //std::cout<<"U"<<std::endl<<U.expand()<<std::endl;
    //std::cout<<"Anew"<<std::endl<<Anew.expand()<<std::endl;
    //std::cout<<"Dinv"<<std::endl<<Dinv<<std::endl;
    return true;
}

template<class T> bool matrix_sparse<T>::partialILUC(
        const matrix_sparse<T>& Arow, matrix_sparse<T>& Anew, const iluplusplus_precond_parameter& IP, bool force_finish, matrix_sparse<T>& U,
        vector_dense<T>& Dinv, Integer last_row_to_eliminate, Real threshold, Integer& zero_pivots,
        Real& time_self, Real mem_factor, Real& total_memory_allocated, Real& total_memory_used)
{
    total_memory_allocated = 0.0;
    time_self = 0.0;
    Integer n = Arow.columns();
    if(!Arow.square_check()){
        std::cerr<<"matrix_sparse::partialILUC: argument matrix must be square. Returning 0x0 matrices."<<std::endl<<std::flush;
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        return false;
    }
    if(n==0){
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        return true;
    }
    clock_t time_begin, time_end;
    time_begin=clock();
    Integer bandwidth, bandwidth_L, bandwidth_U;
    Integer i,j,k;//help;
    Integer h,pos;
    Integer max_fill_in;
    if(IP.get_MAX_FILLIN_IS_INF())  max_fill_in = n;
    else max_fill_in = IP.get_fill_in();
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    T pivot = 0.0;  // dummy initialization
    zero_pivots=0;
    Real norm_U,norm; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    Real max_inv_piv=0.0;
    if(IP.get_DROP_TYPE_L()==4||IP.get_DROP_TYPE_U()==4) bandwidth=Arow.bandwidth(); else bandwidth=0;
    switch (IP.get_DROP_TYPE_L()){
        case 3: bandwidth_L = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_L = bandwidth; break;
        default: bandwidth_L = 0;
    }
    switch (IP.get_DROP_TYPE_U()){
        case 3: bandwidth_U = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_U = bandwidth; break;
        default: bandwidth_U = 0;
    }
    if (threshold > 500.0) threshold=0.0;
    else threshold=std::exp(-threshold*std::log(10.0));
    Real threshold_Schur_factor = std::exp(-IP.get_THRESHOLD_SHIFT_SCHUR()*std::log(10.0));
    Real post_fact_threshold;
    if  (IP.get_POST_FACT_THRESHOLD() > 500.0) post_fact_threshold = 0.0; 
    else post_fact_threshold = threshold*std::exp(-IP.get_POST_FACT_THRESHOLD()*std::log(10.0));
    if(last_row_to_eliminate+1>n) last_row_to_eliminate = n-1;
    if(last_row_to_eliminate<0) last_row_to_eliminate = 0;
    bool use_improved_SCHUR = (IP.get_SCHUR_COMPLEMENT()>0);
    bool use_weightsLU = IP.get_USE_WEIGHTED_DROPPING() || IP.get_USE_WEIGHTED_DROPPING2();
    bool end_level_now = false;  // indicates if next iteration in k-loop starts a new level, i.e. calculations of Schur complement begin.
    bool eliminate = true;       // indicates if standard elimination is being performed or Schur complement being calculated
    Integer k_Anew,n_Anew=0; // set later
    Integer reserved_memory_Anew=0; // will be set later
    T  xplus, xminus, yplus, yminus,vi;
    Real nuplus,numinus;
    Integer nplus, nminus;
    Real weightL, weightU;
#ifdef VERBOSE
    clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_9;
    Real time_init=0.0;
    Real time_read=0.0;
    Real time_calc_L=0.0;
    Real time_scu_L=0.0;  // sorting, copying, updating access information
    Real time_calc_U=0.0;
    Real time_scu_U=0.0;
    Real time_zeroset=0.0;
    Real time_compress=0.0;
    Real time_resort=0.0;
    time_0 = clock();
#endif
    sorted_vector row_reorder_weight;
    bool use_norm_row_U = false;
    vector_dense<Real> norm_row_U;
    Real droppedL_data_memory = 0.0;
    Real droppedL_colindex_memory = 0.0;
    vector_dense<T> vxL,vyL,vxU,vyU,xL,yL,xU,yU;
    vector_dense<Real> weightsL,weightsU;
    index_list list_L, list_U;
    index_list rejected_L, rejected_U;
    Integer reserved_memory_L = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Arow.non_zeroes()));
    Integer reserved_memory_U = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Arow.non_zeroes()));
    Integer reserved_memory_droppedU = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Arow.non_zeroes()));

    std::vector<Integer> firstU(n), firstL(n), firstA(n), listA(n), headA(n), listU(n),listL(n);
    std::vector<Integer> firstUdropped, listUdropped; 

    std::vector< std::queue<T> > droppedL_data;
    std::vector< std::queue<Integer> > droppedL_colindex;
    matrix_sparse<T> droppedU;
    vector_sparse_dynamic<T> w,z;
#ifdef STATISTICS
    vector_dense<Integer> L_total,L_kept,U_total,U_kept;
    Real average_total,average_kept,average_prop, min_prop, max_prop, stand_dev_kept, stand_dev_prop, stand_dev_total;
    Integer min_total, max_total, min_kept, max_kept, help;
    Real sum1, sum2, sum3, prop;
#endif
    Dinv.resize(n,1.0);
    w.resize(n);
    z.resize(n);
    U.reformat(n,n,reserved_memory_U,ROW);
    reformat(n,n,reserved_memory_L,COLUMN);
    if(use_improved_SCHUR){
        firstUdropped.resize(n);
        listUdropped.resize(n);
    }
    if(IP.get_FINAL_ROW_CRIT() <= -1){
        row_reorder_weight.resize(n); 
        if(n>0) row_reorder_weight.remove(0);
    }
    if(IP.get_FINAL_ROW_CRIT() == -3 || IP.get_FINAL_ROW_CRIT() == -4) {
        use_norm_row_U=true; 
        norm_row_U.resize(n,0.0);
    }
    if(use_improved_SCHUR){
        droppedU.reformat(n,n,reserved_memory_droppedU,ROW);
        droppedL_data.resize(n);
        droppedL_colindex.resize(n);
    }
    if(IP.get_USE_INVERSE_DROPPING()){
        xL.resize(n,0);
        yL.resize(n,0);
        vxL.resize(n,0);
        vyL.resize(n,0);
        xU.resize(n,0);
        yU.resize(n,0);
        vxU.resize(n,0);
        vyU.resize(n,0);
    }
    if(use_weightsLU){
        weightsL.resize(n,IP.get_INIT_WEIGHTS_LU());
        weightsU.resize(n,IP.get_INIT_WEIGHTS_LU());  // set equal to 1 for diagonal element
    }
#ifdef STATISTICS
    L_total.resize(n,0); L_kept.resize(n,0); U_total.resize(n,0); U_kept.resize(n,0);
#endif
    initialize_sparse_matrix_fields(n,Arow.pointer,Arow.indices,listA,headA,firstA);
    initialize_triangular_fields(n,listL);
    initialize_triangular_fields(n,listU);
    if(use_improved_SCHUR){
        initialize_triangular_fields(n,listUdropped);
    }
    // (1.) begin for k
#ifdef VERBOSE
    time_1 = clock();
    time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
#endif
    for(k=0;k<n;k++){
#ifdef VERBOSE
        time_2=clock();
#endif
        // (2.) initialize z
        z.zero_reset();
#ifdef VERBOSE
        time_3=clock();
        time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
        // read row of A
        for(j=firstA[k];j<Arow.pointer[k+1];j++) z[Arow.indices[j]] = Arow.data[j];
#ifdef VERBOSE
        time_4=clock();
        time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
#endif
        // (3.) begin while
        h=listL[k];
        while(h!=-1){
            for(j=firstU[h];j<U.pointer[h+1];j++){
                z[U.indices[j]] -= data[firstL[h]]/Dinv[h]*U.data[j]; 
            }  // end for j

            if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of L, small of U):
                for(j=firstUdropped[h];j<droppedU.pointer[h+1];j++){
                    z[droppedU.indices[j]] -= data[firstL[h]]/Dinv[h]*droppedU.data[j];
                } // end for
            } //end if
            h=listL[h];
        } // end while (5.) in algorithm of Saad.
        if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of U, small of L):
            while(droppedL_data[k].size()>0){
                h = droppedL_colindex[k].front();  // read (h = corresponding column index in a fixed row of droppedL)
                for(j=firstU[h];j<U.pointer[h+1];j++){
                    z[U.indices[j]] -= droppedL_data[k].front()/Dinv[h]*U.data[j];
                    // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
                } // end for
                droppedL_colindex[k].pop();  // remove
                droppedL_data[k].pop();
            }  // end while
        } // end if
#ifdef VERBOSE
        time_5=clock();
        time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate && !force_finish && !IP.get_EXTERNAL_FINAL_ROW() && k > IP.get_MIN_ELIM_FACTOR()*n && IP.get_SMALL_PIVOT_TERMINATES() && fabs(z[k]) < IP.get_MIN_PIVOT()){  // terminate level because pivot is too small.
            eliminate = false;
            end_level_now = true;
            threshold *= threshold_Schur_factor;
            last_row_to_eliminate = k-1;  // the current row will already be the first row of Anew
            n_Anew = n-k;
            //reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(mem_factor*Arow.non_zeroes()));
            reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Arow.non_zeroes()));
            Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
            if(use_improved_SCHUR){ 
                for(size_t p = 0; p < droppedL_data.size(); p++)
                    droppedL_data_memory += droppedL_data[p].size();
                for(size_t p = 0; p < droppedL_colindex.size(); p++)
                    droppedL_colindex_memory += droppedL_colindex[p].size();
                droppedL_data_memory *= sizeof(T);
                droppedL_colindex_memory *= sizeof(Integer);
            }
        }
        if(eliminate){  // select pivot scale z/U
            pivot = z[k];
            Dinv[k]=1.0/z[k];
            z.scale(Dinv[k]);
            z[k]=0.0; // eliminate pivot for sorting
        }
        if(use_weightsLU){
            for(j=0;j<z.non_zeroes();j++) weightsU[z.get_pointer(j)] += fabs(z.get_data(j));
        }
        // (8.) read w
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif
        w.zero_reset();
#ifdef VERBOSE
        time_7=clock();
        time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate){
            // read column of A
            h=headA[k];
            while(h!=-1){
                if(h>k) w[h]=Arow.data[firstA[h]];
                h=listA[h];
            }
            // end while
            h=listU[k];
            while(h!=-1){
                // h is current row index of k-th column of U
                for(j=firstL[h];j<pointer[h+1];j++){
                    // (8.) in the algorithm of Saad.
                    w[indices[j]] -= U.data[firstU[h]]/Dinv[h]*data[j];
                }  // end for j
                h=listU[h];
            } // end while (9.) in algorithm of Saad.
        } // end if 
        w.scale(Dinv[k]);
#ifdef VERBOSE
        time_9=clock();
        time_calc_L += (Real)(time_9-time_7)/(Real)CLOCKS_PER_SEC;
#endif
        if(use_weightsLU){
            for(j=0;j<w.non_zeroes();j++){ 
                //weightsL[w.get_pointer(j)] = max(weightsL[w.get_pointer(j)],fabs(w.get_data(j)));
                weightsL[w.get_pointer(j)] += fabs(w.get_data(j));
            }
        }
        if(IP.get_USE_INVERSE_DROPPING() && eliminate){
            if(k==0){
                xU[k]=1.0; yU[k]=1.0;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)]=vxU[z.get_pointer(j)]=z.get_data(j);
            } else {
                // initialise
                xplus  =  1.0 - vxU[k];
                xminus = -1.0 - vxU[k];
                nplus  = 0;
                nminus = 0;
                yplus  =  1.0 - vyU[k];
                yminus = -1.0 - vyU[k];
                nuplus  = 0.0;
                numinus = 0.0;
                // do x_k
                for(j=0;j<z.non_zeroes();j++) nuplus  += fabs(vxU[z.get_pointer(j)]+z.get_data(j)*xplus);
                for(j=0;j<z.non_zeroes();j++) numinus += fabs(vxU[z.get_pointer(j)]+z.get_data(j)*xminus);
                if(nuplus > numinus) xU[k] = xplus;
                else xU[k] = xminus;
                for(j=0;j<z.non_zeroes();j++) vxU[z.get_pointer(j)] +=  z.get_data(j)*xU[k];
                xU[k]=max(fabs(xplus),fabs(xminus));
                // do y_k
                for(j=0;j<z.non_zeroes();j++){
                    vi=vyU[z.get_pointer(j)];
                    if(fabs(vi+z.get_data(j)*yplus) > max(2.0*fabs(vi),(Real)0.5)) nplus++;
                    if(max(2.0*fabs(vi+z.get_data(j)*yplus),(Real) 0.5)<fabs(vi)) nplus--;
                    if(fabs(vi+z.get_data(j)*yminus) > max(2.0*fabs(vi),(Real) 0.5)) nminus++;
                    if(max(2.0*fabs(vi+z.get_data(j)*yminus),(Real) 0.5)<fabs(vi)) nminus--;
                }
                if(nplus > nminus) yU[k]=yplus;
                else yU[k]= yminus;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)] += z.get_data(j)*yU[k];
                yU[k]=max(fabs(yplus),fabs(yminus));
            }
        }   // values for dropping are now in xU[k],yU[k]
#ifdef STATISTICS
        L_total[k]= w.non_zeroes(); 
        U_total[k]= z.non_zeroes();
#endif
        if(!eliminate){
            z.take_largest_elements_by_abs_value_with_threshold(norm_U,list_U,max_fill_in,threshold,last_row_to_eliminate+1,n);
        } else {
            weightU=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()){norm = z.norm2(); if(norm==0.0) norm=1e-16; weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP()/norm);} 
            if(IP.get_USE_STANDARD_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP2());  // drop if |w_i|<tau
            if(IP.get_USE_INVERSE_DROPPING())  weightU = IP.combine(weightU,IP.get_WEIGHT_INVERSE_DROP()*max(fabs(xU[k]),fabs(yU[k])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_WEIGHTED_DROP()*weightsU[k]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP()*w.norm1());
            if(IP.get_USE_ERR_PROP_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP2()*w.norm1()/fabs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_PIVOT_DROP()*fabs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG()) weightU *= fabs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG()){max_inv_piv = max(max_inv_piv,fabs(Dinv[k])); weightU *= max_inv_piv;}
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k+1,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightsU,weightU,max_fill_in-1,threshold,k+1,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_U does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            } else {
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k+1,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,weightsU,weightU,max_fill_in-1,threshold,k+1,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_U does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            }
        }
#ifdef STATISTICS
        U_kept[k]= list_U.dimension();
#endif
        // update U or Anew
        if(eliminate){
            if(U.pointer[k]+list_U.dimension()+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+list_U.dimension()+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for U at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<U.pointer[k]+list_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            U.indices[U.pointer[k]]=k;
            for(j=0;j<list_U.dimension();j++){
                pos=U.pointer[k]+j+1;
                U.data[pos]=z[list_U[j]];
                U.indices[pos]=list_U[j];
                if(use_norm_row_U) norm_row_U[k] += fabs(U.data[pos]);
                //h=startU[U.indices[pos]];
                //startU[U.indices[pos]]=pos;
                //linkU[pos]=h;
                //rowU[pos]=k;
            }
            U.pointer[k+1]=U.pointer[k]+list_U.dimension()+1;
            if(pivot == 0.0){
                zero_pivots++;
                Dinv[k]=1.0;
#ifdef VERBOSE
                std::cerr<<"matrix_sparse::partialILUC: Preconditioner does not exist (zero pivot). Setting diagonal to 1."<<std::endl;
#endif
            }
            if(use_improved_SCHUR){ // update droppedU
                if(droppedU.pointer[k]+rejected_U.dimension()>reserved_memory_droppedU){
                    reserved_memory_droppedU = 2*(droppedU.pointer[k]+rejected_U.dimension());
                    droppedU.enlarge_fields_keep_data(reserved_memory_droppedU);
                    // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for droppedU at position 1"<<std::endl;
                    // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<droppedU.pointer[k]+rejected_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                    // reformat(0,0,0,COLUMN);
                    // U.reformat(0,0,0,ROW);
                    // Dinv.resize_without_initialization(0);
                    // Anew.reformat(0,0,0,ROW);
                    // return false;
                }
                for(j=0;j<rejected_U.dimension();j++){
                    pos=droppedU.pointer[k]+j;
                    droppedU.data[pos]=z[rejected_U[j]];
                    droppedU.indices[pos]=rejected_U[j];
                }
                droppedU.pointer[k+1]=droppedU.pointer[k]+rejected_U.dimension();
            }  // end updating droppedU
        } else {
            k_Anew = k -last_row_to_eliminate-1;
            if(U.pointer[k]+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For Anew: Reserved memory for non-zero elements: "<<reserved_memory_Anew<<" Memory needed: "<<Anew.pointer[k_Anew]+list_U.dimension()<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            if(Anew.pointer[k_Anew]+list_U.dimension()>reserved_memory_Anew){
                reserved_memory_Anew = 2*(Anew.pointer[k_Anew]+list_U.dimension());
                Anew.enlarge_fields_keep_data(reserved_memory_Anew);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For Anew: Reserved memory for non-zero elements: "<<reserved_memory_Anew<<" Memory needed: "<<Anew.pointer[k_Anew]+list_U.dimension()<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            Dinv[k]=1.0;
            U.indices[U.pointer[k]]=k;
            U.pointer[k+1]=U.pointer[k]+1;
            for(j=0;j<list_U.dimension();j++){
                pos=Anew.pointer[k_Anew]+j;
                Anew.data[pos]=z[list_U[j]];
                Anew.indices[pos]=list_U[j];
            }
            Anew.pointer[k_Anew+1]=Anew.pointer[k_Anew]+list_U.dimension();
        }
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
#endif
        // (12.) sort and copy data to L
        if(eliminate){
            if(IP.get_USE_INVERSE_DROPPING()){
                if(k==0){
                    xL[k]=1.0; yL[k]=1.0;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)]=vxL[w.get_pointer(j)]=w.get_data(j);
                } else {
                    // initialise
                    xplus  =  1.0 - vxL[k];
                    xminus = -1.0 - vxL[k];
                    nplus  = 0;
                    nminus = 0;
                    yplus  =  1.0 - vyL[k];
                    yminus = -1.0 - vyL[k];
                    nuplus  = 0.0;
                    numinus = 0.0;
                    // do x_k
                    for(j=0;j<w.non_zeroes();j++) nuplus  += fabs(vxL[w.get_pointer(j)]+w.get_data(j)*xplus);
                    for(j=0;j<w.non_zeroes();j++) numinus += fabs(vxL[w.get_pointer(j)]+w.get_data(j)*xminus);
                    if(nuplus > numinus) xL[k] = xplus;
                    else xL[k] = xminus;
                    for(j=0;j<w.non_zeroes();j++) vxL[w.get_pointer(j)] +=  w.get_data(j)*xL[k];
                    xL[k]=max(fabs(xplus),fabs(xminus));
                    // do y_k
                    for(j=0;j<w.non_zeroes();j++){
                        vi=vyL[w.get_pointer(j)];
                        if(fabs(vi+w.get_data(j)*yplus) > max(2.0*fabs(vi),(Real) 0.5)) nplus++;
                        if(max(2.0*fabs(vi+w.get_data(j)*yplus),(Real) 0.5)<fabs(vi)) nplus--;
                        if(fabs(vi+w.get_data(j)*yminus) > max(2.0*fabs(vi),(Real) 0.5)) nminus++;
                        if(max(2.0*fabs(vi+w.get_data(j)*yminus),(Real) 0.5)<fabs(vi)) nminus--;
                    }
                    if(nplus > nminus) yL[k]=yplus;
                    else yL[k]= yminus;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)] += w.get_data(j)*yL[k];
                    yL[k]=max(fabs(yplus),fabs(yminus));
                }  // values for dropping are now in xL[k],yL[k]
            }
            weightL=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()) {norm = w.norm2(); if(norm==0.0) norm=1e-16; weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP()/norm);}
            if(IP.get_USE_STANDARD_DROPPING2()) weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP2());
            if(IP.get_USE_INVERSE_DROPPING())  weightL = IP.combine(weightL,IP.get_WEIGHT_INVERSE_DROP()*max(fabs(xL[k]),fabs(yL[k])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_WEIGHTED_DROP()*weightsL[k]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP()*z.norm1());
            if(IP.get_USE_ERR_PROP_DROPPING2())weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP2()*z.norm1()/fabs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING())weightL = IP.combine(weightL,IP.get_WEIGHT_PIVOT_DROP()*fabs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG())  weightL *= fabs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG())  weightL *= max_inv_piv;
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightsL,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_L does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            } else {
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,weightsL,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_L does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            }
#ifdef STATISTICS
            L_kept[k]= list_L.dimension();
#endif
            if(pointer[k]+list_L.dimension()+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+list_L.dimension()+1);
                enlarge_fields_keep_data(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for L at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=k;
            for(j=0;j<list_L.dimension();j++){
                pos = pointer[k]+j+1;
                data[pos] = w[list_L[j]]; // scaling has already been performed previously
                indices[pos] = list_L[j];
            } // end for j
            pointer[k+1]=pointer[k]+list_L.dimension()+1;
            if(use_improved_SCHUR){ // update droppedL
                for(j=0;j<rejected_L.dimension();j++){
                    pos = rejected_L[j]; // row index of current element
                    droppedL_colindex[pos].push(k);  // store corresponding column index = k
                    droppedL_data[pos].push(w[pos]);  // store corresponding data element.
                }
            }  // end updating droppedL
        } else {  //  else branch of if(eliminate)
            if(pointer[k]+list_L.dimension()+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+list_L.dimension()+1);
                enlarge_fields_keep_data(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for L at position 2"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=k;
            pointer[k+1]=pointer[k]+1;
        }  //  end:  if(eliminate)

        if(eliminate){
            update_sparse_matrix_fields(k, Arow.pointer,Arow.indices,listA,headA,firstA);
            update_triangular_fields(k, U.pointer,U.indices,listU,firstU);
            if(use_improved_SCHUR)
                update_triangular_fields(k, droppedU.pointer,droppedU.indices,listUdropped,firstUdropped);
        }
        update_triangular_fields(k, pointer,indices,listL,firstL);
#ifdef VERBOSE
        time_0=clock();
        time_scu_L += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate && IP.get_FINAL_ROW_CRIT() < 11 && !force_finish){
            if(IP.get_EXTERNAL_FINAL_ROW()){
                if (k >= last_row_to_eliminate && k >= IP.get_EXT_MIN_ELIM_FACTOR()*n ){
                    end_level_now = true;
                }  // end if (last_row_to_eliminate == k)
            }
            if(end_level_now){
                eliminate = false;
                threshold *= threshold_Schur_factor;
                last_row_to_eliminate = k;
                n_Anew = n-k-1;
                //reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(mem_factor*Arow.non_zeroes()));
                reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Arow.non_zeroes()));
                Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
                if(use_improved_SCHUR){ 
                    for(size_t p = 0; p < droppedL_data.size(); p++)
                        droppedL_data_memory += droppedL_data[p].size();
                    for(size_t p = 0; p < droppedL_colindex.size(); p++)
                        droppedL_colindex_memory += droppedL_colindex[p].size();
                    droppedL_data_memory *= sizeof(T);
                    droppedL_colindex_memory *= sizeof(Integer);
                }
            } // end if(end_level_now)
        } // end if (eliminate)
        if (eliminate && IP.get_REQUIRE_ZERO_SCHUR() && IP.get_REQ_ZERO_SCHUR_SIZE()>= n-k-1){
            eliminate = false;
            if(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                enlarge_fields_keep_data(reserved_memory_L);
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            } 
            if(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                U.enlarge_fields_keep_data(reserved_memory_U);
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            } 
            n_Anew = IP.get_REQ_ZERO_SCHUR_SIZE();
            Anew.reformat(n_Anew,n_Anew,0,ROW);
            for(j=k+1;j<n;j++){
                data[pointer[j]]=1.0;
                indices[pointer[j]]=j;
                pointer[j+1]=pointer[j]+1;
                U.data[U.pointer[j]]=1.0;
                Dinv[j]=1.0;
                U.indices[U.pointer[j]]=j;
                U.pointer[j+1]=U.pointer[j]+1;
            }
            break;
        }  // end if
    }  // (13.) end for k
#ifdef VERBOSE
    time_2 = clock();
#endif
    Real memory_L_allocated = memory();
    Real memory_U_allocated = U.memory();
    Real memory_Anew_allocated = Anew.memory();
    compress();
    U.compress();
    if(eliminate) Anew.reformat(0,0,0,ROW); // if eliminated till end, then Anew is a 0x0 matrix.
    else {
        if(Anew.nnz>0){
            Anew.compress();
            // resort and shift indices to standard
            for (i=0; i<Anew.rows(); i++)
                for(j=Anew.pointer[i]; j<Anew.pointer[i+1]; j++)
                    Anew.indices[j] -= last_row_to_eliminate+1;
            Anew.normal_order();
            Anew.number_columns=n_Anew; // originally, Anew has n columns
        } else {
            Anew.reformat(n_Anew,n_Anew,0,ROW);
        }
    }
    if(IP.get_USE_POS_COMPRESS()){
        positional_compress(IP,post_fact_threshold);
        U.positional_compress(IP,post_fact_threshold);
    }
#ifdef VERBOSE
    time_3=clock();
    time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef VERBOSE
    time_4=clock();
    time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
    std::cout<<"    partialILUC-Times: "<<std::endl;
    std::cout<<"        initialization:                           "<<time_init<<std::endl;
    std::cout<<"        reading matrix:                           "<<time_read<<std::endl;
    std::cout<<"        sparse zero set:                          "<<time_zeroset<<std::endl;
    std::cout<<"        calculating L:                            "<<time_calc_L<<std::endl;
    std::cout<<"        calculating U:                            "<<time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info L: "<<time_scu_L<<std::endl;
    std::cout<<"        sorting, copying, updating access info U: "<<time_scu_U<<std::endl;
    std::cout<<"        compressing:                              "<<time_compress<<std::endl;
    std::cout<<"        resorting:                                "<<time_resort<<std::endl;
    std::cout<<"      Total times:"<<std::endl;
    std::cout<<"        calculations:                             "<<time_calc_L+time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info:   "<<time_scu_L+time_scu_U<<std::endl;
    std::cout<<"        other administration:                     "<<time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Grand total                                 "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
#endif
    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    total_memory_allocated = w.memory() + z.memory() +  norm_row_U.memory() + row_reorder_weight.memory() + list_L.memory() +
        memsize(firstU) + memsize(firstL) + memsize(firstA) + memsize(listA) + memsize(headA) + memsize(listU) +
        memsize(listL) + list_U.memory() + rejected_L.memory() + rejected_U.memory() + droppedU.memory() + vxL.memory() +
        vyL.memory() + vxU.memory() + vyU.memory() + xL.memory() + yL.memory() + xU.memory() + yU.memory() +
        Dinv.memory() + droppedL_data_memory + droppedL_colindex_memory;
    total_memory_used = total_memory_allocated;
    total_memory_allocated += memory_U_allocated + memory_L_allocated + memory_Anew_allocated;
    total_memory_used +=  U.memory() + memory() + Anew.memory();
#ifdef STATISTICS
    // Statistics for A
    matrix_sparse<T> Acol;
    Acol.change_orientation_of_data(Arow);
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Arow.row_density();
    for(k=0;k<n;k++){
        help =  Arow.pointer[k+1]-Arow.pointer[k];
        if (max_total < help) max_total = help;
        if (min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for A"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number: "<<average_total<<std::endl;
    std::cout<<"       Minimum Number     in Row:    "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Row:    "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Row:    "<<stand_dev_total<<std::endl;
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Acol.column_density();
    for(k=0;k<n;k++){
        help =  Acol.pointer[k+1]-Acol.pointer[k];
        if(max_total < help) max_total = help;
        if(min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<"       Minimum Number     in Column: "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Column: "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Column: "<<stand_dev_total<<std::endl;
    // Statistics for L
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total[k] == 0) prop = 1.0;
        else prop = ((Real) L_kept[k])/((Real) L_total[k]);
        if(max_total < L_total[k]) max_total =L_total[k];
        if(min_total > L_total[k]) min_total =L_total[k];
        if(max_kept < L_kept[k]) max_kept =L_kept[k];
        if(min_kept > L_kept[k]) min_kept =L_kept[k];
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += L_total[k];
        sum2 += L_kept[k];
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total[k] == 0) prop = 1.0;
        else prop = ((Real) L_kept[k])/((Real) L_total[k]);
        sum1 += (L_total[k]-average_total)*(L_total[k]-average_total);
        sum2 += (L_kept[k]-average_total)*(L_kept[k]-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for L"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;
    // Statistics for U
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total[k] == 0) prop = 1.0;
        else prop = ((Real) U_kept[k])/((Real) U_total[k]);
        if(max_total < U_total[k]) max_total =U_total[k];
        if(min_total > U_total[k]) min_total =U_total[k];
        if(max_kept < U_kept[k]) max_kept =U_kept[k];
        if(min_kept > U_kept[k]) min_kept =U_kept[k];
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += U_total[k];
        sum2 += U_kept[k];
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total[k] == 0) prop = 1.0;
        else prop = ((Real) U_kept[k])/((Real) U_total[k]);
        sum1 += (U_total[k]-average_total)*(U_total[k]-average_total);
        sum2 += (U_kept[k]-average_total)*(U_kept[k]-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for U"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;

#endif
    return true;
}


template<class T>
bool matrix_sparse<T>::preprocessed_partialILUCDP(
        const iluplusplus_precond_parameter& IP, bool force_finish, const matrix_sparse<T>& A,
        matrix_sparse<T>& Acoarse, matrix_sparse<T>& U, vector_dense<T>& Dinv,
        index_list& permutation_rows, index_list& permutation_columns, index_list& inverse_permutation_rows,
        index_list& inverse_permutation_columns, vector_dense<T>& D_l, vector_dense<T>& D_r,
        Integer max_fill_in, Real threshold, Real perm_tol, Integer& zero_pivots, Real& setup_time,
        Real mem_factor, Real& total_memory_allocated, Real& total_memory_used)
{
    bool use_ILUC;
    if( (IP.get_PERMUTE_ROWS() == 0 || (IP.get_PERMUTE_ROWS() == 1 && !IP.get_EXTERNAL_FINAL_ROW()))
            && (!IP.get_BEGIN_TOTAL_PIV() || (IP.get_BEGIN_TOTAL_PIV() && IP.get_TOTAL_PIV() == 0) )
            && IP.get_perm_tol() > 500.0)
        use_ILUC = true;
    else
        use_ILUC = false;

    clock_t time_1,time_2;
    time_1 = clock();
    matrix_sparse<T> Arow2,Acol;
    bool factorization_exists;
    index_list pr1, pr2, ipr1,ipr2,pc1,pc2,ipc1,ipc2;
    Real partial_setup_time;
    setup_time = 0.0;
    Integer last_row_to_eliminate,bp,bpr,epr,end_PQ;
    if(A.orient() == ROW){
        Arow2 = A;
        end_PQ = Arow2.preprocess(IP,pr1,pc1,ipr1,ipc1,D_l,D_r);
    } else {
        Arow2.change_orientation_of_data(A);
        end_PQ = Arow2.preprocess(IP,pr1,pc1,ipr1,ipc1,D_l,D_r);
    }
    if (force_finish) {
        last_row_to_eliminate = Arow2.rows()-1;
    } else {
        if (IP.get_EXTERNAL_FINAL_ROW()) last_row_to_eliminate = end_PQ-1;
        else last_row_to_eliminate = (Arow2.rows()-1)/2;
    }
    switch (IP.get_PERMUTE_ROWS()) {
        case 0:  bpr = 0; epr = 0; break;
        case 1:  if(IP.get_EXTERNAL_FINAL_ROW() && (!force_finish)){bpr = 0; epr = 0;} else {bpr = end_PQ; epr = Arow2.rows()-1;} break;
        case 2:  if(force_finish){bpr = 0; epr = Arow2.rows()-1;} else {bpr = 0; epr = last_row_to_eliminate;} break;
        case 3:  bpr = 0; epr = Arow2.rows()-1; break;
        default: throw std::runtime_error("matrix_sparse::preprocessed_partialILUCDP::choose permissible value for PERMUTE_ROWS!");
    }
    switch (IP.get_TOTAL_PIV()) {
        case 0:  bp = Acol.rows(); break;
        case 1:  if(force_finish){bp = end_PQ;} else {bp = last_row_to_eliminate+1;} break;
        case 2:  bp = 0;  break;
        default: throw std::runtime_error("matrix_sparse::preprocessed_partialILUCDP::choose permissible value for TOTAL_PIV!");
    }
#ifdef INFO
    std::cout<<std::endl;
    std::cout<<"  ** matrix statistics:"<<std::endl;
    std::cout<<"     n                      = "<<Arow2.rows()<<std::endl;
    std::cout<<"     nnz                    = "<<Arow2.actual_non_zeroes()<<std::endl;
    std::cout<<"     density                = "<<Arow2.row_density()<<std::endl;
    std::cout<<"  ** factorization parameters:"<<std::endl;
    std::cout<<"     max. numb. nnz/row p   = "<<max_fill_in<<std::endl;
    std::cout<<"     tau                    = "<<threshold<<std::endl;
    std::cout<<"     perm tolerance         = "<<perm_tol<<std::endl;
    std::cout<<"     begin permuting rows   = "<<bpr<<std::endl;
    std::cout<<"     end   permuting rows   = "<<epr<<std::endl;
    if(IP.get_EXTERNAL_FINAL_ROW())
        std::cout<<"     last row to eliminate  = "<<last_row_to_eliminate;
    else
        std::cout<<"     last row to eliminate decided by factorization."<<std::endl;
    std::cout<<std::endl;
#endif
    if (use_ILUC){
        factorization_exists = partialILUC(Arow2,Acoarse,IP,force_finish,U,Dinv,last_row_to_eliminate,threshold,zero_pivots,partial_setup_time,mem_factor,total_memory_allocated,total_memory_used);
    } else {
        Acol.change_orientation_of_data(Arow2);
        factorization_exists = partialILUCDP(Arow2,Acol,Acoarse,IP,force_finish,U,Dinv,pc2,pr2,ipc2,ipr2,last_row_to_eliminate,threshold,bp,bpr,epr,zero_pivots,partial_setup_time,mem_factor,total_memory_allocated,total_memory_used);
    }
    if(!factorization_exists) return false;
#ifdef INFO
    std::cout<<"     zero-pivots            = "<<zero_pivots<<std::endl;
    std::cout<<"     local fill-in          = "<<((Real)(actual_non_zeroes()+U.actual_non_zeroes())- (Real) Acol.rows() )/((Real)Acol.actual_non_zeroes())<<std::endl;
#endif
    if(use_ILUC){
        permutation_columns=pc1;
        permutation_rows=pr1;
    } else {
        permutation_columns.compose(pc1,pc2);
        permutation_rows.compose(pr1,pr2);
    }
    inverse_permutation_columns.invert(permutation_columns);
    inverse_permutation_rows.invert(permutation_rows);
    time_2 = clock();
    setup_time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
    return true;
}


/*
template<class T>
bool matrix_sparse<T>::ILUCDPinv(const matrix_sparse<T>& Arow, const matrix_sparse<T>& Acol, matrix_sparse<T>& U, index_list& perm, index_list& permrows, Integer max_fill_in, Real threshold, Real perm_tol,  Integer bpr, Integer& zero_pivots, Real& time_self, Real mem_factor){
    try {
      clock_t time_begin, time_end;
      time_begin=clock();
      if (threshold > 500.0) threshold=0.0;
      else threshold=std::exp(-threshold*std::log(10.0));
      if (perm_tol > 500.0) perm_tol=0.0;
      else perm_tol=std::exp(-perm_tol*std::log(10.0));
      #ifdef VERBOSE
          clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_8,time_9;
          Real time_init=0.0;
          Real time_read=0.0;
          Real time_calc_L=0.0;
          Real time_scu_L=0.0;  // sorting, copying, updating access information
          Real time_calc_U=0.0;
          Real time_scu_U=0.0;
          Real time_zeroset=0.0;
          Real time_compress=0.0;
          Real time_resort=0.0;
          Real time_weights=0.0;
          time_0 = clock();
      #endif
      if(non_fatal_error(!Arow.square_check(),"matrix_sparse::ILUCDPinv: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      if(non_fatal_error(!Acol.square_check(),"matrix_sparse::ILUCDPinv: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      if(non_fatal_error(Acol.rows()!=Arow.rows(),"matrix_sparse::ILUCDPinv: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      Integer n = Acol.columns();
      Integer a,b,k,i,j,p,current_row_col_U,current_col_row_L;
      Integer h,pos, selected_row;
      T current_data_row_L,current_data_col_U;
      zero_pivots=0;
      Real norm_L,norm_U; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
      vector_sparse_dynamic<T> w(n), z(n);
      vector_dense<bool> non_pivot(n,true);
      vector_dense<bool> unused_rows(n,true);
      vector_dense<Integer> numb_el_row_L(n,0), pointer_num_el_row_L(n+1,n);
      vector_dense<T> weights_L(n+1);
      vector_dense<T> weights_U(n+1);
      Real xiplus_L,ximinus_L;
      pointer_num_el_row_L[0]=0;
      index_list list_L, list_U;
      index_list inverse_perm(n), inverse_permrows(n);
      if(max_fill_in<1) max_fill_in = 1;
      if(max_fill_in>n) max_fill_in = n;
      Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*Acol.non_zeroes());
      std::vector<Integer> linkU(reserved_memory); //h=link[startU[i]]] points to second 2nd element, link[h] to next, etc.
      std::vector<Integer> rowU(reserved_memory);   // row indices of elements of U.data.
      std::vector<Integer> startU(n); // startU[i] points to start of points to an index of data belonging to column i
      std::vector<Integer> linkL(reserved_memory); //h=link[startL[i]]] points to second 2nd element, link[h] to next, etc.
      std::vector<Integer> colL(reserved_memory);  // column indices of elements of data.
      std::vector<Integer> startL(n); // startL[i] points to start of points to an index of data belonging to row i
      U.reformat(n,n,reserved_memory,ROW);
      U.pointer[0]=0;
      weights_L[0]=1.0;
      reformat(n,n,reserved_memory,COLUMN);
      pointer[0]=0;
      perm.resize(n);
      permrows.resize(n);
      for(k=0;k<n;k++) startU[k]=-1;
      for(k=0;k<n;k++) startL[k]=-1;
      // (1.) begin for k
      #ifdef VERBOSE
          time_1 = clock();
          time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
      #endif
      for(k=0;k<n;k++){
          #ifdef VERBOSE
              time_2=clock();
          #endif
          // (2.) initialize z
          selected_row = permrows[k];
        unused_rows[selected_row]=false;
          z.zero_reset();
          #ifdef VERBOSE
              time_3=clock();
              time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
          #endif
          // read row of A
          for(i=Arow.pointer[selected_row];i<Arow.pointer[selected_row+1];i++){
              if(non_pivot[Arow.indices[i]]) z[Arow.indices[i]] = Arow.data[i];
          }     // end for i
          #ifdef VERBOSE
              time_4=clock();
              time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
          #endif
          // (3.) begin while
          h=startL[selected_row]; // h=startL[permrows[k]];
          while(h!=-1){
              current_col_row_L=colL[h];
              current_data_row_L=data[h];
              h=linkL[h];
              for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                  if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
              } // end for
          }   // (5.) end while
          #ifdef VERBOSE
              time_5=clock();
              time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
          #endif
          // (6.) sort and copy data to U; update information for accessing columns of U
          z.take_single_weight_largest_elements_by_abs_value_with_threshold_pivot_last(list_U,weights_U,max_fill_in,threshold,perm[k],perm_tol);
              // dropping too stringent?
          if(list_U.dimension()==0){
              if(threshold>0.0)
                  #ifdef VERBOSE
                      std::cout<<"Dropping too stringent, selecting elements without threshold."<<std::endl;
                  #endif
                  z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,0.0,perm[k],perm_tol);
          }
          // still no non-zero elements?
          if(list_U.dimension()==0){
              #ifdef VERBOSE
                 std::cout<<"Obtained a zero row, setting an arbitrary element to 1."<<std::endl;
              #endif
              zero_pivots++;
              z[perm[k]]=1.0;
              list_U.resize(1);
              list_U[0]=perm[k];
          } // end if
          if(U.pointer[k]+list_U.dimension()>reserved_memory){
              std::cerr<<"matrix_sparse::ILUCDPinv: memory reserved was insufficient."<<std::endl;
              return false;
          }
          // copy data, update access information.
          // copy pivot
          U.data[U.pointer[k]]=z[list_U[list_U.dimension()-1]];
          U.indices[U.pointer[k]]=list_U[list_U.dimension()-1];
          for(j=1;j<list_U.dimension();j++){
              pos=U.pointer[k]+j;
              U.data[pos]=z[list_U[list_U.dimension()-1-j]];
              U.indices[pos]=list_U[list_U.dimension()-1-j];
              h=startU[U.indices[pos]];
              startU[U.indices[pos]]=pos;
              linkU[pos]=h;
              rowU[pos]=k;
          }
          U.pointer[k+1]=U.pointer[k]+list_U.dimension();
          if(U.data[U.pointer[k]]==0){
              std::cerr<<"matrix_sparse::ILUCDPinv: Pivot is zero, because pivoting was not permitted. Preconditioner does not exist."<<std::endl;
              std::cout<<"dim list_U "<<list_U.dimension()<<std::endl;
              std::cout<<"last element corresponding to pivot: "<<z[perm[k]]<<std::endl; 
              return false;
          }
          // store positions of columns of U, but without pivot
          // update non-pivots.
          // (7.) update permutations
          p=inverse_perm[U.indices[U.pointer[k]]];
          inverse_perm.switch_index(perm[k],U.indices[U.pointer[k]]);
          perm.switch_index(k,p);
          non_pivot[U.indices[U.pointer[k]]]=false;
          #ifdef VERBOSE
              time_6=clock();
              time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
          #endif
           // (8.) read w
          w.zero_reset();
          #ifdef VERBOSE
              time_7=clock();
              time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
          #endif
          // read column of A
          for(i=Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
              if(unused_rows[Acol.indices[i]])
                  w[Acol.indices[i]] = Acol.data[i];
          }     // end for i
          #ifdef VERBOSE
              time_8=clock();
              time_read += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
          #endif
          // (9.) begin while
          h=startU[perm[k]];
          while(h!=-1){
              current_row_col_U=rowU[h];
              current_data_col_U=U.data[h];
              h=linkU[h];
             // (10.) w = w - U(i,perm(k))*l_i
              for(j=pointer[current_row_col_U];j<pointer[current_row_col_U+1];j++){
                  if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U*data[j];
              } // end for
          }   // (11.) end while
         #ifdef VERBOSE
              time_9=clock();
              time_calc_L += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
          #endif
          // (12.) sort and copy data to L
          // sort
          w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP, list_L,fabs(weights_L[k]),max_fill_in-1,threshold,0,n);
          if(pointer[k]+list_L.dimension()+1>reserved_memory){
              std::cerr<<"matrix_sparse::ILUCDPinv: memory reserved was insufficient."<<std::endl;
              return false;
          }
          // copy data
          data[pointer[k]]=1.0;
          indices[pointer[k]]=selected_row;
          for(j=0;j<list_L.dimension();j++){
              pos = pointer[k]+j+1;
              data[pos] = w[list_L[j]]/U.data[U.pointer[k]];
              b = indices[pos] = list_L[j];
              h=startL[b];
              startL[b]=pos;
              linkL[pos]=h;
              colL[pos]=k;
              // begin updating fields for number elements of row of L
              if (b > bpr) {
                  b = inverse_permrows[b];
                  a = --pointer_num_el_row_L[++numb_el_row_L[b]];
                  inverse_permrows.switch_index(permrows[a],permrows[b]);
                  permrows.switch_index(a,b);
                  numb_el_row_L.switch_entry(a,b);
              }
              // end updating fields
          } // end for j
          // sort permrows if necessary, i.e. if num_el_row_L increases at next iteration.
              if(pointer_num_el_row_L[numb_el_row_L[k]+1] == k+1) 
                  permrows.quicksort_with_inverse(inverse_permrows,pointer_num_el_row_L[numb_el_row_L[k]+1],pointer_num_el_row_L[numb_el_row_L[k]+2]-1);
          // end sorting
          pointer[k+1]=pointer[k]+list_L.dimension()+1;
          //if (k == bpr) threshold /= 100.0;
         #ifdef VERBOSE
              time_0=clock();
              time_scu_L += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
          #endif
           // update weights
          for(j=pointer[k]+1;j<pointer[k+1];j++){
               weights_L[indices[j]] -= weights_L[k]*data[j];
          }
          xiplus_L=1.0+weights_L[k+1];
          ximinus_L=-1.0+weights_L[k+1];
          if(fabs(xiplus_L)<fabs(ximinus_L))weights_L[k+1]=ximinus_L;
          else weights_L[k+1]=xiplus_L;
          for(j=U.pointer[k]+1;j<U.pointer[k+1];j++){
              weights_U[U.indices[j]] -= weights_U[perm[k]]*U.data[j];
          }
          #ifdef VERBOSE
              time_1=clock();
              time_weights += (Real)(time_1-time_0)/(Real)CLOCKS_PER_SEC;
          #endif
      }  // (13.) end for k
      #ifdef VERBOSE
          time_2 = clock();
      #endif
      compress();
      U.compress();
      #ifdef VERBOSE
          time_3=clock();
          time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
      #endif
      #ifdef VERBOSE
          time_4=clock();
          time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
          std::cout<<"    ILUCDPinv-Times: "<<std::endl;
          std::cout<<"        initialization:                           "<<time_init<<std::endl;
          std::cout<<"        reading matrix:                           "<<time_read<<std::endl;
          std::cout<<"        sparse zero set:                          "<<time_zeroset<<std::endl;
          std::cout<<"        calculating L:                            "<<time_calc_L<<std::endl;
          std::cout<<"        calculating U:                            "<<time_calc_U<<std::endl;
          std::cout<<"        sorting, copying, updating access info L: "<<time_scu_L<<std::endl;
          std::cout<<"        sorting, copying, updating access info U: "<<time_scu_U<<std::endl;
          std::cout<<"        compressing:                              "<<time_compress<<std::endl;
          std::cout<<"        resorting:                                "<<time_resort<<std::endl;
          std::cout<<"        updating weights:                         "<<time_weights<<std::endl;
          std::cout<<"      Total times:"<<std::endl;
          std::cout<<"        calculations:                             "<<time_calc_L+time_calc_U<<std::endl;
          std::cout<<"        sorting, copying, updating access info:   "<<time_scu_L+time_scu_U<<std::endl;
          std::cout<<"        other administration:                     "<<time_init+time_read+time_zeroset+time_compress+time_resort+time_weights<<std::endl;
          std::cout<<"      Grand total                                 "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort+time_weights<<std::endl;
          std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
      #endif
      time_end=clock();
      time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
      return true;
   }
   catch(iluplusplus_error ippe){
      std::cerr << "matrix_sparse<T>:: ILUCDPinv: "<<ippe.error_message() << std::endl;
      throw;
   }
  }
*/

} // end namespace iluplusplus
