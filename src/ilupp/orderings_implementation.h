/***************************************************************************
 *   Copyright (C) 2006 by Jan Mayer                                       *
 *   jan.mayer@mathematik.uni-karlsruhe.de                                 *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#ifndef ORDERINGS_IMPLEMENTATION_H
#define ORDERINGS_IMPLEMENTATION_H


//#include <math.h>
#include <cmath>
#include <iostream>

#include "declarations.h"
#include "arrays.h"
#include "functions.h"
#include "orderings.h"

#include "functions_implementation.h"

#ifdef ILUPLUSPLUS_USES_SPARSPAK
#include "sparspak.cpp"
#endif


namespace iluplusplus {

#ifdef ILUPLUSPLUS_USES_SPARSPAK
template <class T> void RCM(const matrix_sparse<T>& Arow, index_list& permP)
{
  int i;
  int output;
  int nr=Arow.rows();
  int nc=Arow.columns();

  std::vector<int> ia(nr+1), ja(Arow.non_zeroes()), p(nr);

   for(i=0;i<=nr;i++) ia[i]=Arow.read_pointer(i); 
   for(i=0;i<Arow.non_zeroes();i++) ja[i]=Arow.read_index(i); 
   output = PERMRCM(nr,nc,Arow.non_zeroes(),&ia[0],&ja[0],&p[0]);
   permP.resize(nr);
   for(i=0;i<nr;i++) permP.set(i)=p[i];
}

template <class T> void RCM(const matrix_sparse<T>& Arow, index_list& permP, int b, int e) // begin and end of row and columns, range is from i=b to i<e.
{
  int i,j;
  int output;
  int counter=0;
  if(non_fatal_error(e>Arow.rows()||b<0||Arow.rows()!=Arow.columns(), "RCM: error in row and column range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
  int nr=e-b;
  int nc=e-b;
  int* ia = 0;
  int* ja = 0;
  int* p = 0;
  try {
   ia = new int[nr+1];
   ja = new int[Arow.non_zeroes()];
   p  = new int[nr];
   ia[0]=1;
   for(i=b;i<e;i++){
       for(j=Arow.read_pointer(i);j<Arow.read_pointer(i+1);j++){
           if(Arow.read_index(j)>=b && Arow.read_index(j)<e){
               ja[counter]=Arow.read_index(j)+1-b; 
               counter++;
           } //end if
       } // end for j
       ia[i-b+1]=counter+1; 
   }  // end for i
   output = PERMRCM(nr,nc,ia[e-b],ia,ja,p);
   permP.resize(Arow.rows());
   for(i=0;i<nr;i++) permP.set(b+i)=b+p[i]-1;
   delete [] ia; delete [] ja; delete [] p;
  }
  catch(iluplusplus_error ippe){
    std::cerr<<"RCM: "<<ippe.error_message()<<" Returning empty permutation."<<std::endl;
    permP.resize(0);
    if(ia != 0) delete [] ia;
    if(ja != 0) delete [] ja;
    if(p != 0) delete [] p;
    throw;
  }
  catch(std::bad_alloc){
    std::cerr<<"RCM: Insufficient memory. Returning empty permutation."<<std::endl;
    permP.resize(0);
    if(ia != 0) delete [] ia;
    if(ja != 0) delete [] ja;
    if(p != 0) delete [] p;
    throw iluplusplus_error(INSUFFICIENT_MEMORY);
  }
}


// requires ia and ja to have indexing beginning by 0, returns permutation beginning at 0. int required for compatibilty with genrcm_
int PERMRCM(int nr, int nc, int nnz, int* ia, int* ja, int *p)
{
  int i, ierr=0;
  if(non_fatal_error(nr !=nc ,"PERMRCM: matrix must be square")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
  int* sia = 0;
  int* sja = 0;
  int* ibuff1 = 0;
  int* ibuff2 = 0;
  try {
   sia = new int[nr+1];
   sja = new int[2*nnz];
   // requires 0 indexing, returns 1-indexing upon request (i.e. true)
   setup_symmetric_graph<int>(nr,ia,ja,sia,sja,true);
   ibuff1=new int[nc+1];
   ibuff2=new int[nc+1];
   // reverse Cuthill-McKee; requires 1-indexing, returns 1-indexing
   genrcm_(&nc,sia,sja, p,ibuff1,ibuff2);
   // shift to 0-indexing
   for(i=0;i<nr;i++) p[i]--;
   delete [] sia;
   delete [] sja;
   delete [] ibuff1;
   delete [] ibuff2;
   return (ierr);
  }
  catch(std::bad_alloc){
    std::cerr<<"PERMRCM: Error allocating memory."<<std::endl;
    if(sia != 0) delete [] sia;
    if(sja != 0) delete [] sja;
    if(ibuff1 != 0) delete [] ibuff1;
    if(ibuff2 != 0) delete [] ibuff2;
    throw iluplusplus_error(INSUFFICIENT_MEMORY);
  }
}
#endif // ILUPLUSPLUS_USES_SPARSPAK

template <class int_type> void setup_symmetric_graph(int_type dim, int_type* pointer, int_type* indices, int_type* sym_pointer, int_type* sym_indices, bool shift_index)
{

    // pointer, indices: contain data of non-symmetric matrix A
    // sym_pointer, sym_indices: contain data of A+A^T
    // sym_pointer must have same length as pointer, dim+1;
    // sym_indices must have twice the length of indices
    // REQUIRES: 0-indexing. Returns 1-indexing if shift is true, 0 indexing if shift is false.
    int_type j,k,pos,T_pos,sym_pos;
    array<int_type> T_num_element, T_pointer, T_indices;
    T_num_element.resize(dim,0);
    T_pointer.resize(dim+1);
    T_indices.resize(pointer[dim]);
    // count number of elements in each column/row of transpose
    for(j=0;j<pointer[dim];j++) T_num_element[indices[j]]++;
    // set-up pointer for transpose
    T_pointer[0] = 0;
    for(j=0;j<dim;j++) T_pointer[j+1] = T_pointer[j] + T_num_element[j];
    // generate T_indices
    T_num_element.set_all(0);
    for(j=0;j<dim;j++)
        for(k=pointer[j];k<pointer[j+1];k++)
            T_indices[T_pointer[indices[k]]+(T_num_element[indices[k]]++)]=j;
    // copy data to sym_pointer, sym_indices
    sym_pointer[0]=0;
    for(j=0;j<dim;j++){
        pos     = pointer[j];
        T_pos   = T_pointer[j];
        sym_pos = sym_pointer[j];
        while(pos < pointer[j+1] && T_pos < T_pointer[j+1]){
            if(indices[pos] < T_indices[T_pos]){
                sym_indices[sym_pos++] = indices[pos++];
                continue;
            }
            if(indices[pos] > T_indices[T_pos]){
                sym_indices[sym_pos++] = T_indices[T_pos++];
                continue;
            }
            if(indices[pos] == T_indices[T_pos]){
                sym_indices[sym_pos++] = indices[pos++];
                T_pos++;
                continue;
            }
        }
        // because of while, at most one of these for-loops is non-empty
        while(pos < pointer[j+1]){
            sym_indices[sym_pos++] = indices[pos++];
        }
        while(T_pos < T_pointer[j+1]){
            sym_indices[sym_pos++] = T_indices[T_pos++];
        }
        sym_pointer[j+1] = sym_pos;
    }
    if(shift_index){
        for(j=0;j<sym_pointer[dim];j++) sym_indices[j]++;
        for(j=0;j<=dim;j++) sym_pointer[j]++;
    }
}


template <class T> void setup_symmetric_graph(const matrix_sparse<T> &A, Integer* sym_pointer, Integer* sym_indices)
{

    // non-symmetric matrix A
    // sym_pointer, sym_indices: contain data of A+A^T
    // sym_pointer must have same length as pointer, dim+1;
    // sym_indices must have twice the length of indices
    // requires and returns 0-indexing
    Integer j,k,pos,T_pos,sym_pos,dim=A.read_pointer_size()-1;
    array<Integer> T_num_element, T_pointer, T_indices;
    T_num_element.resize(dim,0);
    T_pointer.resize(dim+1);
    T_indices.resize(A.read_pointer(dim));
    // count number of elements in each column/row of transpose
    for(j=0;j<A.read_pointer(dim);j++) T_num_element[A.read_index(j)]++;
    // set-up pointer for transpose
    T_pointer[0] = 0;
    for(j=0;j<dim;j++) T_pointer[j+1] = T_pointer[j] + T_num_element[j];
    // generate T_indices
    T_num_element.set_all(0);
    for(j=0;j<dim;j++)
        for(k=A.read_pointer(j);k<A.read_pointer(j+1);k++)
            T_indices[T_pointer[A.read_index(k)]+(T_num_element[A.read_index(k)]++)]=j;
    // copy data to sym_pointer, sym_indices
    sym_pointer[0]=0;
    for(j=0;j<dim;j++){
        pos     = A.read_pointer(j);
        T_pos   = T_pointer[j];
        sym_pos = sym_pointer[j];
        while(pos < A.read_pointer(j+1) && T_pos < T_pointer[j+1]){
            if(A.read_index(pos) < T_indices[T_pos]){
                sym_indices[sym_pos++] = A.read_index(pos++);
                continue;
            }
            if(A.read_index(pos) > T_indices[T_pos]){
                sym_indices[sym_pos++] = T_indices[T_pos++];
                continue;
            }
            if(A.read_index(pos) == T_indices[T_pos]){
                sym_indices[sym_pos++] = A.read_index(pos++);
                T_pos++;
                continue;
            }
        }
        // because of while, at most one of these for-loops is non-empty
        while(pos < A.read_pointer(j+1)){
            sym_indices[sym_pos++] = A.read_index(pos++);
        }
        while(T_pos < T_pointer[j+1]){
            sym_indices[sym_pos++] = T_indices[T_pos++];
        }
        sym_pointer[j+1] = sym_pos;
    }
}

void setup_symmetric_graph_without_diag(Integer dim, Integer* pointer, Integer* indices, Integer* sym_pointer, Integer* sym_indices)
{

    // pointer, indices: contain data of non-symmetric matrix A
    // sym_pointer, sym_indices: contain data of A+A^T, without diagonal
    // sym_pointer must have same length as pointer, dim+1;
    // sym_indices must have twice the length of indices
    // requires and returns 0-indexing
    Integer j,k,pos,T_pos,sym_pos;
    array<Integer> T_num_element, T_pointer, T_indices;
    T_num_element.resize(dim,0);
    T_pointer.resize(dim+1);
    T_indices.resize(pointer[dim]);
    // count number of elements in each column/row of transpose
    for(j=0;j<pointer[dim];j++) T_num_element[indices[j]]++;
    // set-up pointer for transpose
    T_pointer[0] = 0;
    for(j=0;j<dim;j++) T_pointer[j+1] = T_pointer[j] + T_num_element[j];
    // generate T_indices
    T_num_element.set_all(0);
    for(j=0;j<dim;j++)
        for(k=pointer[j];k<pointer[j+1];k++)
            T_indices[T_pointer[indices[k]]+(T_num_element[indices[k]]++)]=j;
    // copy data to sym_pointer, sym_indices
    sym_pointer[0]=0;
    for(j=0;j<dim;j++){
        pos     = pointer[j];
        T_pos   = T_pointer[j];
        sym_pos = sym_pointer[j];
        while(pos < pointer[j+1] && T_pos < T_pointer[j+1]){
            if(indices[pos] < T_indices[T_pos]){
                if(indices[pos] != j)
                    sym_indices[sym_pos++] = indices[pos++];
                else pos++;
                continue;
            }
            if(indices[pos] > T_indices[T_pos]){
                if(T_indices[T_pos] != j)
                    sym_indices[sym_pos++] = T_indices[T_pos++];
                else T_pos++;
                continue;
            }
            if(indices[pos] == T_indices[T_pos]){
                if(indices[pos] != j)
                    sym_indices[sym_pos++] = indices[pos++];
                else pos++;
                T_pos++;
                continue;
            }
        }
        // because of while, at most one of these for-loops is non-empty
        while(pos < pointer[j+1]){
            if(indices[pos] != j)
                sym_indices[sym_pos++] = indices[pos++];
            else pos++;
        }
        while(T_pos < T_pointer[j+1]){
            if(T_indices[T_pos] != j)
                sym_indices[sym_pos++] = T_indices[T_pos++];
            else T_pos++;
        }
        sym_pointer[j+1] = sym_pos;
    }
}


template <class T> void setup_symmetric_graph_without_diag(const matrix_sparse<T> &A, Integer* sym_pointer, Integer* sym_indices)
{

    // non-symmetric matrix A
    // sym_pointer, sym_indices: contain data of A+A^T
    // sym_pointer must have same length as pointer, dim+1;
    // sym_indices must have twice the length of indices
    // requires and returns 0-indexing
    if(non_fatal_error(!(A.square_check()),"setup_symmetric_graph_without_diag: matrix must be square")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer j,k,pos,T_pos,sym_pos,dim=A.read_pointer_size()-1;
    array<Integer> T_num_element, T_pointer, T_indices;
    T_num_element.resize(dim,0);
    T_pointer.resize(dim+1);
    T_indices.resize(A.read_pointer(dim));
    // count number of elements in each column/row of transpose
    for(j=0;j<A.read_pointer(dim);j++) T_num_element[A.read_index(j)]++;
    // set-up pointer for transpose
    T_pointer[0] = 0;
    for(j=0;j<dim;j++) T_pointer[j+1] = T_pointer[j] + T_num_element[j];
    // generate T_indices
    T_num_element.set_all(0);
    for(j=0;j<dim;j++)
        for(k=A.read_pointer(j);k<A.read_pointer(j+1);k++)
            T_indices[T_pointer[A.read_index(k)]+(T_num_element[A.read_index(k)]++)]=j;
    // copy data to sym_pointer, sym_indices
    sym_pointer[0]=0;
    for(j=0;j<dim;j++){
        pos     = A.read_pointer(j);
        T_pos   = T_pointer[j];
        sym_pos = sym_pointer[j];
        while(pos < A.read_pointer(j+1) && T_pos < T_pointer[j+1]){
            if(A.read_index(pos) < T_indices[T_pos]){
                if(A.read_index(pos) != j)
                    sym_indices[sym_pos++] = A.read_index(pos++);
                else pos++;
                continue;
            }
            if(A.read_index(pos) > T_indices[T_pos]){
                if(T_indices[T_pos] != j)
                    sym_indices[sym_pos++] = T_indices[T_pos++];
                else T_pos++;
                continue;
            }
            if(A.read_index(pos) == T_indices[T_pos]){
                if(A.read_index(pos) != j)
                    sym_indices[sym_pos++] = A.read_index(pos++);
                else pos++;
                T_pos++;
                continue;
            }
        }
        // because of while, at most one of these for-loops is non-empty
        while(pos < A.read_pointer(j+1)){
            if(A.read_index(pos) != j)
                sym_indices[sym_pos++] = A.read_index(pos++);
            else pos++;
        }
        while(T_pos < T_pointer[j+1]){
            if(T_indices[T_pos] != j)
                sym_indices[sym_pos++] = T_indices[T_pos++];
            else T_pos++;
        }
        sym_pointer[j+1] = sym_pos;
    }
}


#ifdef ILUPLUSPLUS_USES_METIS

template <class T> void METIS_NODE_ND(const matrix_sparse<T>& A, Integer* p,  Integer* invp)
{
  if(non_fatal_error(!(A.square_check()),"METIS_NODE_ND: matrix must be square")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
  Integer dim =A.read_pointer_size()-1;
  Integer* sia = 0;
  Integer* sja = 0;
  try {
   sia = new Integer[dim+1];
   sja = new Integer[2*A.non_zeroes()];
   setup_symmetric_graph_without_diag(A,sia,sja);
   int options[10];
   options[0] = 0;
   int numflag = 0;  // ensures 0-indexing
   METIS_NodeND(&dim, sia, sja, &numflag, options, p, invp);
   delete [] sia; delete [] sja;
  }
  catch(std::bad_alloc){
    std::cerr<<"METIS_NODE_ND: Error allocating memory."<<std::endl;
    if(sia != 0) delete [] sia;
    if(sja != 0) delete [] sja;
    throw iluplusplus_error(INSUFFICIENT_MEMORY);
  }
}



template <class T> void METIS_NODE_ND(const matrix_sparse<T>& A, index_list& permP,  index_list& inv_permP )
{
  if(non_fatal_error(!(A.square_check()),"METIS_NODE_ND: matrix must be square")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
  Integer i;
  Integer dim =A.read_pointer_size()-1;
  Integer* p = 0;
  Integer* invp = 0;
  try {
   p =  new Integer[dim];
   invp =  new  Integer[dim];
   METIS_NODE_ND(A,p,invp);
   permP.resize(dim);
   inv_permP.resize(dim);
   for(i=0;i<dim;i++) permP.set(i)=p[i];
   for(i=0;i<dim;i++) inv_permP.set(i)=invp[i];
   delete [] p; delete [] invp;
  }
  catch(iluplusplus_error ippe){
    std::cerr<<"METIS_NODE_ND: "<<ippe.error_message()<<" Returning empty permutation."<<std::endl;
    permP.resize(0);
    inv_permP.resize(0);
    if(p != 0) delete [] p;
    if(invp != 0) delete [] invp;
    throw;
  }
  catch(std::bad_alloc){
    std::cerr<<"METIS_NODE_ND: Error allocating memory. Returning empty permutation."<<std::endl;
    permP.resize(0);
    inv_permP.resize(0);
    if(p != 0) delete [] p;
    if(invp != 0) delete [] invp;
    throw iluplusplus_error(INSUFFICIENT_MEMORY);
  }
}

#endif // end ILUPLUSPLUS_USES_METIS

/***********************************************************************************

class preprocessing_sequence

************************************************************************************/

/*
preprocessing_sequence& preprocessing_sequence::operator = (const preprocessing_sequence& A) {
    try {
        if(this == &A) return *this;
        erase_resize_data_field(A.size);
        for(Integer i=0;i<size;i++) data[i]=A.data[i];
        return *this;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"preprocessing_sequence::operator = : "<<ippe.error_message()<<std::endl;
        throw;
    }
  }
*/

void preprocessing_sequence::set_test_new(){
    resize(2);
    set(0) = TEST_ORDERING;
    set(1) = MAX_WEIGHTED_MATCHING_ORDERING;
}

void preprocessing_sequence::set_none(){
    resize(0);
}

void preprocessing_sequence::set_normalize(){
    resize(2);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
}

void preprocessing_sequence::set_PQ(){
    resize(3);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PQ_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = UNIT_OR_ZERO_DIAGONAL_SCALING;
}
/*
void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING(){
  try {
    resize(3);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
  }
  catch(iluplusplus_error ippe){
    std::cerr<<"preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING(: "<<ippe.error_message()<<std::endl;
    throw;
  }
}
*/
/*
void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(){
  try {
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
  }
  catch(iluplusplus_error ippe){
    std::cerr<<"preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM: "<<ippe.error_message()<<std::endl;
    throw;
  }
}
*/
void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(2) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}
/*
void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(){
  try {
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
  }
  catch(iluplusplus_error ippe){
    std::cerr<<"preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM: "<<ippe.error_message()<<std::endl;
    throw;
  }
}
*/
void preprocessing_sequence::set_SPARSE_FIRST(){
    resize(1);
    set(0) = SPARSE_FIRST_ORDERING;
}

void preprocessing_sequence::set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING(){
    resize(2);
    set(0) = SPARSE_FIRST_ORDERING;
    set(1) = MAX_WEIGHTED_MATCHING_ORDERING;
}

void preprocessing_sequence::set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG(){
    resize(3);
    set(0) = SPARSE_FIRST_ORDERING;
    set(1) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(2) = UNIT_OR_ZERO_DIAGONAL_SCALING;
}

void preprocessing_sequence::set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(){
    resize(3);
    set(0) = SPARSE_FIRST_ORDERING;
    set(1) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(2) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM(){
    resize(4);
    set(0) = SPARSE_FIRST_ORDERING;
    set(1) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(2) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(3) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING(){
    resize(1);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING(){
    resize(3);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_PQ(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = PQ_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_PQ(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = PQ_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT2_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT2_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMM_PQ;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMM_PQ;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(2) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(4) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(2) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(4) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}


#ifdef ILUPLUSPLUS_USES_METIS
void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(){
    resize(4);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(3) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(){
    resize(6);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(5) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(){
    resize(4);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(3) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(){
    resize(6);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(5) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_METIS(){
    resize(2);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_METIS(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = PQ_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = PQ_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMM_PQ;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMM_PQ;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(){
    resize(3);
    set(0) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}
#endif  // METIS available



#ifdef ILUPLUSPLUS_USES_SPARSPAK
void preprocessing_sequence::set_RCM(){
    resize(3);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = REVERSE_CUTHILL_MCKEE_ORDERING;
}

void preprocessing_sequence::set_PQ_RCM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PQ_ORDERING;
    set(3) = REVERSE_CUTHILL_MCKEE_ORDERING;
}
#endif

#ifdef ILUPLUSPLUS_USES_METIS  // requires only METIS
void preprocessing_sequence::set_METIS_NODE_ND_ORDERING(){
    resize(3);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = METIS_NODE_ND_ORDERING;
}

void preprocessing_sequence::set_PQ_METIS_NODE_ND_ORDERING(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PQ_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
}
#endif

#ifdef ILUPLUSPLUS_USES_PARDISO
void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING(){
    resize(1);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING(){
    resize(3);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_PQ(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = PQ_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_PQ(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = PQ_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = WGT2_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = WGT2_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMM_PQ;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMM_PQ;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(2) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(4) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(2) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(4) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}
#endif

#if defined(ILUPLUSPLUS_USES_METIS) && defined(ILUPLUSPLUS_USES_PARDISO)
void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = DD_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(){
    resize(4);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(3) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(){
    resize(6);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(5) = SP_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(){
    resize(4);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(3) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(){
    resize(6);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = UNIT_OR_ZERO_DIAGONAL_SCALING;
    set(5) = SP_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS(){
    resize(2);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS(){
    resize(4);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = PQ_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = PQ_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMM_PQ;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMM_PQ;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMB_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = SYMB_SYMM_MOVE_CORNER_ORDERING_IM;
}

void preprocessing_sequence::set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(){
    resize(3);
    set(0) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(1) = METIS_NODE_ND_ORDERING;
    set(2) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}

void preprocessing_sequence::set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(){
    resize(5);
    set(0) = NORMALIZE_COLUMNS;
    set(1) = NORMALIZE_ROWS;
    set(2) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(3) = METIS_NODE_ND_ORDERING;
    set(4) = WGT2_SYMM_MOVE_CORNER_ORDERING;
}
#endif  // end PARDISO & METIS available

#if defined(ILUPLUSPLUS_USES_METIS) && defined(ILUPLUSPLUS_USES_PARDISO) && defined(ILUPLUSPLUS_USES_SPARSPAK)
// only for testing purposes of program... makes no sense for preconditioning
void preprocessing_sequence::set_test(){
    resize(10);
    set(0) = PQ_ORDERING;
    set(1) = NORMALIZE_COLUMNS;
    set(2) = REVERSE_CUTHILL_MCKEE_ORDERING;
    set(3) = NORMALIZE_ROWS;
    set(4) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
    set(5) = METIS_NODE_ND_ORDERING;
    set(6) = NORMALIZE_COLUMNS;
    set(7) = NORMALIZE_ROWS;
    set(8) = PQ_ORDERING;
    set(9) = PARDISO_MAX_WEIGHTED_MATCHING_ORDERING;
}
#endif

void preprocessing_sequence::print() const {
    for(Integer i=0; i<dimension();i++) std::cout<<"     "<<long_string(get(i))<<std::endl;
}

std::string preprocessing_sequence::string() const {
    std::string output;
    for(Integer i=0; i<dimension();i++){
        output = (output + iluplusplus::string(get(i)));
        if(i!=dimension()-1) output = (output + "+");
    }
    if(dimension() == 0) output = "none";
    return output;
}

std::string preprocessing_sequence::string_with_hyphen() const {
    std::string output;
    for(Integer i=0; i<dimension();i++){
        output = (output + iluplusplus::string(get(i)));
        if(i!=dimension()-1) output = (output + "-");
    }
    if(dimension() == 0) output = "none";
    return output;
}

} // end namespace iluplusplus

#endif

