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


#ifndef PRECONDITIONER_IMPLEMTATION_H
#define PRECONDITIONER_IMPLEMTATION_H
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cmath>


#include "declarations.h"
#include "sparse.h"

#include "sparse_implementation.h"

//***********************************************************************************************************************//
//                                                                                                                       //
//         The base class preconditioner: declaration                                                                    //
//                                                                                                                       //
//***********************************************************************************************************************//

namespace iluplusplus {

//***********************************************************************************************************************//
//                                                                                                                       //
//         The base class preconditioner:     implementation                                                             //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
void preconditioner<T,matrix_type,vector_type>::apply_preconditioner_only(matrix_usage_type use, T* data, Integer dim) const {
    vector_type y;
    y.interchange(data,dim);
    apply_preconditioner_only(use,y);
    y.interchange(data,dim);
}


template <class T, class matrix_type, class vector_type>
void preconditioner<T,matrix_type,vector_type>::apply_preconditioner_only(matrix_usage_type use, std::vector<T>& data) const {
    preconditioner<T,matrix_type,vector_type>::apply_preconditioner_only(use,&data[0],data.size());
}

template <class T, class matrix_type, class vector_type>
std::string preconditioner<T,matrix_type,vector_type>::special_info() const {
    std::ostringstream info;
    info<<"";
    return info.str();
}

template <class T, class matrix_type, class vector_type>
void preconditioner<T,matrix_type,vector_type>::preconditioned_residual(preconditioner_application1_type PA1,const matrix_type& A, const vector_type &b, const vector_type &x, vector_type &r) const {
    vector_type h;
    h.residual(ID,A,x,b);
    preconditioned_rhs(PA1,A,h,r);
}


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: split_preconditioner,                                                                              //
//                i.e. one operating from both sides of a matrix, either directly or indirectly                          //
//                or any preconditioner constisting of two factors.                                                      //
//                                                                                                                       //
//***********************************************************************************************************************//

// For the preconditioner of A, the composition
// Operator_associated_with_Precond_left o A o Operator_associated_with_Precond_right
// should approximate the identity operator.
// i.e. for a direct preconditioner: Precond_left * A * Precond_right = I (approx.), I being the identity matrix,
// for an indirect preconditioner:   Precond_left^{-1} * A * Precond_right^{-1} = I


template <class T, class matrix_type, class vector_type>
void split_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_only(matrix_usage_type use, const vector_type &x, vector_type &y) const {
    if(use==ID){
        apply_preconditioner_left(use,x,y);
        apply_preconditioner_right(use,y);
    } else { // TRANSPOSE
        apply_preconditioner_right(use,x,y);
        apply_preconditioner_left(use,y);
    }
}

template <class T, class matrix_type, class vector_type>
void split_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_only(matrix_usage_type use,vector_type &y) const {
    if(use==ID){
        apply_preconditioner_left(use,y);
        apply_preconditioner_right(use,y); 
    } else { // TRANSPOSE
        apply_preconditioner_right(use,y); 
        apply_preconditioner_left(use,y);
    }
}

template <class T, class matrix_type, class vector_type>
    void split_preconditioner<T,matrix_type,vector_type>::
    apply_preconditioner_and_matrix(preconditioner_application1_type PA1, matrix_usage_type use, const matrix_type &A,const vector_type &v, vector_type &w) const
{
    switch(PA1){
        case NONE:
            A.matrix_vector_multiplication(use,v,w);
            break;
        case LEFT:
            if(use == ID){
                A.matrix_vector_multiplication(ID,v,w);
                apply_preconditioner_left(ID,w);
                apply_preconditioner_right(ID,w);
            } else {
                apply_preconditioner_right(TRANSPOSE,v,w);
                apply_preconditioner_left(TRANSPOSE,w);
                A.matrix_vector_multiplication(TRANSPOSE,w);
            }
            break;
        case RIGHT:
            if(use == ID){
                apply_preconditioner_left(ID,v,w);
                apply_preconditioner_right(ID,w);
                A.matrix_vector_multiplication(ID,w);
            } else {
                A.matrix_vector_multiplication(TRANSPOSE,v,w);
                apply_preconditioner_right(TRANSPOSE,w);
                apply_preconditioner_left(TRANSPOSE,w);
            }
            break;
        case SPLIT:
            if(use == ID){
                apply_preconditioner_right(ID,v,w);
                A.matrix_vector_multiplication(ID,w);
                apply_preconditioner_left(ID,w);
            } else {
                apply_preconditioner_left(TRANSPOSE,v,w);
                A.matrix_vector_multiplication(TRANSPOSE,w);
                apply_preconditioner_right(TRANSPOSE,w);
            }
            break;
        default:
            std::cerr <<"split_preconditioner::apply_preconditioner_and_matrix: only NONE, LEFT, RIGHT, SPLIT as usage possible."<<std::endl;
            throw iluplusplus_error(ARGUMENT_NOT_ALLOWED);
            break;
    }
}


template <class T, class matrix_type, class vector_type>
    void split_preconditioner<T,matrix_type,vector_type>::
    apply_preconditioner_and_matrix_transposed(preconditioner_application1_type PA1, matrix_usage_type use, const matrix_type &A,const vector_type &v, vector_type &w) const
{
    switch(PA1){
        case NONE:
            A.matrix_vector_multiplication(other_usage(use),v,w);
            break;
        case RIGHT:
            if(use == ID){
                A.matrix_vector_multiplication(TRANSPOSE,v,w);
                apply_preconditioner_left(TRANSPOSE,w);
                apply_preconditioner_right(TRANSPOSE,w);
            } else {
                apply_preconditioner_right(ID,v,w);
                apply_preconditioner_left(ID,w);
                A.matrix_vector_multiplication(ID,w);
            }
            break;
        case LEFT:
            if(use == ID){
                apply_preconditioner_left(TRANSPOSE,v,w);
                apply_preconditioner_right(TRANSPOSE,w);
                A.matrix_vector_multiplication(TRANSPOSE,w);
            } else {
                A.matrix_vector_multiplication(ID,v,w);
                apply_preconditioner_right(ID,w);
                apply_preconditioner_left(ID,w);
            }
            break;
        case SPLIT:
            if(use == ID){
                apply_preconditioner_left(TRANSPOSE,v,w);
                A.matrix_vector_multiplication(TRANSPOSE,w);
                apply_preconditioner_right(TRANSPOSE,w);
            } else {
                apply_preconditioner_right(ID,v,w);
                A.matrix_vector_multiplication(ID,w);
                apply_preconditioner_left(ID,w);
            }
            break;
        default:
            std::cerr <<"split_preconditioner::apply_preconditioner_and_matrix_transposed: only NONE, LEFT, RIGHT, SPLIT as usage possible."<<std::endl;
            throw iluplusplus_error(ARGUMENT_NOT_ALLOWED);
            break;
    }
}

template <class T, class matrix_type, class vector_type>
void split_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_rhs(
        preconditioner_application1_type PA1, const matrix_type &A, const vector_type &b, vector_type &c) const
{
    switch(PA1) {
        case NONE:
            c=b;
            break;
        case LEFT:
            apply_preconditioner_left(ID,b,c);
            apply_preconditioner_right(ID,c);
            break;
        case RIGHT:
            c=b;
            break;
        case SPLIT:
            apply_preconditioner_left(ID,b,c);
            break;
        default:
            std::cerr <<"split_preconditioner::apply_preconditioner_rhs: only NONE, LEFT, RIGHT, SPLIT as usage possible."<<std::endl;
            throw iluplusplus_error(ARGUMENT_NOT_ALLOWED);
    }
}


template <class T, class matrix_type, class vector_type>
void split_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_solution(
        preconditioner_application1_type PA1, const matrix_type &A, const vector_type &y, vector_type &x) const
{
    switch(PA1) {
        case NONE:
            x=y;
            break;
        case LEFT:
            x=y;
            break;
        case RIGHT:
            apply_preconditioner_left(ID,y,x);
            apply_preconditioner_right(ID,x);
            break;
        case SPLIT:
            apply_preconditioner_right(ID,y,x);
            break;
        default:
            std::cerr <<"split_preconditioner::apply_preconditioner_solution: only NONE, LEFT, RIGHT, SPLIT as usage possible."<<std::endl;
            throw iluplusplus_error(ARGUMENT_NOT_ALLOWED);
    }
}


template <class T, class matrix_type, class vector_type>
void split_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_starting_value(
        preconditioner_application1_type PA1, const matrix_type &A, const vector_type &x, vector_type &y) const
{
    switch(PA1) {
        case NONE:
            y=x;
            break;
        case LEFT:
            y=x;
            break;
        case RIGHT:
            unapply_preconditioner_right(ID,x,y);
            unapply_preconditioner_left(ID,y);
            break;
        case SPLIT:
            unapply_preconditioner_right(ID,x,y);
            break;
        default:
            std::cerr <<"split_preconditioner::apply_preconditioner_starting_value: only NONE, LEFT, RIGHT as usage possible."<<std::endl;
            throw iluplusplus_error(ARGUMENT_NOT_ALLOWED);
    }
}


template <class T, class matrix_type, class vector_type>
    bool split_preconditioner<T,matrix_type,vector_type>::compatibility_check(preconditioner_application1_type PA1,const matrix_type& A, const vector_type& b) const
    {
      bool system_check = ( A.rows()!=b.dimension());
      switch (PA1){
          case NONE:
              return system_check;
          case LEFT:
              return ( (this->pre_image_size!=A.rows()) || system_check );
          case RIGHT:
              return ( (A.columns()!=this->image_size) || system_check );
          case SPLIT:
              return ( (this->pre_image_size!=A.rows()) || (A.columns()!=this->image_size) || system_check );
          default:
              std::cerr <<"split_preconditioner::compatibility check: this usage is not possible for this preconditioner."<<std::endl;
              throw iluplusplus_error(ARGUMENT_NOT_ALLOWED);
              return false; // some compilers complain if there is no return.
      }
      std::cerr <<"split_preconditioner::compatibility check: this usage is not possible for this preconditioner."<<std::endl;
      throw iluplusplus_error(ARGUMENT_NOT_ALLOWED);
      return false; // this never happens, but some compilers complain if there is no return.
    }


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: indirect_split_triangular_preconditioner                                                                      //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const {
    Precond_left.triangular_solve(left_form,use,v,w);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_left(matrix_usage_type use, vector_type &w) const {
    Precond_left.triangular_solve(left_form,use,w);
}

template <class T, class matrix_type, class vector_type> 
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const {
    Precond_right.triangular_solve(right_form,use,v,w);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_right(matrix_usage_type use, vector_type &w) const {
    Precond_right.triangular_solve(right_form,use,w);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_preconditioner::unapply_preconditioner_left: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_left(matrix_usage_type use, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_preconditioner::unapply_preconditioner_left: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_preconditioner::unapply_preconditioner_right: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_right(matrix_usage_type use, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_preconditioner::unapply_preconditioner_right: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_preconditioner<T,matrix_type,vector_type>::print_info() const {
    std::cout<<"The left matrix of the preconditioner:"<<std::endl;
    Precond_left.print_info();
    std::cout<<"The right matrix of the preconditioner:"<<std::endl;
    Precond_right.print_info();
}


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: indirect_split_triangular_multilevel_preconditioner                                                //
//                                                                                                                       //
//***********************************************************************************************************************//

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const {
    w=v;
    this->apply_preconditioner_left(use,w);
    //for(Integer i=0; i<number_levels; i++){
    //    w.inverse_scale_at_end(D_l.get(i));
    //    Precond_left.get(i).triangular_solve_with_smaller_matrix_permute_first(left_form,use,permutation_rows.get(i),w);
    //}
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_left(matrix_usage_type use, vector_type &w) const {
    if(use == ID){
        for(Integer i=0; i<number_levels; i++){
            w.inverse_scale_at_end(D_l.get(i));
            Precond_left.get(i).triangular_solve_with_smaller_matrix_permute_first(left_form,use,permutation_rows.get(i),w);
        }
    } else {
        for(Integer i=number_levels-1; i>=0; i--){
            Precond_left.get(i).triangular_solve_with_smaller_matrix_permute_last(left_form,use,inverse_permutation_rows.get(i),w);
            w.inverse_scale_at_end(D_l.get(i));
        }

    }
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const {
    w=v;
    //for(Integer i=number_levels-1; i>=0; i--){
    //    w.scale_at_end(Precond_middle.get(i));
    //    //w.inverse_scale_at_end(Precond_middle.get(i));
    //    Precond_right.get(i).triangular_solve_with_smaller_matrix_permute_last(right_form,use,inverse_permutation_columns.get(i),w);
    //    w.inverse_scale_at_end(D_r.get(i));
    //}
    this->apply_preconditioner_right(use,w);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::apply_preconditioner_right(matrix_usage_type use, vector_type &w) const {
    if(use == ID){
        for(Integer i=number_levels-1; i>=0; i--){
            w.scale_at_end(Precond_middle.get(i));
            //w.inverse_scale_at_end(Precond_middle.get(i));
            Precond_right.get(i).triangular_solve_with_smaller_matrix_permute_last(right_form,use,inverse_permutation_columns.get(i),w);
            w.inverse_scale_at_end(D_r.get(i));
        }
    } else {
        for(Integer i=0; i<number_levels; i++){
            w.inverse_scale_at_end(D_r.get(i));
            //w.inverse_scale_at_end(Precond_middle.get(i));
            Precond_right.get(i).triangular_solve_with_smaller_matrix_permute_first(right_form,use,permutation_columns.get(i),w);
            w.scale_at_end(Precond_middle.get(i));
        }
    }
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_multilevel_preconditioner::unapply_preconditioner_left: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_left(matrix_usage_type use, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_multilevel_preconditioner::unapply_preconditioner_left: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_multilevel_preconditioner::unapply_preconditioner_right: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::unapply_preconditioner_right(matrix_usage_type use, vector_type &w) const{
    std::cerr<<"indirect_split_triangular_multilevel_preconditioner::unapply_preconditioner_right: undoing this preconditioner is not yet implemented."<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::init(Integer memory_max_level){
    Precond_left.destroy_resize_data_field(memory_max_level);     // the left preconditioning matrices
    Precond_right.destroy_resize_data_field(memory_max_level);    // the right preconditioning matrices
    Precond_middle.destroy_resize_data_field(memory_max_level);
    begin_next_level.destroy_resize_data_field(memory_max_level);
    permutation_rows.destroy_resize_data_field(memory_max_level);
    permutation_columns.destroy_resize_data_field(memory_max_level);
    inverse_permutation_rows.destroy_resize_data_field(memory_max_level);
    inverse_permutation_columns.destroy_resize_data_field(memory_max_level);
    D_l.destroy_resize_data_field(memory_max_level);  // scaling
    D_r.destroy_resize_data_field(memory_max_level);  // scaling
}

template <class T, class matrix_type, class vector_type>
  Integer indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::dim(Integer k) const {
      #ifdef DEBUG
          if(non_fatal_error(k<0||k>=number_levels, "indirect_split_triangular_multilevel_preconditioner: dim: This level does not exist.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      #endif
      return Precond_left.get(k).rows();
  }

template <class T, class matrix_type, class vector_type>
  void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::print_dimensions() const{for(Integer k=0;k<number_levels; k++) std::cout<<dim(k)<<"  "; std::cout<<std::endl;}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::write_abs_diagonal(std::string filename) const {
    Integer i;
    vector_type v;
    for(i=0;i<number_levels;i++){
        v.absvalue(Precond_middle.get(i),0,Precond_middle.get(i).dimension()-Precond_middle.get(i+1).dimension());
        v.append(filename);
    }
}

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::write_abs_diagonal_with_indices(std::string filename) const {
    Integer counter = 0;
    Integer i;
    vector_type v;
    for(i=0;i<number_levels;i++){
        v.absvalue(Precond_middle.get(i),0,Precond_middle.get(i).dimension()-Precond_middle.get(i+1).dimension());
        v.append_with_indices(filename,counter);
        counter = Precond_middle.get(i).dimension()-Precond_middle.get(i+1).dimension(); // new starting position
    }
}

template <class T, class matrix_type, class vector_type>
Integer indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::number_small_pivots(Real tau) const { // returns the number of pivots whose abs. value is less than or equal to tau.
    Integer counter = 0;
    Integer i,j;
    for(i=0;i<number_levels;i++){
        for(j=0; j< Precond_middle.get(i).dimension()-Precond_middle.get(i+1).dimension(); j++) 
            if (fabs(Precond_middle.get(i).get(j))>= 1.0/tau) counter++;
    }
    return counter;
}

template <class T, class matrix_type, class vector_type>
Integer indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::left_nnz() const {
    Integer sum=0;
    for(Integer k=0;k<number_levels;k++) sum += Precond_left.get(k).actual_non_zeroes()-Precond_left.get(k).rows();
    return sum;
}

template <class T, class matrix_type, class vector_type>
Integer indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::right_nnz() const {
    Integer sum=0;
    for(Integer k=0;k<number_levels;k++) sum += Precond_right.get(k).actual_non_zeroes()-Precond_right.get(k).rows();
    return sum;
}

template <class T, class matrix_type, class vector_type>
Integer indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::middle_nnz() const {
    Integer sum=0;
    for(Integer k=0;k<number_levels;k++) sum += Precond_middle.get(k).dimension();
    return sum;
}

template <class T, class matrix_type, class vector_type>
Real indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::memory(Integer k) const {
    if(non_fatal_error(k<0||k>=number_levels, "indirect_split_triangular_multilevel_preconditioner: memory: This level does not exist.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    return D_l.get(k).memory() + D_r.get(k).memory() + 
        inverse_permutation_columns.get(k).memory() + inverse_permutation_rows.get(k).memory() +
        permutation_columns.get(k).memory() + permutation_rows.get(k).memory() +
        Precond_left.get(k).memory() + Precond_middle.get(k).memory() + Precond_right.get(k).memory();
}

template <class T, class matrix_type, class vector_type>
Real indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::memory() const {
    Real mem = 0.0;
    for(Integer p = 0; p<number_levels; p++) mem += memory(p);
    return mem;
}

template <class T, class matrix_type, class vector_type>
  matrix_type indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::left_preconditioning_matrix(Integer k) {return Precond_left.get(k);}

template <class T, class matrix_type, class vector_type>
  matrix_type indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::right_preconditioning_matrix(Integer k){return Precond_right.get(k);}

template <class T, class matrix_type, class vector_type>
  void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::print(Integer k) const {
      std::cout<<"The left matrix of chosen level:"<<std::endl;
      std::cout<< extract_left_matrix(k).expand();
      std::cout<<"The right matrix of chosen level:"<<std::endl;
      std::cout<<extract_right_matrix(k).expand();
      std::cout<<"The middle matrix of chosen level:"<<std::endl;
      std::cout<<extract_middle_matrix(k);
      std::cout<<"Left Scaling"<<std::endl<<extract_left_scaling(k);
      std::cout<<"Right Scaling"<<std::endl<<extract_right_scaling(k);
      std::cout<<"Row Permutation"<<std::endl<<extract_permutation_rows(k);
      std::cout<<"Column Permutation"<<std::endl<<extract_permutation_columns(k);
            }

template <class T, class matrix_type, class vector_type>
  void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::print_info(Integer k) const {
      std::cout<<" left matrix at level " << k << ":\n";
      Precond_left.get(k).print_info();
      std::cout<<" right matrix at level " << k << ":\n";
      Precond_right.get(k).print_info();
      std::cout<<" middle matrix at level " << k << ":\n";
      Precond_middle.get(k).print_info();
            }

template <class T, class matrix_type, class vector_type>
  void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::print_info() const {
      std::cout<<"A multilevel incomplete LDU preconditioner:"<<std::endl;
      for(Integer k=0;k<number_levels;k++){
          std::cout<<"* Level: "<<k<<std::endl; 
          print_info(k);
      }
      std::cout<<"\n";
      std::cout<<"    Memory: needed to store preconditioner        (kB): "<<this->memory()/1000.0<<std::endl;
      std::cout<<"            needed to calculate preconditioner    (kB): "<<this->memory_used_calculations()/1000.0<<std::endl;
      std::cout<<"            allocated to calculate preconditioner (kB): "<<this->memory_allocated_calculations()/1000.0<<std::endl;
      std::cout<<"    Non-zero elements in factors of preconditioner    : "<<this->total_nnz()<<std::endl;
  }

template <class T, class matrix_type, class vector_type>
  void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::print() const {
      std::cout<<"A multilevel incomplete LDU preconditioner:"<<std::endl;
      for(Integer k=0;k<number_levels;k++){
          std::cout<<"* Level: "<<k<<std::endl; 
          print(k);}
  }

template <class T, class matrix_type, class vector_type>
void indirect_split_triangular_multilevel_preconditioner<T,matrix_type,vector_type>::clear(){
    this->pre_image_size=0;
    this->image_size=0;
    this->intermediate_size=0;
    init(0);
}


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: indirect_split_pseudo_triangular_preconditioner                                                    //
//              (if permutation is applied to the columns Precond_right, a upper-triangular matrix results.              //
//                                                                                                                       //
//***********************************************************************************************************************//

template <class T, class matrix_type, class vector_type>
void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::apply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const {
    switch(left_form){
        case LOWER_TRIANGULAR:
            Precond_left.triangular_solve(LOWER_TRIANGULAR,use,v,w);
            break;
        case UPPER_TRIANGULAR:
            Precond_left.triangular_solve(UPPER_TRIANGULAR,use,v,w);
            break;
        case PERMUTED_LOWER_TRIANGULAR:
            if(left_matrix_usage == PERM1)
                Precond_left.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation,v,w);
            else
                Precond_left.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation2,v,w);
            break;
        case PERMUTED_UPPER_TRIANGULAR:
            if(left_matrix_usage == PERM1)
                Precond_left.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation,v,w);
            else
                Precond_left.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation2,v,w);
            break;
        default:
            std::cerr<<"indirect_split_pseudo_triangular_preconditioner::apply_preconditioner_left: only triangular forms are allowed."<<std::endl;
    }
}

template <class T, class matrix_type, class vector_type>
void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::apply_preconditioner_left(matrix_usage_type use, vector_type &w) const {
    switch(left_form){
        case LOWER_TRIANGULAR:
            Precond_left.triangular_solve(LOWER_TRIANGULAR,use,w);
            break;
        case UPPER_TRIANGULAR:
            Precond_left.triangular_solve(UPPER_TRIANGULAR,use,w);
            break;
        case PERMUTED_LOWER_TRIANGULAR:
            if(left_matrix_usage == PERM1)
                Precond_left.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation,w);
            else
                Precond_left.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation2,w);
            break;
        case PERMUTED_UPPER_TRIANGULAR:
            if(left_matrix_usage == PERM1)
                Precond_left.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation,w);
            else
                Precond_left.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation2,w);
            break;
        default:
            std::cerr<<"indirect_split_pseudo_triangular_preconditioner::apply_preconditioner_left: only triangular forms are allowed."<<std::endl;
    }
}

template <class T, class matrix_type, class vector_type>
void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::apply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const {
    switch(right_form){
        case LOWER_TRIANGULAR:
            Precond_right.triangular_solve(LOWER_TRIANGULAR,use,v,w);
            break;
        case UPPER_TRIANGULAR:
            Precond_right.triangular_solve(UPPER_TRIANGULAR,use,v,w);
            break;
        case PERMUTED_LOWER_TRIANGULAR:
            if(right_matrix_usage == PERM1)
                Precond_right.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation,v,w);
            else
                Precond_right.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation2,v,w);
            break;
        case PERMUTED_UPPER_TRIANGULAR:
            if(right_matrix_usage == PERM1)
                Precond_right.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation,v,w);
            else
                Precond_right.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation2,v,w);
            break;
        default:
            std::cerr<<"indirect_split_pseudo_triangular_preconditioner::apply_preconditioner_right: only triangular forms are allowed."<<std::endl;
    }
}

template <class T, class matrix_type, class vector_type>
void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::apply_preconditioner_right(matrix_usage_type use,vector_type &w) const {
    switch(right_form){
        case LOWER_TRIANGULAR:
            Precond_right.triangular_solve(LOWER_TRIANGULAR,use,w);
            break;
        case UPPER_TRIANGULAR:
            Precond_right.triangular_solve(UPPER_TRIANGULAR,use,w);
            break;
        case PERMUTED_LOWER_TRIANGULAR:
            if(right_matrix_usage == PERM1)
                Precond_right.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation,w);
            else
                Precond_right.triangular_solve_perm(PERMUTED_LOWER_TRIANGULAR,use,permutation2,w);
            break;
        case PERMUTED_UPPER_TRIANGULAR:
            if(right_matrix_usage == PERM1)
                Precond_right.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation,w);
            else
                Precond_right.triangular_solve_perm(PERMUTED_UPPER_TRIANGULAR,use,permutation2,w);
            break;
        default:
            std::cerr<<"indirect_split_pseudo_triangular_preconditioner::apply_preconditioner_left: only triangular forms are allowed."<<std::endl;
    }
}

template <class T, class matrix_type, class vector_type>
  void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::unapply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const{
      std::cerr<<"indirect_split_pseudo_triangular_preconditioner::unapply_preconditioner_left: undoing this preconditioner is not yet implemented."<<std::endl;
      throw iluplusplus_error(OTHER_ERROR);
  }

template <class T, class matrix_type, class vector_type>
  void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::unapply_preconditioner_left(matrix_usage_type use,vector_type &w) const{
      std::cerr<<"indirect_split_pseudo_triangular_preconditioner::unapply_preconditioner_left: undoing this preconditioner is not yet implemented."<<std::endl;
      throw iluplusplus_error(OTHER_ERROR);
  }

template <class T, class matrix_type, class vector_type>
  void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::unapply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const{
      std::cerr<<"indirect_split_pseudo_triangular_preconditioner::unapply_preconditioner_right: undoing this preconditioner is not yet implemented."<<std::endl;
      throw iluplusplus_error(OTHER_ERROR);
  }

template <class T, class matrix_type, class vector_type>
  void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::unapply_preconditioner_right(matrix_usage_type use,vector_type &w) const{
      std::cerr<<"indirect_split_pseudo_triangular_preconditioner::unapply_preconditioner_right: undoing this preconditioner is not yet implemented."<<std::endl;
      throw iluplusplus_error(OTHER_ERROR);
    }

template <class T, class matrix_type, class vector_type>
  void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::print_info() const {
      std::cout<<"The left matrix of the preconditioner:"<<std::endl;
      Precond_left.print_info();
      std::cout<<"The right matrix of the preconditioner:"<<std::endl;
      Precond_right.print_info();
  }

template <class T, class matrix_type, class vector_type>
void indirect_split_pseudo_triangular_preconditioner<T, matrix_type, vector_type>::eliminate_permutations(matrix_type& A, vector_type &b){
    matrix_type H,L;
    vector_type h;
    switch(left_matrix_usage){
        case PERM1: {
                        if(Precond_left.orient()==COLUMN){
                            H.change_orientation_of_data(Precond_left);
                            Precond_left.permute(H,permutation);
                        } else {
                            H=Precond_left;
                            Precond_left.permute(H,permutation);
                        }
                    } break;
        case PERM2: {
                        if(Precond_left.orient()==COLUMN){
                            H.change_orientation_of_data(Precond_left);
                            Precond_left.permute(H,permutation2);
                        } else {
                            H=Precond_left;
                            Precond_left.permute(H,permutation2);
                        }
                    } break;
        default: {}
    }
    switch(right_matrix_usage){
        case PERM1: {
                        if(Precond_right.orient()==ROW){ 
                            H.change_orientation_of_data(Precond_right);
                            Precond_right.permute(H,permutation); std::cout<<Precond_right.expand();
                        } else { 
                            H=Precond_right;
                            Precond_right.permute(H,permutation);
                        }
                    } break;
        case PERM2: {
                        if(Precond_right.orient()==ROW){
                            H.change_orientation_of_data(Precond_right);
                            Precond_right.permute(H,permutation2);
                        } else { 
                            H=Precond_right;
                            Precond_right.permute(H,permutation2);
                        }
                    } break;
        default: {}
    }
    if(A.orient()==ROW){
        switch (left_matrix_usage){
            case PERM1: H.permute(A,permutation); break;
            case PERM2: H.permute(A,permutation2); break;
            default:    H=A;
        }
        switch (right_matrix_usage){
            case PERM1: {L.change_orientation_of_data(H); A.permute(L,permutation);} break;
            case PERM2: {L.change_orientation_of_data(H); A.permute(L,permutation2);} break;
            default:    A=H;
        }
    } else {
        switch (right_matrix_usage){
            case PERM1: H.permute(A,permutation); break;
            case PERM2: H.permute(A,permutation2); break;
            default:    H=A;
        }
        switch (left_matrix_usage){
            case PERM1: {L.change_orientation_of_data(H); A.permute(L,permutation);} break;
            case PERM2: {L.change_orientation_of_data(H); A.permute(L,permutation2);} break;
            default:    A=H;
        }
        switch (left_matrix_usage){
            case PERM1: H.permute(A,permutation); break;
            case PERM2: H.permute(A,permutation2); break;
            default:    H=A;
        }
    } // end if
    switch (left_matrix_usage){
        case PERM1: {h.permute(b,permutation);  b=h;} break;
        case PERM2: {h.permute(b,permutation2); b=h;} break;
        default:    {}
    }
    permutation.resize(0);
    permutation2.resize(0);
    left_matrix_usage = NOPERM;
    right_matrix_usage = NOPERM;
}



//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: NullPrecondioner: does not precondition.                                                           //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
void NullPreconditioner<T,matrix_type,vector_type>::apply_preconditioner(matrix_usage_type use, const vector_type &v, vector_type &w) const {
    w=v;
}

template <class T, class matrix_type, class vector_type>
void NullPreconditioner<T,matrix_type,vector_type>::apply_preconditioner(matrix_usage_type use, vector_type &w) const {}

template <class T, class matrix_type, class vector_type>
void NullPreconditioner<T,matrix_type,vector_type>::unapply_preconditioner(matrix_usage_type use, const vector_type &v, vector_type &w) const{
    w=v;
}

template <class T, class matrix_type, class vector_type>
void NullPreconditioner<T,matrix_type,vector_type>::unapply_preconditioner(matrix_usage_type use, vector_type &w) const{}

template <class T, class matrix_type, class vector_type>
  NullPreconditioner<T,matrix_type,vector_type>::NullPreconditioner() {
      this->pre_image_size=0;
      this->image_size=0;
      this->setup_time = 0.0;
      this->memory_allocated_to_create=0.0;
      this->memory_used_to_create=0.0;
  }

template <class T, class matrix_type, class vector_type>
  NullPreconditioner<T,matrix_type,vector_type>::NullPreconditioner(Integer m, Integer n){
      this->image_size=m;
      this->pre_image_size=n;
      this->setup_time = 0.0;
      this->memory_allocated_to_create=0.0;
      this->memory_used_to_create=0.0;
  }

template <class T, class matrix_type, class vector_type>
  void NullPreconditioner<T,matrix_type,vector_type>::read_binary(std::string filename) {}

template <class T, class matrix_type, class vector_type>
  void NullPreconditioner<T,matrix_type,vector_type>::write_binary(std::string filename) const {std::cerr<<"This function makes no sense for this preconditioner.";}

template <class T, class matrix_type, class vector_type>
  void NullPreconditioner<T,matrix_type,vector_type>::print_info() const {
      std::cout<<"NullPreconditioner does not precondition."<<std::endl;
      std::cout<<"  pre-image size: "<<this->pre_image_size<<std::endl;
      std::cout<<"  image size:     "<<this->image_size<<std::endl;
}

//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUC Preconditioner (Saad):                                                                         //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
ILUCPreconditioner<T,matrix_type,vector_type>::ILUCPreconditioner(const matrix_type &A, Integer max_fill_in, Real threshold){
    const clock_t time_begin = clock();
    if (A.orient() == ROW) {
        this->preconditioner_exists = ILUC2(A, this->Precond_left, this->Precond_right, max_fill_in, threshold);
        this->left_form=LOWER_TRIANGULAR;
        this->right_form=UPPER_TRIANGULAR;
    } else {
        this->preconditioner_exists = ILUC2(A, this->Precond_right, this->Precond_left, max_fill_in, threshold);
        this->left_form=LOWER_TRIANGULAR;
        this->right_form=UPPER_TRIANGULAR;
        //this->Precond_left.transpose_in_place();
        //this->Precond_right.transpose_in_place();
    }
    this->setup_time = ((Real)clock() - (Real) time_begin) / (Real) CLOCKS_PER_SEC;
    this->pre_image_size=this->Precond_left.rows();
    this->image_size=this->Precond_right.columns();
    this->intermediate_size=this->Precond_left.columns();
    this->memory_allocated_to_create=0.0;
    this->memory_used_to_create=0.0;
}

template <class T, class matrix_type, class vector_type>
  void ILUCPreconditioner<T,matrix_type,vector_type>::write_binary(std::string filename) const {
      if(this->exists()){
          this->Precond_left.write_binary(filename+"-L.isp");
          this->Precond_right.write_binary(filename+"-R.isp");
      } else {
          matrix_type N;
          N.write_binary(filename+"-L.isp");
          N.write_binary(filename+"-R.isp");
      }
  }

template <class T, class matrix_type, class vector_type>
  void ILUCPreconditioner<T,matrix_type,vector_type>::read_binary(std::string filename){
      this->Precond_left.read_binary(filename+"-L.isp");
      this->Precond_right.read_binary(filename+"-R.isp");
      if(non_fatal_error((this->Precond_left.columns()!=this->Precond_right.rows()),"ILUC::read_binary: the dimensions of the two matrices are incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      this->image_size=this->Precond_right.columns();
      this->intermediate_size=this->Precond_right.rows();
      this->pre_image_size=this->Precond_left.rows();
  }



//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUT Preconditioner (Saad):                                                                        //
//                                                                                                                       //
//***********************************************************************************************************************//



template <class T, class matrix_type, class vector_type>
ILUTPreconditioner<T,matrix_type,vector_type>::ILUTPreconditioner(const matrix_type &A, Integer max_fill_in, Real threshold){
    if(A.orient()==ROW){
        // NOTE: ILUT2 does not yet work.
        this->preconditioner_exists = ILUT(A, this->Precond_left, this->Precond_right, max_fill_in, threshold, this->setup_time);
        this->left_form=LOWER_TRIANGULAR;
        this->right_form=UPPER_TRIANGULAR;
    } else {
        this->preconditioner_exists = ILUT(A, this->Precond_right, this->Precond_left, max_fill_in, threshold, this->setup_time);
        this->Precond_left.transpose_in_place();
        this->Precond_right.transpose_in_place();
        this->left_form=LOWER_TRIANGULAR;
        this->right_form=UPPER_TRIANGULAR;
    }
    this->pre_image_size=this->Precond_left.rows();
    this->image_size=this->Precond_right.columns();
    this->intermediate_size=this->Precond_left.columns();
    this->memory_allocated_to_create=0.0;
    this->memory_used_to_create=0.0;
}

template <class T, class matrix_type, class vector_type>
  std::string ILUTPreconditioner<T,matrix_type,vector_type>::special_info() const {
      std::ostringstream info;
      info<<"";
      return info.str();
  }

template <class T, class matrix_type, class vector_type>
  void ILUTPreconditioner<T,matrix_type,vector_type>::write_binary(std::string filename) const {
      if(this->exists()){
          this->Precond_left.write_binary(filename+"-L.isp");
          this->Precond_right.write_binary(filename+"-R.isp");
      } else {
          matrix_type N;
          N.write_binary(filename+"-L.isp");
          N.write_binary(filename+"-R.isp");
      }
  }

template <class T, class matrix_type, class vector_type>
  void ILUTPreconditioner<T,matrix_type,vector_type>::read_binary(std::string filename){
      this->Precond_left.read_binary(filename+"-L.isp");
      this->Precond_right.read_binary(filename+"-R.isp");
      if(non_fatal_error((this->Precond_left.columns()!=this->Precond_right.rows()),"ILUT::read_binary: the dimensions of the two matrices are incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      this->image_size=this->Precond_right.columns();
      this->intermediate_size=this->Precond_right.rows();
      this->pre_image_size=this->Precond_left.rows();
  }

template <class T, class matrix_type, class vector_type>
  Integer ILUTPreconditioner<T,matrix_type,vector_type>::left_nnz() const {return this->Precond_left.actual_non_zeroes()-this->image_size;}

template <class T, class matrix_type, class vector_type>
  Integer ILUTPreconditioner<T,matrix_type,vector_type>::right_nnz() const {return this->Precond_right.actual_non_zeroes();}


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUTP Preconditioner (Saad):                                                                       //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
ILUTPPreconditioner<T,matrix_type,vector_type>::ILUTPPreconditioner(const matrix_type &A, Integer max_fill_in, Real threshold, Real perm_tol, Integer row_pos, Real mem_factor)
{
    if(A.orient()==ROW){
        this->preconditioner_exists = ILUTP2(A, this->Precond_left,
                this->Precond_right, this->permutation, max_fill_in, threshold,
                perm_tol, row_pos, this->zero_pivots, this->setup_time, mem_factor);
        this->left_form=LOWER_TRIANGULAR;
        this->right_form=PERMUTED_UPPER_TRIANGULAR;
        this->left_matrix_usage = NOPERM;
        this->right_matrix_usage = PERM1;
    } else {
        this->preconditioner_exists = ILUTP2(A, this->Precond_right,
                this->Precond_left, this->permutation, max_fill_in, threshold,
                perm_tol, row_pos, this->zero_pivots, this->setup_time, mem_factor);
        this->Precond_left.transpose_in_place();
        this->Precond_right.transpose_in_place();
        this->left_form=PERMUTED_LOWER_TRIANGULAR;
        this->right_form=UPPER_TRIANGULAR;
        this->left_matrix_usage = PERM1;
        this->right_matrix_usage = NOPERM;
    }
    this->pre_image_size=this->Precond_left.rows();
    this->image_size=this->Precond_right.columns();
    this->intermediate_size=this->Precond_left.columns();
    this->memory_allocated_to_create=0.0;
    this->memory_used_to_create=0.0;
}

template <class T, class matrix_type, class vector_type>
  std::string ILUTPPreconditioner<T,matrix_type,vector_type>::special_info() const {
      std::ostringstream info;
      if (this->zero_pivots == 0) info<<"";
      else info<<"*";
      return info.str();
  }

template <class T, class matrix_type, class vector_type>
  void ILUTPPreconditioner<T,matrix_type,vector_type>::write_binary(std::string filename) const {
      if(this->exists()){
          this->Precond_left.write_binary(filename+"-L.isp");
          this->Precond_right.write_binary(filename+"-R.isp");
      } else {
          matrix_type N;
          N.write_binary(filename+"-L.isp");
          N.write_binary(filename+"-R.isp");
      }
  }

template <class T, class matrix_type, class vector_type>
  void ILUTPPreconditioner<T,matrix_type,vector_type>::read_binary(std::string filename){
      this->Precond_left.read_binary(filename+"-L.isp");
      this->Precond_right.read_binary(filename+"-R.isp");
      if(non_fatal_error((this->Precond_left.columns()!=this->Precond_right.rows()),"ILUTP::read_binary: the dimensions of the two matrices are incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      this->image_size=this->Precond_right.columns();
      this->intermediate_size=this->Precond_right.rows();
      this->pre_image_size=this->Precond_left.rows();
  }


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUCP Preconditioner:                                                                              //
//                                                                                                                       //
//***********************************************************************************************************************//

template <class T, class matrix_type, class vector_type>
ILUCPPreconditioner<T,matrix_type,vector_type>::ILUCPPreconditioner(const
        matrix_type &Acol, Integer max_fill_in, Real threshold, Real perm_tol,
        Integer rp, Real mem_factor)
{
    if(Acol.orient()==COLUMN){
        this->preconditioner_exists = ILUCP4(Acol, this->Precond_left,
                this->Precond_right, this->permutation, max_fill_in, threshold,
                perm_tol, rp, this->zero_pivots, this->setup_time, mem_factor);
        this->left_form=LOWER_TRIANGULAR;
        this->right_form=PERMUTED_UPPER_TRIANGULAR;
        this->left_matrix_usage = NOPERM;
        this->right_matrix_usage = PERM1;
    } else {
        this->preconditioner_exists = ILUCP4(Acol, this->Precond_right,
                this->Precond_left, this->permutation, max_fill_in, threshold,
                perm_tol, rp, this->zero_pivots, this->setup_time, mem_factor);
        this->left_form=PERMUTED_LOWER_TRIANGULAR;
        this->right_form=UPPER_TRIANGULAR;
        this->left_matrix_usage = PERM1;
        this->right_matrix_usage = NOPERM;
    }
    this->pre_image_size=this->Precond_left.rows();
    this->image_size=this->Precond_right.columns();
    this->intermediate_size=this->Precond_left.columns();
    this->memory_allocated_to_create=0.0;
    this->memory_used_to_create=0.0;
}

template <class T, class matrix_type, class vector_type>
ILUCPPreconditioner<T,matrix_type,vector_type>::ILUCPPreconditioner(const matrix_type &Acol, const ILUCP_precond_parameter& p)
    : ILUCPPreconditioner<T,matrix_type,vector_type>(Acol, p.get_fill_in(), p.get_threshold(), p.get_perm_tol(), p.get_row_pos())
{
}

template <class T, class matrix_type, class vector_type>
  std::string ILUCPPreconditioner<T,matrix_type,vector_type>::special_info() const {
      std::ostringstream info;
      if (this->zero_pivots == 0) info<<"";
      else info<<"*";
      return info.str();
  }

template <class T, class matrix_type, class vector_type>
  void ILUCPPreconditioner<T,matrix_type,vector_type>::write_binary(std::string filename) const {
      if(this->exists()){
          this->Precond_left.write_binary(filename+"-L.isp");
          this->Precond_right.write_binary(filename+"-R.isp");
      } else {
          matrix_type N;
          N.write_binary(filename+"-L.isp");
          N.write_binary(filename+"-R.isp");
      }
  }

template <class T, class matrix_type, class vector_type>
  void ILUCPPreconditioner<T,matrix_type,vector_type>::read_binary(std::string filename){
      this->Precond_left.read_binary(filename+"-L.isp");
      this->Precond_right.read_binary(filename+"-R.isp");
      if(non_fatal_error((this->Precond_left.columns()!=this->Precond_right.rows()),"ILUCP::read_binary: the dimensions of the two matrices are incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      this->image_size=this->Precond_right.columns();
      this->intermediate_size=this->Precond_right.rows();
      this->pre_image_size=this->Precond_left.rows();
  }


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUCDP Preconditioner:                                                                             //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
ILUCDPPreconditioner<T,matrix_type,vector_type>::ILUCDPPreconditioner(const matrix_type &Arow, const matrix_type &Acol, Integer max_fill_in, Real threshold, Real perm_tol, Integer bpr){
    if(Acol.orient()==COLUMN && Arow.orient()==ROW){
        this->preconditioner_exists = this->Precond_left.ILUCDP(Arow,Acol,this->Precond_right,this->permutation,this->permutation2,max_fill_in,threshold,perm_tol,bpr,this->zero_pivots,this->setup_time);      // preconditioner of A.
        this->left_form=PERMUTED_LOWER_TRIANGULAR;
        this->right_form=PERMUTED_UPPER_TRIANGULAR;
        this->left_matrix_usage = PERM2;
        this->right_matrix_usage = PERM1;
    } else {
        std::cerr<<"ILUCDPPreconditioner::ILUCDPPreconditioner: Matrix needs to be provided in CSR and CSC format."<<std::endl;
    }
    this->pre_image_size=this->Precond_left.rows();
    this->image_size=this->Precond_right.columns();
    this->intermediate_size=this->Precond_left.columns();
    this->memory_allocated_to_create=0.0;
    this->memory_used_to_create=0.0;
}


template <class T, class matrix_type, class vector_type>
ILUCDPPreconditioner<T,matrix_type,vector_type>:: ILUCDPPreconditioner(const matrix_type &Arow, const matrix_type &Acol, matrix_type &Anew, Integer max_fill_in, Real threshold, Real perm_tol, Integer bpr){
    index_list invperm,invperm2; Integer last_row;
    Real memory_allocated_factorization, memory_used_factorization;
    if(Acol.orient()==COLUMN && Arow.orient()==ROW){
        last_row=4;
        this->preconditioner_exists = this->Precond_left.partialILUCDP(Arow,Acol,Anew,this->Precond_right,this->permutation,this->permutation2,invperm,invperm2,last_row,threshold,bpr,this->zero_pivots,this->setup_time,10.0,memory_allocated_factorization,memory_used_factorization);      // preconditioner of A.
        this->left_form=PERMUTED_LOWER_TRIANGULAR;
        this->right_form=PERMUTED_UPPER_TRIANGULAR;
        this->left_matrix_usage = PERM2;
        this->right_matrix_usage = PERM1;
    } else {
        std::cerr<<"ILUCDPPreconditioner::ILUCDPPreconditioner: Matrix needs to be provided in CSR and CSC format."<<std::endl;
    }
    this->pre_image_size=this->Precond_left.rows();
    this->image_size=this->Precond_right.columns();
    this->intermediate_size=this->Precond_left.columns();
    this->memory_allocated_to_create=0.0;
    this->memory_used_to_create=0.0;
}


template <class T, class matrix_type, class vector_type>
ILUCDPPreconditioner<T,matrix_type,vector_type>::ILUCDPPreconditioner(const matrix_type &Arow, const matrix_type &Acol, const ILUCDP_precond_parameter& p)
    :   ILUCDPPreconditioner<T,matrix_type,vector_type>::ILUCDPPreconditioner(Arow, Acol, p.get_fill_in(),
            p.get_threshold(), p.get_perm_tol(), p.get_begin_perm_row())
{}

template <class T, class matrix_type, class vector_type>
  Integer ILUCDPPreconditioner<T,matrix_type,vector_type>::zero_pivots_encountered(){return this->zero_pivots;}

template <class T, class matrix_type, class vector_type>
  std::string ILUCDPPreconditioner<T,matrix_type,vector_type>::special_info() const {
      std::ostringstream info;
      if (this->zero_pivots == 0) info<<"";
      else info<<"*";
      return info.str();
  }

template <class T, class matrix_type, class vector_type>
  void ILUCDPPreconditioner<T,matrix_type,vector_type>::write_binary(std::string filename) const {
      if(this->exists()){
          this->Precond_left.write_binary(filename+"-L.isp");
          this->Precond_right.write_binary(filename+"-R.isp");
      } else {
          matrix_type N;
          N.write_binary(filename+"-L.isp");
          N.write_binary(filename+"-R.isp");
      }
  }

template <class T, class matrix_type, class vector_type>
  void ILUCDPPreconditioner<T,matrix_type,vector_type>::read_binary(std::string filename){
      this->Precond_left.read_binary(filename+"-L.isp");
      this->Precond_right.read_binary(filename+"-R.isp");
      if(non_fatal_error((this->Precond_left.columns()!=this->Precond_right.rows()),"ILUCDP::read_binary: the dimensions of the two matrices are incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      this->image_size=this->Precond_right.columns();
      this->intermediate_size=this->Precond_right.rows();
      this->pre_image_size=this->Precond_left.rows();
  }


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: multilevelILUCDP Preconditioner:                                                                   //
//                                                                                                                       //
//***********************************************************************************************************************//

template <class T, class matrix_type, class vector_type>
  multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::multilevelILUCDPPreconditioner() {
      this->init(0);
      this->pre_image_size=0;
      this->image_size=0;
      this->intermediate_size=0;
      this->number_levels=0;
      this->preconditioner_exists=true;
      dim_zero_matrix_factored=0;
      this->memory_allocated_to_create= 0.0;
      this->memory_used_to_create=0.0;
}

template <class T, class matrix_type, class vector_type>
void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::init(Integer mem_levels){
    indirect_split_triangular_multilevel_preconditioner<T,matrix_type, vector_type>::init(mem_levels);
    zero_pivots.resize(mem_levels);
    this->memory_allocated_to_create = 0.0;
    this->memory_used_to_create=0.0;
}

template <class T, class matrix_type, class vector_type>
iluplusplus_precond_parameter multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::extract_parameters() const {
    return param;
}

template <class T, class matrix_type, class vector_type>
Integer multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::zero_pivots_encountered(Integer k) const {
#ifdef DEBUG
    if(non_fatal_error(k<0||k>=this->number_levels, "multilevelILUCDPPreconditioner::zero_pivots_encountered: This level does not exist.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    return this->zero_pivots[k];
}


template <class T, class matrix_type, class vector_type>
Integer multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::dimension_zero_matrix_factored() const {
    return this->dim_zero_matrix_factored;
}


template <class T, class matrix_type, class vector_type>
  Integer multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::zero_pivots_encountered() const {
      Integer sum = 0;
      for(Integer k=0;k<this->levels();k++) sum += this->zero_pivots_encountered(k);
      return sum;
  }

template <class T, class matrix_type, class vector_type>
  std::string multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::special_info() const {
      std::ostringstream info;
      bool zp=false;
      for(Integer k=0;k<this->number_levels;k++)
          zp |= (this->zero_pivots[k] == 0);
      if (zp) info<<"-"<<this->number_levels;
      else info<<"-"<<this->number_levels<<"*";
      if(dim_zero_matrix_factored>0) info << "#"<<this->dim_zero_matrix_factored;
      return info.str();
  }

template <class T, class matrix_type, class vector_type>
  void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::write_binary(std::string filename) const {
       std::cout<<"multilevelILUCDP::write_binary: not implemented yet"<<std::endl;
  }

template <class T, class matrix_type, class vector_type>
  void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::read_binary(std::string filename){
       std::cout<<"multilevelILUCDP::read_binary: not implemented yet"<<std::endl;
  }

template <class T, class matrix_type, class vector_type>
void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::make_preprocessed_multilevelILUCDP(const matrix_type &A, const iluplusplus_precond_parameter& IP){
    if(IP.get_PRECON_PARAMETER() < 0){
        std::cout<<"multilevelILUCDPPreconditioner::make_preprocessed_multilevelILUCDP: PRECON_PARAMETER < 0: these values are reserved for special solvers. Please choose permissible value. Doing nothing and returning without preconditioner."<<std::endl;
        return;
    }
    this->memory_allocated_to_create = 0.0;
    this->memory_used_to_create = 0.0;
    Real memory_allocated_factorization = 0.0;
    Real memory_used_factorization = 0.0;
    Real max_memory_allocated = 0.0;
    Real max_memory_used = 0.0;
    Real memory_currently_used = 0.0;
    Real memory_currently_allocated = 0.0;
    Real memory_previous_levels = 0.0;
    Real Amemory = A.memory();
    Real memory_matrices = 0.0;
#ifdef INFO
    Integer equality_permutations = 0;
    Integer total_equality_permutations = 0;
#endif
    clock_t time_1,time_2;
    time_1 = clock();
    init(IP.get_MEMORY_MAX_LEVELS());
    param = IP;
    Real tau = IP.get_threshold();
    bool use_ILUC;
    if ((IP.get_PERMUTE_ROWS() == 0 || (IP.get_PERMUTE_ROWS() == 1 && !IP.get_EXTERNAL_FINAL_ROW()))
            && (!IP.get_BEGIN_TOTAL_PIV() || (IP.get_BEGIN_TOTAL_PIV() && IP.get_TOTAL_PIV() == 0) )
            && IP.get_perm_tol() > 500.0)
        use_ILUC = true;
    else
        use_ILUC = false;
    bool have_zero_matrix = false;
    matrix_type Akrow,Akcol,Akrow_next;
    index_list pr1, pr2, ipr1,ipr2,pc1,pc2,ipc1,ipc2;
    Real partial_setup_time;
    Real mem_factor=IP.get_MEM_FACTOR();
    this->setup_time = 0.0;
    Integer last_row_to_eliminate,bp,bpr,epr,end_PQ;
    if(A.orient() == ROW) Akrow = A;
    else Akrow.change_orientation_of_data(A);
    Integer matrix_size = Akrow.rows();
    Integer nonzeroes = Akrow.actual_non_zeroes();
    this->preconditioner_exists = true;
    this->number_levels=0;
    if(IP.get_PREPROCESSING().dim()<2){  // in this case, only one copy of matrix is needed (total of 2 matrices) 
        max_memory_allocated = 2.0 * Amemory;
        max_memory_used = 2.0 * Amemory;
    } else {
        max_memory_allocated = 3.0 * Amemory;  // in this case, two copies are neeed (total of 3 matrices)
        max_memory_used = 3.0 * Amemory;
    }
    while(matrix_size>IP.get_MIN_ML_SIZE() && this->number_levels<IP.get_MAX_LEVELS()-1 && nonzeroes>0){
        have_zero_matrix =  IP.get_USE_THRES_ZERO_SCHUR() && Akrow.rows()<= IP.get_MIN_SIZE_ZERO_SCHUR() && Akrow.numerical_zero_check(IP.get_THRESHOLD_ZERO_SCHUR());
        if(have_zero_matrix) {
#ifdef INFO
            std::cout<<"multilevelILUCDPPreconditioner::make_preprocessed_multilevelILUCDP: zero coefficient matrix of dimension "<<Akrow.rows()<<", calculating Drazin inverse."<<std::endl;
#endif
            dim_zero_matrix_factored = Akrow.rows();
            Akrow_next.reformat(0,0,0,ROW);
            this->Precond_left[this->number_levels].square_diag(Akrow.rows(),1.0,COLUMN);
            this->Precond_right[this->number_levels].square_diag(Akrow.rows(),1.0,ROW);
            this->Precond_middle[this->number_levels].resize(Akrow.rows(),1.0);  // (almost) Drazin inverse.
            this->zero_pivots[this->number_levels] = Akrow.rows();
            this->permutation_columns[this->number_levels].resize(Akrow.rows());
            this->permutation_rows[this->number_levels ].resize(Akrow.rows());
            this->inverse_permutation_columns[this->number_levels].resize(Akrow.rows());
            this->inverse_permutation_rows[this->number_levels].resize(Akrow.rows());
            this->number_levels++;
            matrix_size = 0;
            nonzeroes = 0;
#ifdef INFO
            equality_permutations = Akrow.rows();
            total_equality_permutations += equality_permutations;
            std::cout<<"     symmetry of permutations used in factorization          = "<<(Real) 1.0 <<std::endl;
#endif
        } else {
            end_PQ = Akrow.preprocess(IP,pr1,pc1,ipr1,ipc1,this->D_l[this->number_levels],this->D_r[this->number_levels]);
            if (IP.get_EXTERNAL_FINAL_ROW()) last_row_to_eliminate = end_PQ-1;
            else last_row_to_eliminate = (Akrow.rows()-1)/2;
            switch (IP.get_PERMUTE_ROWS()) {
                case 0:  bpr = 0; epr = 0; break;
                case 1:  if(IP.get_EXTERNAL_FINAL_ROW()){bpr = 0; epr = 0;} else {bpr = end_PQ; epr = Akrow.rows()-1;} break;
                case 2:  bpr = 0; epr = last_row_to_eliminate; break;
                case 3:  bpr = 0; epr = Akrow.rows()-1; break;
                default: std::cerr<<"choose permissible value for PERMUTE_ROWS!"<<std::endl; throw iluplusplus_error(OTHER_ERROR);
            }
            switch (IP.get_TOTAL_PIV()) {
                case 0:  bp = Akrow.rows(); break;
                case 1:  bp = last_row_to_eliminate+1; break;
                case 2:  bp = 0;  break;
                default: std::cerr<<"choose permissible value for TOTAL_PIV!"<<std::endl; throw iluplusplus_error(OTHER_ERROR);
            }
            if(!use_ILUC) Akcol.change_orientation_of_data(Akrow);
#ifdef INFO
            std::cout<<std::endl;
            std::cout<<"**** level: "<<this->number_levels<<" ****"<<std::endl;
            std::cout<<"  ** matrix statistics:"<<std::endl;
            std::cout<<"     n                      = "<<Akrow.rows()<<std::endl;
            std::cout<<"     nnz                    = "<<Akrow.actual_non_zeroes()<<std::endl;
            std::cout<<"     density                = "<<Akrow.row_density()<<std::endl;
            std::cout<<"  ** preconditioner parameters:"<<std::endl;
            std::cout<<"     max. numb. nnz/row p   = "<<IP.get_fill_in()<<std::endl;
            std::cout<<"     tau                    = "<<tau<<std::endl;
            std::cout<<"     perm tolerance         = "<<IP.get_perm_tol()<<std::endl;
            std::cout<<"     begin permuting rows   = "<<bpr<<std::endl;
            std::cout<<"     end   permuting rows   = "<<epr<<std::endl;
            if(IP.get_EXTERNAL_FINAL_ROW())
                std::cout<<"     last row to eliminate  = "<<last_row_to_eliminate;
            else
                std::cout<<"     last row to eliminate decided by preconditioner."<<std::endl;
            std::cout<<std::endl;
#endif

            if(use_ILUC)
                this->preconditioner_exists &= this->Precond_left[this->number_levels].partialILUC(
                        Akrow, Akrow_next, IP, false, this->Precond_right[this->number_levels],
                        this->Precond_middle[this->number_levels], last_row_to_eliminate, tau,
                        this->zero_pivots[this->number_levels], partial_setup_time, mem_factor,
                        memory_allocated_factorization, memory_used_factorization);
            else
                this->preconditioner_exists &= this->Precond_left[this->number_levels].partialILUCDP(
                        Akrow, Akcol, Akrow_next, IP, false, this->Precond_right[this->number_levels],
                        this->Precond_middle[this->number_levels], pc2, pr2, ipc2, ipr2, last_row_to_eliminate, tau,
                        bp, bpr, epr, this->zero_pivots[this->number_levels], partial_setup_time, mem_factor,
                        memory_allocated_factorization, memory_used_factorization);

            if (!this->preconditioner_exists)
                return;

#ifdef INFO
            std::cout<<"     zero-pivots            = "<<this->zero_pivots[this->number_levels]<<std::endl;
            std::cout<<"     local fill-in          = "<<((Real)(this->Precond_left[this->number_levels].actual_non_zeroes()+this->Precond_right[this->number_levels].actual_non_zeroes())- (Real) Akrow.rows() )/((Real)Akrow.actual_non_zeroes())<<std::endl;
            if(use_ILUC) equality_permutations = Akrow.rows()-Akrow_next.rows();
            else equality_permutations = pc2.equality(pr2,0,Akrow.rows()-Akrow_next.rows());
            total_equality_permutations += equality_permutations;
            std::cout<<"     symmetry of permutations used in factorization          = "<<(Real) equality_permutations/(Real)(Akrow.rows()-Akrow_next.rows()) <<std::endl;
#endif

            if(use_ILUC){
                this->permutation_columns[this->number_levels] = pc1;
                this->permutation_rows[this->number_levels ]= pr1;                  
            } else {
                this->permutation_columns[this->number_levels].compose(pc1,pc2);
                this->permutation_rows[this->number_levels].compose(pr1,pr2);                  
            }
            this->inverse_permutation_columns[this->number_levels].invert(this->permutation_columns[this->number_levels]);
            this->inverse_permutation_rows[this->number_levels].invert(this->permutation_rows[this->number_levels]);
            this->number_levels++;
            memory_matrices = Amemory + Akrow.memory() + Akcol.memory(); // Akrow_next is not needed, as this is included in the memory counted by the factorization.
            memory_currently_used = memory_matrices + memory_used_factorization + memory_previous_levels;
            memory_currently_allocated = memory_matrices + memory_allocated_factorization + memory_previous_levels;
            if(memory_currently_allocated > max_memory_allocated) max_memory_allocated = memory_currently_allocated;
            if(memory_currently_used > max_memory_used) max_memory_used = memory_currently_used;
            memory_previous_levels += this->memory(this->number_levels-1);
            Akrow.interchange(Akrow_next);
            this->setup_time += partial_setup_time;
            matrix_size = Akrow.rows();
            nonzeroes = Akrow.actual_non_zeroes();
            if(!IP.get_EXTERNAL_FINAL_ROW()) last_row_to_eliminate = min(last_row_to_eliminate/2,matrix_size-1);
            tau += IP.get_VARY_THRESHOLD_FACTOR();
            switch(IP.get_VARIABLE_MEM()){
                case 0: mem_factor = IP.get_MEM_FACTOR(); break;
                case 1: mem_factor = IP.get_MEM_FACTOR()*((Real)A.rows())/((Real)matrix_size); break;
                case 2: mem_factor = IP.get_MEM_FACTOR()*((Real)A.rows())/((Real)matrix_size)*((Real)A.rows())/((Real)matrix_size); break;
                default: std::cerr<<"Please use permissible value for VARIABLE_MEM."<<std::endl; throw iluplusplus_error(OTHER_ERROR);
            }
        }
    } // end while
    if(matrix_size>0){
        have_zero_matrix =  IP.get_USE_THRES_ZERO_SCHUR() && Akrow.rows()<= IP.get_MIN_SIZE_ZERO_SCHUR() && Akrow.numerical_zero_check(IP.get_THRESHOLD_ZERO_SCHUR());
        if(have_zero_matrix) {
#ifdef INFO
            std::cout<<"multilevelILUCDPPreconditioner::make_preprocessed_multilevelILUCDP: zero coefficient matrix, calculating Drazin inverse."<<std::endl;
#endif
            dim_zero_matrix_factored = Akrow.rows();
            Akrow_next.reformat(0,0,0,ROW);
            this->Precond_left[this->number_levels].square_diag(Akrow.rows(),1.0,COLUMN);
            this->Precond_right[this->number_levels].square_diag(Akrow.rows(),1.0,ROW);
            this->Precond_middle[this->number_levels].resize(Akrow.rows(),0.0);  // Drazin inverse
            this->zero_pivots[this->number_levels] = Akrow.rows();
            this->permutation_columns[this->number_levels].resize(Akrow.rows());
            this->permutation_rows[this->number_levels ].resize(Akrow.rows());
            this->inverse_permutation_columns[this->number_levels].resize(Akrow.rows());
            this->inverse_permutation_rows[this->number_levels].resize(Akrow.rows());
            this->number_levels++;
            matrix_size = 0;
            nonzeroes = 0;
#ifdef INFO
            equality_permutations = Akrow.rows();
            total_equality_permutations += equality_permutations;
            std::cout<<"     symmetry of permutations used in factorization          = "<<(Real) 1.0 <<std::endl;
#endif
        } else {
            end_PQ = Akrow.preprocess(IP,pr1,pc1,ipr1,ipc1,this->D_l[this->number_levels],this->D_r[this->number_levels]);
            last_row_to_eliminate = Akrow.rows()-1;  // must complete LU factorisation
            switch (IP.get_PERMUTE_ROWS()) {
                case 0:  bpr = 0; epr = 0; break;                   // never permute
                case 1:  bpr = end_PQ; epr = Akrow.rows()-1; break; // only permute rows unaffected by PQ
                case 2:  bpr = 0; epr = Akrow.rows()-1; break;
                case 3:  bpr = 0; epr = Akrow.rows()-1; break;
                default: std::cerr<<"choose permissible value for PERMUTE_ROWS!"<<std::endl; throw iluplusplus_error(OTHER_ERROR);
            }
            switch (IP.get_TOTAL_PIV()) {
                case 0:  bp = Akrow.rows(); break;
                case 1:  bp = end_PQ; break;
                case 2:  bp = 0;  break;
                default: std::cerr<<"choose permissible value for TOTAL_PIV!"<<std::endl; throw iluplusplus_error(OTHER_ERROR);
            }
            if(!use_ILUC) Akcol.change_orientation_of_data(Akrow);
            if (IP.get_USE_FINAL_THRESHOLD()) tau += IP.get_FINAL_THRESHOLD();
#ifdef INFO
            std::cout<<std::endl;
            std::cout<<"**** level: "<<this->number_levels<<" ****"<<std::endl;
            std::cout<<"  ** matrix statistics:"<<std::endl;
            std::cout<<"     n                      = "<<Akrow.rows()<<std::endl;
            std::cout<<"     nnz                    = "<<Akrow.actual_non_zeroes()<<std::endl;
            std::cout<<"     density                = "<<Akrow.row_density()<<std::endl;
            std::cout<<"  ** preconditioner parameters:"<<std::endl;
            std::cout<<"     max. numb. nnz/row p   = "<<IP.get_fill_in()<<std::endl;
            std::cout<<"     tau                    = "<<tau<<std::endl;
            std::cout<<"     perm tolerance         = "<<IP.get_perm_tol()<<std::endl;
            std::cout<<"     begin permuting rows   = "<<bpr<<std::endl;
            std::cout<<"     end   permuting rows   = "<<epr<<std::endl;
            std::cout<<std::endl;
#endif
            if(use_ILUC) this->preconditioner_exists &= this->Precond_left[this->number_levels].partialILUC(Akrow,Akrow_next,IP,true,this->Precond_right[this->number_levels],this->Precond_middle[this->number_levels],last_row_to_eliminate,tau,this->zero_pivots[this->number_levels],partial_setup_time,mem_factor,memory_allocated_factorization,memory_used_factorization);
            else this->preconditioner_exists &= this->Precond_left[this->number_levels].partialILUCDP(Akrow,Akcol,Akrow_next,IP,true,this->Precond_right[this->number_levels],this->Precond_middle[this->number_levels],pc2,pr2,ipc2,ipr2,last_row_to_eliminate,tau,bp,bpr,epr,this->zero_pivots[this->number_levels],partial_setup_time,mem_factor,memory_allocated_factorization,memory_used_factorization);
#ifdef INFO
            std::cout<<"     zero-pivots            = "<<this->zero_pivots[this->number_levels]<<std::endl;
            std::cout<<"     local fill-in          = "<<((Real)(this->Precond_left[this->number_levels].actual_non_zeroes()+this->Precond_right[this->number_levels].actual_non_zeroes())- (Real) Akrow.rows() )/((Real)Akrow.actual_non_zeroes())<<std::endl;
            if(use_ILUC) equality_permutations = Akrow.rows()-Akrow_next.rows();
            else equality_permutations = pc2.equality(pr2,0,Akrow.rows()-Akrow_next.rows());
            total_equality_permutations += equality_permutations;
            std::cout<<"     symmetry of permutations used in factorization          = "<<(Real) equality_permutations/(Real)(Akrow.rows()-Akrow_next.rows()) <<std::endl;
#endif
            if(! this->preconditioner_exists) return;
            if(use_ILUC){
                this->permutation_columns[this->number_levels] = pc1;
                this->permutation_rows[this->number_levels] = pr1;
            } else {
                this->permutation_columns[this->number_levels].compose(pc1,pc2);
                this->permutation_rows[this->number_levels].compose(pr1,pr2);
            }
            this->inverse_permutation_columns[this->number_levels].invert(this->permutation_columns[this->number_levels]);
            this->inverse_permutation_rows[this->number_levels].invert(this->permutation_rows[this->number_levels]);
            this->number_levels++;
            memory_matrices = Amemory + Akrow.memory() + Akcol.memory(); // Akrow_next is not needed, as this is included in the memory counted by the factorization.
            memory_currently_used = memory_matrices + memory_used_factorization + memory_previous_levels;
            memory_currently_allocated = memory_matrices + memory_allocated_factorization + memory_previous_levels;
            if(memory_currently_allocated > max_memory_allocated) max_memory_allocated = memory_currently_allocated;
            if(memory_currently_used > max_memory_used) max_memory_used = memory_currently_used;
            memory_previous_levels += this->memory(this->number_levels-1);
            //this->setup_time += partial_setup_time;
        }
    }
    this->left_form=LOWER_TRIANGULAR;
    this->right_form=UPPER_TRIANGULAR;
    this->pre_image_size=A.rows();
    this->image_size=A.rows();
    this->intermediate_size=A.rows();
    time_2 = clock();
    this->setup_time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
    this->memory_allocated_to_create = max_memory_allocated;
    this->memory_used_to_create = max_memory_used;
#ifdef INFO
    std::cout<<"     Overall symmetry of permutations used in factorization          = "<<(Real) total_equality_permutations/(Real) A.rows() <<std::endl;
#endif
}


template <class T, class matrix_type, class vector_type>
void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::make_preprocessed_multilevelILUCDP(T* Adata, Integer* Aindices, Integer* Apointer, Integer Adim, Integer Annz, orientation_type Aorient, const iluplusplus_precond_parameter& IP){
# ifdef DEBUG
    std::cerr<<"multilevelILUCDPPreconditioner::make_preprocessed_multilevelILUCDP: the coefficient matrix required as an argument of this function is being provided by pointers to arrays. These arrays are being interpreted as the matrix stored in CSR/CSC-format. No index check on accessing these arrays is possible. Cross your fingers...."<<std::endl<<std::flush;
# endif
    Integer Anumber_rows = Adim;
    Integer Anumber_columns = Adim;
    Integer nnz = Annz;
    orientation_type AO = Aorient;
    matrix_type Arow;
    Arow.interchange(Adata,Aindices,Apointer,Anumber_rows,Anumber_columns,nnz,AO);
    make_preprocessed_multilevelILUCDP(Arow,IP);
    Arow.interchange(Adata,Aindices,Apointer,Anumber_rows,Anumber_columns,nnz,AO);
}


template <class T, class matrix_type, class vector_type>
void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::make_preprocessed_multilevelILUCDP(std::vector<T>& Adata, std::vector<Integer>& Aindices, std::vector<Integer>& Apointer, orientation_type Aorient, const iluplusplus_precond_parameter& IP){
# ifdef DEBUG
    std::cerr<<"multilevelILUCDPPreconditioner::make_preprocessed_multilevelILUCDP: the coefficient matrix required as an argument of this function is being provided by C++ vectors, which are being interpreted as C-style arrays for a matrix stored in CSR/CSC-format. No index check on accessing these vectors is possible. Cross your fingers...."<<std::endl<<std::flush;
# endif
    Integer Anumber_rows = Apointer.size()-1;
    Integer Anumber_columns =  Apointer.size()-1;
    Integer nnz =  Apointer[Apointer.size()-1];
    orientation_type AO = Aorient;
    matrix_type Arow;
    T* ptr_Adata = &Adata[0];
    Integer* ptr_Aindices = &Aindices[0];
    Integer* ptr_Apointer = &Apointer[0];
    Arow.interchange(ptr_Adata,ptr_Aindices,ptr_Apointer,Anumber_rows,Anumber_columns,nnz,AO);
    make_preprocessed_multilevelILUCDP(Arow,IP);
    Arow.interchange(ptr_Adata,ptr_Aindices,ptr_Apointer,Anumber_rows,Anumber_columns,nnz,AO);
}

          // only for testing purposes.
template <class T, class matrix_type, class vector_type>
void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::make_single_level_of_preprocessed_multilevelILUCDP(const matrix_type &Arow, const iluplusplus_precond_parameter& IP, bool force_finish, matrix_type& Acoarse, Real threshold){
    this->memory_allocated_to_create = 0.0;
    this->memory_used_to_create=0.0;
    Real memory_allocated_factorization = 0.0;
    Real memory_used_factorization = 0.0;
    Real memory_matrices = 0.0;
    param = IP;
    clock_t time_1,time_2;
    time_1 = clock();
    if(Arow.orient()==ROW){
        matrix_type Akrow,Akcol,Akrow_next;
        index_list pr1, pr2, ipr1,ipr2,pc1,pc2,ipc1,ipc2;
        Real partial_setup_time;
        Real mem_factor=IP.get_MEM_FACTOR();
        this->setup_time = 0.0;
        Integer last_row_to_eliminate,bp,bpr,epr,end_PQ;
        Akrow = Arow;
        Integer matrix_size = Akrow.rows();
        this->preconditioner_exists = true;
        this->number_levels=0;
        end_PQ = Akrow.preprocess(IP,pr1,pc1,ipr1,ipc1,this->D_l[this->number_levels],this->D_r[this->number_levels]);
        if (IP.get_EXTERNAL_FINAL_ROW()) last_row_to_eliminate = end_PQ-1;
        else last_row_to_eliminate = (Akrow.rows()-1)/2;
        switch (IP.get_PERMUTE_ROWS()) {
            case 0:  bpr = 0; epr = 0; break;
            case 1:  if(IP.get_EXTERNAL_FINAL_ROW()){bpr = 0; epr = 0;} else {bpr = end_PQ; epr = Akrow.rows()-1;} break;
            case 2:  bpr = 0; epr = last_row_to_eliminate; break;
            case 3:  bpr = 0; epr = Akrow.rows()-1; break;
            default: std::cerr<<"choose permissible value for PERMUTE_ROWS!"<<std::endl; throw iluplusplus_error(OTHER_ERROR);
        }
        switch (IP.get_TOTAL_PIV()) {
            case 0:  bp = Akcol.rows(); break;
            case 1:  bp = last_row_to_eliminate+1; break;
            case 2:  bp = 0;  break;
            default: std::cerr<<"choose permissible value for TOTAL_PIV!"<<std::endl; throw iluplusplus_error(OTHER_ERROR);
        }
        Akcol.change_orientation_of_data(Akrow);
#ifdef INFO
        std::cout<<std::endl;
        std::cout<<"**** level: "<<this->number_levels<<" ****"<<std::endl;
        std::cout<<"  ** matrix statistics:"<<std::endl;
        std::cout<<"     n                      = "<<Akrow.rows()<<std::endl;
        std::cout<<"     nnz                    = "<<Akrow.actual_non_zeroes()<<std::endl;
        std::cout<<"     density                = "<<Akrow.row_density()<<std::endl;
        std::cout<<"  ** preconditioner parameters:"<<std::endl;
        std::cout<<"     max. numb. nnz/row p   = "<<IP.get_fill_in()<<std::endl;
        std::cout<<"     tau                    = "<<threshold<<std::endl;
        std::cout<<"     perm tolerance         = "<<IP.get_perm_tol()<<std::endl;
        std::cout<<"     begin permuting rows   = "<<bpr<<std::endl;
        std::cout<<"     end   permuting rows   = "<<epr<<std::endl;
        if(IP.get_EXTERNAL_FINAL_ROW())
            std::cout<<"     last row to eliminate  = "<<last_row_to_eliminate;
        else
            std::cout<<"     last row to eliminate decided by preconditioner."<<std::endl;
        std::cout<<std::endl;
#endif
        this->preconditioner_exists &= this->Precond_left[this->number_levels].partialILUCDP(Akrow,Akcol,Acoarse,IP,force_finish,this->Precond_right[this->number_levels],this->Precond_middle[this->number_levels],pc2,pr2,ipc2,ipr2,last_row_to_eliminate,threshold,bp,bpr,epr,this->zero_pivots[this->number_levels],partial_setup_time,mem_factor,memory_allocated_factorization,memory_used_factorization);
        if(! this->preconditioner_exists) return;
#ifdef INFO
        std::cout<<"     zero-pivots            = "<<this->zero_pivots[this->number_levels]<<std::endl;
        std::cout<<"     local fill-in          = "<<((Real)(this->Precond_left[this->number_levels].actual_non_zeroes()+this->Precond_right[this->number_levels].actual_non_zeroes())- (Real) Akcol.rows() )/((Real)Akcol.actual_non_zeroes())<<std::endl;
#endif
        this->permutation_columns[this->number_levels].compose(pc1,pc2);
        this->permutation_rows[this->number_levels].compose(pr1,pr2);
        this->inverse_permutation_columns[this->number_levels].invert(this->permutation_columns[this->number_levels]);
        this->inverse_permutation_rows[this->number_levels].invert(this->permutation_rows[this->number_levels]);
        this->number_levels++;
        Akrow.interchange(Akrow_next);
        this->setup_time += partial_setup_time;
        matrix_size = Akrow.rows();
        threshold += IP.get_VARY_THRESHOLD_FACTOR();
        switch(IP.get_VARIABLE_MEM()){
            case 0: mem_factor = IP.get_MEM_FACTOR(); break;
            case 1: mem_factor = IP.get_MEM_FACTOR()*((Real)Arow.rows())/((Real)matrix_size); break;
            case 2: mem_factor = IP.get_MEM_FACTOR()*((Real)Arow.rows())/((Real)matrix_size)*((Real)Arow.rows())/((Real)matrix_size); break;
            default: std::cerr<<"Please use permissible value for VARIABLE_MEM."<<std::endl; throw iluplusplus_error(OTHER_ERROR);
        }
        this->left_form=LOWER_TRIANGULAR;
        this->right_form=UPPER_TRIANGULAR;
        memory_matrices = Arow.memory() + Akrow.memory() + Akcol.memory(); // Akrow_next is not needed, as this is included in the memory counted by the factorization.
    } else {
        std::cerr<<"multilevelILUCDPPreconditioner::make_single_level_of_preprocessed_multilevelILUCDP: Matrix needs to be provided in CSR format."<<std::endl;
    }
    this->pre_image_size=Arow.rows();
    this->image_size=Arow.rows();
    this->intermediate_size=Arow.rows();
    time_2 = clock();
    this->setup_time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
    if(IP.get_PREPROCESSING().dim()<2){  // in this case, only one copy of matrix is needed (total of 2 matrices) 
        this->memory_allocated_to_create = max(memory_matrices + memory_allocated_factorization, 2.0*Arow.memory());  // max of factorization and preprocessing
        this->memory_used_to_create = max(memory_matrices + memory_used_factorization, 2.0*Arow.memory());
    } else {
        this->memory_allocated_to_create = max(memory_matrices + memory_allocated_factorization, 3.0*Arow.memory());
        this->memory_used_to_create = max(memory_matrices + memory_used_factorization, 3.0*Arow.memory());
    }
}

template <class T, class matrix_type, class vector_type>
void multilevelILUCDPPreconditioner<T,matrix_type,vector_type>::make_single_level_of_preprocessed_multilevelILUCDP(const matrix_type &Arow, const iluplusplus_precond_parameter& IP, bool force_finish, matrix_type& Acoarse){
    make_single_level_of_preprocessed_multilevelILUCDP(Arow,IP,force_finish,Acoarse,IP.get_threshold());
}

} // end namespace iluplusplus

#endif
