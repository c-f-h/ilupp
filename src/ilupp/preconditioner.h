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


#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

// flags: INFO: more info on levels for Multilevel Preconditioner

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cmath>

#include<vector>


#include "declarations.h"
#include "sparse.h"

//***********************************************************************************************************************//
//                                                                                                                       //
//         The base class preconditioner: declaration                                                                    //
//                                                                                                                       //
//***********************************************************************************************************************//

namespace iluplusplus {

template <class T, class matrix_type, class vector_type> class preconditioner
  {
       protected:
          Real setup_time;
          Real memory_allocated_to_create;
          Real memory_used_to_create;
          Integer pre_image_size;
          Integer image_size;
          bool preconditioner_exists;
          virtual void apply_preconditioner_and_matrix(preconditioner_application1_type PA1, matrix_usage_type use, const matrix_type &A,const vector_type &v, vector_type &w) const = 0;
          virtual void apply_preconditioner_and_matrix_transposed(preconditioner_application1_type PA1, matrix_usage_type use, const matrix_type &A,const vector_type &v, vector_type &w) const = 0;
          virtual void apply_preconditioner_rhs(preconditioner_application1_type PA1, const matrix_type &A,const vector_type &b, vector_type &c) const = 0;
          virtual void apply_preconditioner_solution(preconditioner_application1_type PA1, const matrix_type &A,const vector_type &y, vector_type &x) const = 0;
          virtual void apply_preconditioner_starting_value(preconditioner_application1_type PA1, const matrix_type &A,const vector_type &x, vector_type &y) const = 0;
       public:
          preconditioner() : pre_image_size(0), image_size(0) {}

        // essentials
        // preconditioned multiplication with prescribed usage: from left, from right or split preconditioning used as simple, QTQ or QQT preconditioning
          virtual void preconditioned_matrix_vector_multiplication(preconditioner_application1_type PA1, const matrix_type &A, const vector_type &v, vector_type &w) const {
              apply_preconditioner_and_matrix(PA1,ID,A,v,w);
          }

        // preconditioned multiplication with prescribed usage: from left, from right or split preconditioning
          virtual void preconditioned_matrix_transposed_vector_multiplication(preconditioner_application1_type PA1, const matrix_type &A, const vector_type &v, vector_type &w) const {
              apply_preconditioner_and_matrix_transposed(PA1,ID,A,v,w);
          }

        // appropriate modification of rhs based on preconditioning technique chosen
          virtual void preconditioned_rhs(preconditioner_application1_type PA1, const matrix_type& A, const vector_type &b, vector_type &c) const {
              apply_preconditioner_rhs(PA1, A, b, c);
          }

        // appropriate adaption of solution based on preconditioning technique choses
          virtual void adapt_solution(preconditioner_application1_type PA1, const matrix_type& A, const vector_type &y, vector_type &x) const {
              apply_preconditioner_solution(PA1, A, y, x);
          }

        // appropriate starting value for iteration y based on initial guess of solution x (inverse of adapt_solution).
          virtual void preconditioned_starting_value(preconditioner_application1_type PA1, const matrix_type& A, const vector_type &x, vector_type &y) const {
              apply_preconditioner_starting_value(PA1, A, x, y);
          }

        // appropriate residual: r= L*(b-A*x), L being the left part of the preconditioner.
          virtual void preconditioned_residual(preconditioner_application1_type PA1, const matrix_type& A, const vector_type &b, const vector_type &x, vector_type &r) const;
          virtual void apply_preconditioner_only(matrix_usage_type use, const vector_type &x, vector_type &y) const = 0;
          virtual void apply_preconditioner_only(matrix_usage_type use, vector_type &y) const = 0;
          virtual void apply_preconditioner_only(matrix_usage_type use, T* data, Integer dim) const;
          virtual void apply_preconditioner_only(matrix_usage_type use, std::vector<T>& data) const;

          virtual bool compatibility_check(preconditioner_application1_type PA1, const matrix_type& A, const vector_type& b, const vector_type& x) const {
              return (compatibility_check(PA1,A,b) && (A.columns()!=x.dimension()));
          }
          virtual bool compatibility_check(preconditioner_application1_type PA1, const matrix_type& A, const vector_type& b) const = 0;

          Integer pre_image_dimension() const   { return pre_image_size; }
          Integer image_dimension() const       { return image_size; }

          bool exists() const                   { return preconditioner_exists; }
          Real memory_used_calculations() const { return memory_used_to_create; }
          Real memory_allocated_calculations() const { return memory_allocated_to_create; }
          Real time() const                     { return setup_time; }

          virtual Real memory() const           { return 0.0; }
          virtual std::string special_info() const;
        // info
          virtual Integer total_nnz() const = 0;
          virtual void print_info() const = 0;
          virtual void read_binary(std::string filename) = 0;
          virtual void write_binary(std::string filename) const = 0;
  };


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
  class split_preconditioner : public preconditioner  <T, matrix_type, vector_type>
  {
       protected:
          Integer intermediate_size;
          virtual void apply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const = 0;
          virtual void apply_preconditioner_left(matrix_usage_type use,  vector_type &w) const = 0;
          virtual void apply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const = 0;
          virtual void apply_preconditioner_right(matrix_usage_type use,  vector_type &w) const = 0;
          virtual void unapply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const = 0;
          virtual void unapply_preconditioner_left(matrix_usage_type use, vector_type &w) const = 0;
          virtual void unapply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const = 0;
          virtual void unapply_preconditioner_right(matrix_usage_type use, vector_type &w) const = 0;
          virtual void apply_preconditioner_and_matrix(preconditioner_application1_type PA1, matrix_usage_type use, const matrix_type &A,const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner_and_matrix_transposed(preconditioner_application1_type PA1, matrix_usage_type use, const matrix_type &A,const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner_rhs(preconditioner_application1_type PA1, const matrix_type &A,const vector_type &b, vector_type &c) const;
          virtual void apply_preconditioner_solution(preconditioner_application1_type PA1, const matrix_type &A,const vector_type &y, vector_type &x) const;
          virtual void apply_preconditioner_starting_value(preconditioner_application1_type PA1, const matrix_type &A,const vector_type &x, vector_type &y) const;
       public:
          virtual Integer left_nnz() const = 0;
          virtual Integer right_nnz() const = 0;
          virtual Integer total_nnz() const         { return left_nnz() + right_nnz(); }

          virtual bool compatibility_check(preconditioner_application1_type PA1, const matrix_type& A, const vector_type& b) const;
          using preconditioner<T,matrix_type,vector_type>::apply_preconditioner_only;
          virtual void apply_preconditioner_only(matrix_usage_type use, const vector_type &x, vector_type &y) const;
          virtual void apply_preconditioner_only(matrix_usage_type use,vector_type &y) const;
   };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: indirect_split_triangular_preconditioner                                                                      //
//                                                                                                                       //
//***********************************************************************************************************************//

template <class T, class matrix_type, class vector_type>
  class indirect_split_triangular_preconditioner : public split_preconditioner <T,matrix_type, vector_type>
  {
       protected:
          matrix_type Precond_left;     // the left preconditioning matrix
          matrix_type Precond_right;    // the right preconditioning matrix
          special_matrix_type left_form;
          special_matrix_type right_form;
          virtual void apply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner_left(matrix_usage_type use, vector_type &w) const;
          virtual void apply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const ;
          virtual void apply_preconditioner_right(matrix_usage_type use, vector_type &w) const;
          virtual void unapply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void unapply_preconditioner_left(matrix_usage_type use, vector_type &w) const;
          virtual void unapply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void unapply_preconditioner_right(matrix_usage_type use, vector_type &w) const;
       public:
          const matrix_type& left_matrix() const    { return Precond_left; }
          const matrix_type& right_matrix() const   { return Precond_right; }
          virtual Integer left_nnz() const          { return Precond_left.actual_non_zeroes(); }
          virtual Integer right_nnz() const         { return Precond_right.actual_non_zeroes(); }

          virtual void print_info() const;
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: indirect_split_triangular_multilevel_preconditioner                                                //
//                                                                                                                       //
//***********************************************************************************************************************//

template <class T, class matrix_type, class vector_type>
  class indirect_split_triangular_multilevel_preconditioner : public split_preconditioner <T,matrix_type, vector_type>
  {
       protected:
          array<matrix_type> Precond_left;     // the left preconditioning matrices
          array<matrix_type> Precond_right;    // the right preconditioning matrices
          array<vector_type> Precond_middle;
          special_matrix_type left_form;
          special_matrix_type right_form;
          Integer number_levels;
          array<Integer> begin_next_level;
          array<index_list> permutation_rows;
          array<index_list> permutation_columns;
          array<index_list> inverse_permutation_rows;
          array<index_list> inverse_permutation_columns;
          array<vector_type> D_l;  // scaling
          array<vector_type> D_r;  // scaling
          virtual void apply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner_left(matrix_usage_type use, vector_type &w) const;
          virtual void apply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner_right(matrix_usage_type use, vector_type &w) const;
          virtual void unapply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void unapply_preconditioner_left(matrix_usage_type use, vector_type &w) const;
          virtual void unapply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void unapply_preconditioner_right(matrix_usage_type use, vector_type &w) const;
       public:
          virtual void clear();  // resize everything to 0
          virtual void init(Integer memory_max_level);

          const matrix_type& extract_left_matrix(Integer k) const                  { return Precond_left[k]; }
          const matrix_type& extract_right_matrix(Integer k) const                 { return Precond_right[k]; }
          const vector_type& extract_middle_matrix(Integer k) const                { return Precond_middle[k]; }
          const index_list& extract_permutation_rows(Integer k) const              { return permutation_rows[k]; }
          const index_list& extract_permutation_columns(Integer k) const           { return permutation_columns[k]; }
          const index_list& extract_inverse_permutation_rows(Integer k) const      { return inverse_permutation_rows[k]; }
          const index_list& extract_inverse_permutation_columns(Integer k) const   { return inverse_permutation_columns[k]; }
          const vector_dense<T>& extract_left_scaling(Integer k) const             { return D_l[k]; }
          const vector_dense<T>& extract_right_scaling(Integer k) const            { return D_r[k]; }

          Integer levels() const                        { return number_levels; }
          virtual Integer left_nnz(Integer k) const     { return Precond_left[k].actual_non_zeroes(); }
          virtual Integer right_nnz(Integer k) const    { return Precond_right[k].actual_non_zeroes(); }
          virtual Integer total_nnz(Integer k) const    { return Precond_left[k].actual_non_zeroes()+Precond_right[k].actual_non_zeroes();}

          virtual Real memory() const;
          virtual Real memory(Integer k) const;
          virtual Integer dim(Integer k) const;
          virtual void print_dimensions() const;
          virtual void write_abs_diagonal(std::string filename) const;
          virtual void write_abs_diagonal_with_indices(std::string filename) const;
          virtual Integer number_small_pivots(Real tau) const;
          virtual Integer left_nnz() const;
          virtual Integer right_nnz() const;
          virtual Integer middle_nnz() const;
          virtual Integer total_nnz() const             { return left_nnz() + right_nnz() + middle_nnz(); }
          virtual matrix_type left_preconditioning_matrix(Integer k);
          virtual matrix_type right_preconditioning_matrix(Integer k);
          virtual void print(Integer k) const;
          virtual void print_info(Integer k) const;
          virtual void print_info() const;
          virtual void print() const;
  };



// The class: indirect_split_pseudo_triangular_preconditioner
// (if permutation is applied to the columns Precond_right, a upper-triangular matrix results)
template <class T, class matrix_type, class vector_type>
  class indirect_split_pseudo_triangular_preconditioner : public split_preconditioner <T,matrix_type, vector_type>
  {
       protected:
          matrix_type Precond_left;     // the left preconditioning matrix
          matrix_type Precond_right;    // the right preconditioning matrix
          special_matrix_type left_form;
          special_matrix_type right_form;
          perm_usage_type left_matrix_usage;
          perm_usage_type right_matrix_usage;
          index_list permutation;
          index_list permutation2;
          virtual void apply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner_left(matrix_usage_type use, vector_type &w) const;
          virtual void apply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner_right(matrix_usage_type use, vector_type &w) const;
          virtual void unapply_preconditioner_left(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void unapply_preconditioner_left(matrix_usage_type use,vector_type &w) const;
          virtual void unapply_preconditioner_right(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void unapply_preconditioner_right(matrix_usage_type use,vector_type &w) const;
       public:
          const matrix_type& left_matrix() const    { return Precond_left; }
          const matrix_type& right_matrix() const   { return Precond_right; }
          virtual Integer left_nnz() const          { return Precond_left.actual_non_zeroes(); }
          virtual Integer right_nnz() const         { return Precond_right.actual_non_zeroes(); }
          virtual const index_list& extract_permutation() const    { return permutation; }
          virtual const index_list& extract_permutation2() const   { return permutation2; }

          virtual void print_info() const;
          virtual void eliminate_permutations(matrix_type& A, vector_type &b);
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: NullPrecondioner: does not precondition.                                                           //
//                                                                                                                       //
//***********************************************************************************************************************//

template <class T, class matrix_type, class vector_type>
  class NullPreconditioner : public preconditioner <T,matrix_type, vector_type>
  {
       protected:
          virtual void apply_preconditioner(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void apply_preconditioner(matrix_usage_type use, vector_type &w) const;
          virtual void unapply_preconditioner(matrix_usage_type use, const vector_type &v, vector_type &w) const;
          virtual void unapply_preconditioner(matrix_usage_type use, vector_type &w) const;
       public:
          NullPreconditioner();
          NullPreconditioner(Integer m, Integer n);
          virtual void read_binary(std::string filename);
          virtual void write_binary(std::string filename) const;
          virtual void print_info() const;
          virtual Integer total_nnz() const     { return 0; }
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUC Preconditioner (Saad):                                                                         //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
  class ILUCPreconditioner : public indirect_split_triangular_preconditioner <T,matrix_type, vector_type>
  {
       public:
          ILUCPreconditioner(const matrix_type &A, Integer max_fill_in, Real threshold);  // default threshold=-1.0
          virtual void write_binary(std::string filename) const;
          virtual void read_binary(std::string filename);
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUT Preconditioner (Saad):                                                                         //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
  class ILUTPreconditioner : public indirect_split_triangular_preconditioner <T,matrix_type, vector_type>
  {
       public:
          ILUTPreconditioner(const matrix_type &A, Integer max_fill_in, Real threshold); // default threshold=1000
          virtual std::string special_info() const;
          virtual void write_binary(std::string filename) const;
          virtual void read_binary(std::string filename);
          virtual Integer left_nnz() const;
          virtual Integer right_nnz() const;
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUTP Preconditioner (Saad):                                                                       //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
  class ILUTPPreconditioner : public indirect_split_pseudo_triangular_preconditioner <T,matrix_type, vector_type>
  {
       private:
          Integer zero_pivots;
       public:
          ILUTPPreconditioner(const matrix_type &A, Integer max_fill_in, Real threshold, Real perm_tol, Integer row_pos, Real mem_factor); // default threshold=-1.0, pt = 0.0,  rp=0
          virtual Integer zero_pivots_encountered()         { return zero_pivots; }
          virtual std::string special_info() const;
          virtual void write_binary(std::string filename) const;
          virtual void read_binary(std::string filename);
          virtual Integer left_nnz() const                  { return this->Precond_left.actual_non_zeroes() - this->image_size; }
          virtual Integer right_nnz() const                 { return this->Precond_right.actual_non_zeroes(); }
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUCP Preconditioner:                                                                              //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
  class ILUCPPreconditioner : public indirect_split_pseudo_triangular_preconditioner <T,matrix_type, vector_type>
  {
       private:
          Integer zero_pivots;
       public:
          ILUCPPreconditioner(const matrix_type &Acol, Integer max_fill_in, Real threshold=-1.0, Real perm_tol=0.0, Integer rp=-1, Real mem_factor=10.0);
          ILUCPPreconditioner(const matrix_type &Acol, const ILUCP_precond_parameter& p);
          virtual Integer zero_pivots_encountered()         { return zero_pivots; }
          virtual std::string special_info() const;
          virtual void write_binary(std::string filename) const;
          virtual void read_binary(std::string filename);
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUCDP Preconditioner:                                                                             //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
  class ILUCDPPreconditioner : public indirect_split_pseudo_triangular_preconditioner <T,matrix_type, vector_type>
  {
       private:
          Integer zero_pivots;
       public:
          ILUCDPPreconditioner(const matrix_type &Arow, const matrix_type &Acol, Integer max_fill_in, Real threshold, Real perm_tol, Integer bpr); // default: threshold=1000.0, perm_tol=1000.0, bpr = -1
          ILUCDPPreconditioner(const matrix_type &Arow, const matrix_type &Acol, matrix_type &Anew, Integer max_fill_in, Real threshold, Real perm_tol, Integer); // default: threshold=1000.0, perm_tol=1000.0, bpr = -1
          ILUCDPPreconditioner(const matrix_type &Arow, const matrix_type &Acol, const ILUCDP_precond_parameter& p);
          virtual Integer zero_pivots_encountered();
          virtual std::string special_info() const;
          virtual void write_binary(std::string filename) const;
          virtual void read_binary(std::string filename);
  };


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: multilevelILUCDP Preconditioner:                                                                   //
//                                                                                                                       //
//***********************************************************************************************************************//


template <class T, class matrix_type, class vector_type>
  class multilevelILUCDPPreconditioner : public indirect_split_triangular_multilevel_preconditioner <T,matrix_type, vector_type>
  {
       private:
          std::vector<Integer> zero_pivots;
          iluplusplus_precond_parameter param;
          Integer dim_zero_matrix_factored;
       public:
          multilevelILUCDPPreconditioner();
          virtual void init(Integer mem_levels);
          virtual iluplusplus_precond_parameter extract_parameters() const;
          virtual Integer dimension_zero_matrix_factored() const;
          virtual Integer zero_pivots_encountered(Integer k) const;
          virtual Integer zero_pivots_encountered() const;
          virtual std::string special_info() const;
          virtual void write_binary(std::string filename) const;
          virtual void read_binary(std::string filename);
          virtual void make_preprocessed_multilevelILUCDP(const matrix_type &A, const iluplusplus_precond_parameter& IP);
          virtual void make_preprocessed_multilevelILUCDP(T* Adata, Integer* Aindices, Integer* Apointer, Integer Adim, Integer Annz, orientation_type Aorient, const iluplusplus_precond_parameter& IP);
          virtual void make_preprocessed_multilevelILUCDP(std::vector<T>& Adata, std::vector<Integer>& Aindices, std::vector<Integer>& Apointer, orientation_type Aorient, const iluplusplus_precond_parameter& IP);
          // only for testing purposes.
          virtual void make_single_level_of_preprocessed_multilevelILUCDP(const matrix_type &Arow, const iluplusplus_precond_parameter& IP, bool force_finish, matrix_type& Acoarse, Real threshold);
          virtual void make_single_level_of_preprocessed_multilevelILUCDP(const matrix_type &Arow, const iluplusplus_precond_parameter& IP, bool force_finish, matrix_type& Acoarse); 
  };

} // end namespace iluplusplus

#endif
