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


#ifndef ILUPLUSPLUS_INTERFACE_H
#define ILUPLUSPLUS_INTERFACE_H

#include "iluplusplus_declarations.h"  // will include all declarations

namespace iluplusplus {

class vector;
class matrix;
class multilevel_preconditioner;

class vector : public vector_dense<Coeff_Field>
{
    public:
        vector();
        vector(Integer m);
        vector(Integer m, Coeff_Field d);
        ~vector();
        vector(const vector& x);
        vector& operator =(const vector& x);
        Coeff_Field  get(Integer j) const;
        Coeff_Field  read(Integer j) const;  // read_data
        Coeff_Field& set(Integer j);
        Integer dim() const;
        void resize(Integer newsize, Coeff_Field d);
        void resize_without_initialization(Integer newsize);
        void set_all(Coeff_Field t);
        void interchange(Coeff_Field*& vdata, Integer& dim);
};

class matrix : public matrix_sparse<Coeff_Field>
{
    public:
        matrix();
        ~matrix();
        matrix(const matrix& x);
        matrix& operator = (const matrix& x);
        void interchange(matrix& x);
        vector operator * (const vector& x);
        void multiply(vector &v) const;
        void multiply(const vector& v, vector &w) const;
        void interchange(Coeff_Field*& Adata, Integer*& Aindices, Integer*& Apointer, Integer& Anumber_rows, Integer& Anumber_columns, Integer& Annz, orientation_type& Aorientation);
        Integer rows() const;
        Integer columns() const;
        Integer non_zeroes() const;        // returns the number of non-zeroes of *this. i.e. the reserved memory.
        Integer actual_non_zeroes() const; // returns the actual number of non-zeroes, i.e. the used memory.
        void print_info() const;
        void print_all() const;
};


class multilevel_preconditioner : public multilevelILUCDPPreconditioner<Coeff_Field,matrix_sparse<Coeff_Field>,vector_dense<Coeff_Field> >
{
    public:
        multilevel_preconditioner();
        ~multilevel_preconditioner();
        multilevel_preconditioner(const multilevel_preconditioner& x);
        multilevel_preconditioner& operator =(const multilevel_preconditioner& x);
        void setup(const matrix &A, const iluplusplus_precond_parameter& IP);
        void setup(Coeff_Field* Adata, Integer* Aindices, Integer* Apointer, Integer dim, Integer Annz, orientation_type Aorient, const iluplusplus_precond_parameter& IP);
        void setup(std::vector<Coeff_Field>& Adata, std::vector<Integer>& Aindices, std::vector<Integer>& Apointer, orientation_type Aorient, const iluplusplus_precond_parameter& IP);
        void apply_preconditioner(const vector &x, vector &y) const;
        void apply_preconditioner(vector &y) const;
        void apply_preconditioner(std::vector<Coeff_Field> &y) const;
        void apply_preconditioner(Coeff_Field* data, Integer dim) const;
        Real memory_used_calculations() const;
        Real memory_allocated_calculations() const;
        Real memory() const;
        bool exists() const;
        std::string special_info() const;
        Integer total_nnz() const;
        void print_info() const;
        Integer dim() const;
};


std::ostream& operator << (std::ostream& os, const vector& x);
std::ostream& operator << (std::ostream& os, const matrix& x);


bool BiCGstab(const multilevel_preconditioner& P, const matrix& A, const vector& b, vector& x, Integer min_iter,
           Integer& max_iter_iterations_used,Real& eps_rel_residual, Real& abs_residual);

bool solve_with_multilevel_preconditioner(const matrix& A, const vector& b, const vector& x_exact, vector& x, bool exact_solution_known,
          Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error,
          std::string directory, std::string matrix_name, const iluplusplus_precond_parameter &IP, bool detailed_output = false, std::string directory_data = "");

bool solve_with_multilevel_preconditioner(Integer n, Integer nnz, orientation_type O, Coeff_Field*& data, Integer*& indices, Integer*& pointer,
         Coeff_Field*& b, Integer& n_x_exact, Coeff_Field*& x_exact,  Integer& n_x, Coeff_Field*& x, bool exact_solution_known,
          Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error,
          std::string directory, std::string matrix_name, const iluplusplus_precond_parameter &IP, bool detailed_output = false, std::string directory_data = "");

bool solve_with_multilevel_preconditioner(orientation_type O, const std::vector<Coeff_Field>& data_vec, const std::vector<Integer>& indices_vec, const std::vector<Integer>& pointer_vec,
          const std::vector<Coeff_Field>& b_vec, const std::vector<Coeff_Field>& x_exact_vec, std::vector<Coeff_Field>& x_vec, bool exact_solution_known,
          Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error,
          std::string directory, std::string matrix_name, const iluplusplus_precond_parameter &IP, bool detailed_output = false, std::string directory_data = "");
}  // end namespace iluplusplus

#endif
