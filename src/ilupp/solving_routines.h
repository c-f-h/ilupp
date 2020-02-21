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


#ifndef SOLVING_ROUTINES_H
#define SOLVING_ROUTINES_H


namespace iluplusplus {

template <class T, class matrix_type, class vector_type>
    bool solve_linear_system(preconditioner_type t, const preconditioner<T,matrix_type,vector_type>& P, 
          const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
          Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error,
          Real& time, std::string directory, std::string filename, std::string matrix_name, std::string precond_name);

// ***************************************** //

template <class T, class matrix_type, class vector_type>
    bool solve_without_preconditioner(const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
                  Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error,
                  Real& time, std::string directory, std::string matrix_name);

template <class T, class matrix_type, class vector_type>
    bool test_without_preconditioner(const matrix_type& A, const vector_type& b,Real eps, Integer max_iter, Real& time, std::string directory, std::string matrix_name);


template <class T, class matrix_type, class vector_type>
    bool test_without_preconditioner(const matrix_type& A, Real eps, Integer max_iter,Real& time, std::string directory, std::string matrix_name);


template <class T, class matrix_type, class vector_type>
    bool test_without_preconditioner_with_random_matrix(Integer dim, Integer min_nnz, Integer max_nnz, Real eps, Integer max_iter, Real& time, std::string directory);


// ***************************************** //


template <class T, class matrix_type, class vector_type>
    bool solve_with_multilevel_preconditioner(const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
          Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error,
          Real& time, std::string directory, std::string matrix_name, const iluplusplus_precond_parameter &IP);

template <class T, class matrix_type, class vector_type>
    bool solve_with_multilevel_preconditioner_with_detailed_output(const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
                  Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error,
                  Real& time, std::string directory, std::string matrix_name, std::string output_directory, const iluplusplus_precond_parameter &IP);

template <class T, class matrix_type, class vector_type>
    void test_multilevel_preconditioner(iluplusplus_precond_parameter &IP, const matrix_type& A,
        Real begin_threshold, Real end_threshold, Integer testnumber, Real eps, Integer max_iter,
        Real& time, std::string directory, std::string matrix_name, bool write_detailed_output = false, std::string output_directory = "");

template <class T, class matrix_type, class vector_type>
    void test_multilevel_preconditioner(iluplusplus_precond_parameter &IP, const matrix_type& A, const vector_type& b,
        Real begin_threshold, Real end_threshold, Integer testnumber, Real eps, Integer max_iter,
        Real& time, std::string directory, std::string matrix_name, bool write_detailed_output = false, std::string output_directory = "");

template <class T, class matrix_type, class vector_type>
    void test_multilevel_preconditioner_with_random_matrix(iluplusplus_precond_parameter &IP,
        Integer dim, Integer min_nnz, Integer max_nnz,
        Real begin_threshold, Real end_threshold, Integer testnumber, Real eps, Integer max_iter,
        Real& time, std::string directory, bool write_detailed_output = false, std::string output_directory = "");


} // end namespace iluplusplus



#endif
