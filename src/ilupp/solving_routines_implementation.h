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



#ifndef SOLVING_ROUTINES_IMPLEMENTATION_H
#define SOLVING_ROUTINES_IMPLEMENTATION_H

#include "preconditioner.h"
#include "iterative_solvers.h"
#include "solving_routines.h"


#include "preconditioner_implementation.h"
#include "iterative_solvers_implementation.h"

namespace iluplusplus {

// NORPRECOND   = no preconditioning
// ILUC         = incomplete LU in Crout's formulation
// ILUT         = incomplete LU with threshold but without pivoting
// ILUTP        = incomplete LU with threshold and pivoting
// ILUCP        = incomplete LU in Crout's formulation with pivoting
// ILUCDP       = incomplete LU in Crout's formulation with dual pivoting
// ML_ILUCDP    = multilevel incomplete LU in Crout's formulation with dual pivoting
// DLML_ILUCDP  = dual level multilevel incomplete LU in Crout's formulation with dual pivoting
// MG_ILUCDP    = multigrid  incomplete LU in Crout's formulation with dual pivoting




template <class T, class matrix_type, class vector_type>
    bool solve_linear_system(preconditioner_type t, const preconditioner<T,matrix_type,vector_type>& P,
          const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
          Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error, 
          Real& time, std::string directory, std::string filename, std::string matrix_name, std::string precond_name)
{
    try {
          const Integer MIN_ITER = 1;
          //Integer restart=60;
          bool dimensions_compatible = ((P.image_dimension()==P.pre_image_dimension()) && (A.rows()== b.dimension()) && (A.rows() == A.columns()) && (A.rows()== P.image_dimension()) );
          clock_t time_begin, time_end;
          time = 0.0;
          Real matrix_memory = A.memory();
          Real fill_in = (Real)P.total_nnz()/(Real)A.actual_non_zeroes();
          Real mem_factor_storage = P.memory()/matrix_memory;
          Real mem_factor_calc_used = P.memory_used_calculations()/matrix_memory;
          Real mem_factor_calc_alloc = P.memory_allocated_calculations()/matrix_memory;
          bool use_split = (t==ILUC || t==ILUT || t==ILUTP  || t==ILUCP || t == ILUCDP || t==ML_ILUCDP || t == DLML_ILUCDP);
          bool no_iteration_possible = !dimensions_compatible || !P.exists();
          bool iteration_successful = true;
          if(no_iteration_possible){
              std::cout<<"solve_linear_system: dimensions are incompatible or preconditioner does not exist. Returning zero vector."<<std::endl;
              x.resize(A.columns(),0.0);
              abs_error = log10(-1.0);
              eps_rel_residual = log10(-1.0);
              abs_residual = log10(-1.0);
              max_iter_iterations_used = 0;
          } else {
              // iteration.
              if(use_split){
                  time_begin = clock();
                  //iteration_successful = bicgstab<T,matrix_type,vector_type>(P,LEFT,A,b,x,MIN_ITER,max_iter_iterations_used,eps_rel_residual,abs_residual,true);
                  iteration_successful = bicgstab<T,matrix_type,vector_type>(P,SPLIT,A,b,x,MIN_ITER,max_iter_iterations_used,eps_rel_residual,abs_residual,true);
                  // note: gmres experimentally did not work as well as bicgstab
                  //iteration_successful = gmres<T,matrix_type,vector_type>(P,SPLIT,A,b,x,restart,MIN_ITER,max_iter_iterations_used,eps_rel_residual,abs_residual,true);
                  time_end = clock();
              } else {
                  time_begin = clock();
                  iteration_successful = bicgstab<T,matrix_type,vector_type>(P,LEFT,A,b,x,MIN_ITER,max_iter_iterations_used,eps_rel_residual,abs_residual,true);
                  // note: gmres experimentally did not work as well as bicgstab
                  //iteration_successful = gmres<T,matrix_type,vector_type>(P,LEFT,A,b,x,restart,MIN_ITER,max_iter_iterations_used,eps_rel_residual,abs_residual,true);
                  time_end = clock();
              }
              if(exact_solution_known)abs_error = -log10((x-x_exact).norm_max());
              else abs_error = std::log(-1.0); // will be nan
              time=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
              //fill_in=(Real)P.total_nnz()/(Real)A.actual_non_zeroes();
          }
          if (!matrix_name.empty()) {
              // only write to output file if a matrix name was given
              std::ofstream outfile((directory+filename).c_str(), std::ios_base::app);
              if(outfile) {
                  outfile.setf(std::ios_base::left);
                  outfile<<std::setw(12)<<matrix_name<<" | ";
                  outfile<<std::setw(40)<<(precond_name+P.special_info())<<" | ";
                  outfile.setf(std::ios::right|std::ios::fixed);
                  if(no_iteration_possible){
                      outfile<<"                                                ***        Preconditioner does not exist.       ***                |"<<std::endl;
                      return false;
                  }
                  outfile<<std::setw(6)<<std::setprecision(2)<<fill_in<<" | ";
                  outfile.setf(std::ios_base::right);
                  outfile<<std::setw(6)<<std::setprecision(2)<<mem_factor_storage<<" | ";
                  outfile.setf(std::ios_base::right);
                  outfile<<std::setw(6)<<std::setprecision(2)<<mem_factor_calc_used<<" | ";
                  outfile.setf(std::ios_base::right);
                  outfile<<std::setw(6)<<std::setprecision(2)<<mem_factor_calc_alloc<<" || ";
                  outfile.setf(std::ios_base::right);
                  outfile<<std::setw(3)<<max_iter_iterations_used<<" | ";
                  outfile.setf(std::ios::right|std::ios::fixed);
                  if(exact_solution_known)
                      outfile<<std::setw(6)<<std::setprecision(2)<<abs_error<<" | ";
                  else
                      outfile<<"unknown| ";
                  outfile.setf(std::ios::right|std::ios::fixed);
                  outfile<<std::setw(6)<<std::setprecision(2)<<eps_rel_residual<<" | ";
                  outfile.setf(std::ios::right|std::ios::fixed);
                  outfile<<std::setw(6)<<std::setprecision(2)<<abs_residual<<" | ";
                  outfile.setf(std::ios_base::right);
                  outfile<<std::setw(9)<<P.total_nnz()<<" | ";
                  outfile.setf(std::ios::right|std::ios::fixed);
                  outfile<<std::setw(6)<<std::setprecision(2)<<P.time()<<" | ";
                  outfile.setf(std::ios::right|std::ios::fixed);
                  outfile<<std::setw(6)<<std::setprecision(2)<<time<<" | ";
                  outfile.setf(std::ios::right|std::ios::fixed);
                  outfile<<std::setw(6)<<std::setprecision(2)<<P.time()+time<<" |";

                  if(iteration_successful)
                      outfile<<"success|";
                  else
                      outfile<<"failure|";
                  outfile<<std::endl;
                  outfile.close();
              }
          }

          return iteration_successful;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"solve_linear_system: "<<ippe.error_message()<<". Returning."<<std::endl;
        return false;
    }
    catch(...){
        std::cerr<<"solve_linear_system: Unknown Error. Returning."<<std::endl;
        return false;
    }
}


// ************************************************************************************************* //
// ************************************************************************************************* //
//                  Testing without preconditioner                                            //
// ************************************************************************************************* //
// ************************************************************************************************* //


template <class T, class matrix_type, class vector_type>
    bool solve_without_preconditioner(const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
                  Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error, 
                  Real& time, std::string directory, std::string matrix_name)
{
    try {
          bool success;
          std::string precond_name = "NP";
          std::string filename = "NP.out";
          NullPreconditioner<T,matrix_type, vector_type> P(A.columns(),A.rows());
          success = solve_linear_system<T,matrix_type,vector_type>(NOPRECOND,P,A,b,x_exact,x,exact_solution_known,eps_rel_residual, abs_residual, max_iter_iterations_used, abs_error,time,directory,filename,matrix_name,precond_name);
          return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"solve: "<<ippe.error_message()<<". Returning."<<std::endl;
        return false;
    }
}

// tests with known right hand side b
template <class T, class matrix_type, class vector_type>
    bool test_without_preconditioner(const matrix_type& A, const vector_type& b,Real eps, Integer max_iter, Real& time, std::string directory, std::string matrix_name)
{
    try {
        bool success;
        if(non_fatal_error(!A.square_check() || A.rows() != b.dimension(),"test_without_preconditioner: Argument needs to be square and matrix needs same dimension as right hand side.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        vector_type x, x_exact;  // x_exact is dummy declaration
        Real eps_rel_residual, abs_residual, abs_error;
        Integer max_iter_iterations_used;
        A.print_info();
        std::cout<<"beginning with "<<matrix_name<<" wihout preconditioner."<<std::endl;
        eps_rel_residual = eps;
        abs_residual = eps;
        max_iter_iterations_used = max_iter;
        abs_error = -100.0;  // dummy
        success = solve_without_preconditioner<T,matrix_type,vector_type>(A,b,x_exact,x,false,eps_rel_residual,abs_residual, max_iter_iterations_used, abs_error,time,directory,matrix_name);
        std::cout<<std::endl;
        return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"test_without_preconditioner: "<<ippe.error_message()<<". Returning."<<std::endl;
        return false;
    }
}


// tests without preconditioner with artificial right hand side
template <class T, class matrix_type, class vector_type>
    bool test_without_preconditioner(const matrix_type& A, Real eps, Integer max_iter,Real& time, std::string directory, std::string matrix_name)
{
    try {
        bool success;
        if(non_fatal_error(!A.square_check(),"test_without_preconditioner: Argument needs to be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        vector_type x_exact,b,x;
        Real eps_rel_residual, abs_residual, abs_error;
        Integer max_iter_iterations_used;
        x_exact.resize(A.rows(),1.0);
        //x_exact.resize_set_natural_numbers(A.rows());
        A.matrix_vector_multiplication(ID,x_exact,b);
        A.print_info();
        std::cout<<"beginning with "<<matrix_name<<" wihout preconditioner."<<std::endl;
        eps_rel_residual = eps;
        abs_residual = eps;
        max_iter_iterations_used = max_iter;
        abs_error = -100.0;  // dummy
        success = solve_without_preconditioner<T,matrix_type,vector_type>(A,b,x_exact,x,true,eps_rel_residual,abs_residual, max_iter_iterations_used, abs_error,time,directory,matrix_name);
        std::cout<<std::endl;
        return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"test__without_preconditioner: "<<ippe.error_message()<<". Returning."<<std::endl;
        return false;
    }
}


template <class T, class matrix_type, class vector_type>
    bool test_without_preconditioner_with_random_matrix(Integer dim, Integer min_nnz, Integer max_nnz, Real eps, Integer max_iter, Real& time, std::string directory)
{
    try {
        bool success;
        matrix_type A;
        std::string matrix_name = ("random-"+integertostring(dim)+"-"+integertostring(min_nnz)+"-"+integertostring(max_nnz));
        A.random(dim,dim,ROW,min_nnz,max_nnz);
        success = test_without_preconditioner<T,matrix_type,vector_type>(A,eps,max_iter,time,directory,matrix_name);
        return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"test_without_preconditioner_with_random_matrix: "<<ippe.error_message()<<". Returning."<<std::endl;
        return false;
    }
}


// ************************************************************************************************* //
// ************************************************************************************************* //
//                  Testing for multilevel preconditioner                                            //
// ************************************************************************************************* //
// ************************************************************************************************* //

template <class T, class matrix_type, class vector_type>
    bool solve_with_multilevel_preconditioner(const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
                  Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error, 
                  Real& time, std::string directory, std::string matrix_name, const iluplusplus_precond_parameter &IP)
{
    try {
          bool success;
          std::string precond_name = IP.precondname();
          std::string filename = IP.filename(); 
          multilevelILUCDPPreconditioner<T,matrix_type, vector_type> P;
          if(IP.get_PRECON_PARAMETER()>= 0){ 
              P.make_preprocessed_multilevelILUCDP(A,IP);
              success = solve_linear_system<T,matrix_type,vector_type>(ML_ILUCDP,P,A,b,x_exact,x,exact_solution_known,eps_rel_residual, abs_residual, max_iter_iterations_used, abs_error,time,directory,filename,matrix_name,precond_name);
          } else {
              std::cout<<"solve_with_multilevel_preconditioner: PRECON_PARAMETER is negative. This is not permitted. Returning without solving."<<std::endl;
              return false;
          }
          return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"solve_with_multilevel_preconditioner: "<<ippe.error_message()<<". Returning."<<std::endl;
        return false;
    }
}

template <class T, class matrix_type, class vector_type>
    bool solve_with_multilevel_preconditioner_with_detailed_output(const matrix_type& A, const vector_type& b, const vector_type& x_exact, vector_type& x, bool exact_solution_known,
                  Real& eps_rel_residual, Real& abs_residual, Integer& max_iter_iterations_used, Real& abs_error, 
                  Real& time, std::string directory, std::string matrix_name, std::string output_directory, const iluplusplus_precond_parameter &IP)
{
    try {
          std::string precond_name = IP.precondname();
          std::string filename = IP.filename(); 
          std::string filenname_with_matrix = IP.filename(matrix_name);
          bool successful_solve;
          multilevelILUCDPPreconditioner<T,matrix_type, vector_type> P;
          if(IP.get_PRECON_PARAMETER()>= 0){ 
              P.make_preprocessed_multilevelILUCDP(A,IP);
              successful_solve = solve_linear_system<T,matrix_type,vector_type>(ML_ILUCDP,P,A,b,x_exact,x,exact_solution_known,eps_rel_residual, abs_residual, max_iter_iterations_used, abs_error,time,directory,filename,matrix_name,precond_name);
          } else {
              std::cout<<"solve_with_multilevel_preconditioner_with_detailed_output: PRECON_PARAMETER is negative. This is not permitted. Returning without solving."<<std::endl;
              return false;
          }
          // write data to files:
          std::ofstream outfile_data;
          std::string filename_data;
          filename_data = IP.filename(matrix_name);
          Real matrix_memory = A.memory();
          outfile_data.open((output_directory+filename_data).c_str(), std::ios_base::app);
          if(!outfile_data){
              std::cerr<<"solve_with_multilevel_preconditioner_with_detailed_output: error opening file to write. Returning."<<std::endl;
              return successful_solve;
          }
          outfile_data.setf(std::ios_base::left);
          if(successful_solve && P.exists()) outfile_data<<"1\t";
          else outfile_data<<"0\t";
          outfile_data<<std::setprecision(4)<<IP.get_threshold()<<"\t"<<std::setprecision(4)<<(Real)P.total_nnz()/(Real)A.actual_non_zeroes();
          outfile_data<<std::setprecision(4)<<"\t"<<P.memory()/matrix_memory<<"\t"<<std::setprecision(4)<<P.memory_used_calculations()/matrix_memory;
          outfile_data<<std::setprecision(4)<<"\t"<<std::setprecision(4)<<P.memory_allocated_calculations()/matrix_memory;
          outfile_data<<std::setprecision(4)<<"\t"<<std::setprecision(4)<<max_iter_iterations_used;
          if(exact_solution_known) outfile_data<<"\t"<<std::setprecision(4)<<abs_error;
          else outfile_data<<"\t"<<"NaN";
          outfile_data<<std::setprecision(4)<<"\t"<<eps_rel_residual<<"\t"<<std::setprecision(4)<<abs_residual;
          outfile_data<<"\t"<<std::setprecision(4)<<P.time();
          outfile_data<<std::setprecision(4)<<"\t"<<time<<"\t"<<std::setprecision(4)<<P.time()+time;
          outfile_data<<std::setprecision(4)<<"\t"<<A.dimension()<<"\t"<<std::setprecision(4)<<A.non_zeroes()<<"\t"<<std::setprecision(4)<<P.levels();
          outfile_data<<std::endl;
          outfile_data.close();
          return successful_solve;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"solve_with_multilevel_preconditioner_with_detailed_output: "<<ippe.error_message()<<". Returning."<<std::endl;
        return false;
    }
    catch(...){
        std::cerr<<"solve_with_multilevel_preconditioner_with_detailed_output: Unknown Error. Returning."<<std::endl;
        return false;
    }
}



// tests preconditioner with known right hand side b
template <class T, class matrix_type, class vector_type>
    void test_multilevel_preconditioner(iluplusplus_precond_parameter &IP, const matrix_type& A, const vector_type& b,
                Real begin_threshold, Real end_threshold, Integer testnumber, Real eps, Integer max_iter,
                Real& time, std::string directory, std::string matrix_name, bool write_detailed_output, std::string output_directory)
{
    try {
        if(non_fatal_error(!A.square_check() || A.rows() != b.dimension(),"test_multilevel_preconditioner: Argument needs to be square and matrix needs same dimension as right hand side.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        vector_type x, x_exact;  // x_exact is dummy declaration
        Real eps_rel_residual, abs_residual, abs_error;
        Integer max_iter_iterations_used;
        linear_function<Real> drop;
        drop.set(0.0, begin_threshold, (Real)(testnumber-1),end_threshold);
        std::cout<<"*************************************************************************"<<std::endl;
        A.print_info();
        std::cout<<"beginning with "<<matrix_name<<" using preconditioner number "<<IP.get_PRECON_PARAMETER()<<" with preprocessing "<<IP.get_PREPROCESSING().string()<<std::endl;
        std::cout<<"    ";
        for(Integer i=0;i<testnumber;i++){
            IP.set_threshold(drop(i));
            eps_rel_residual = eps;
            abs_residual = eps;
            max_iter_iterations_used = max_iter;
            abs_error = -100.0;  // dummy
            if (write_detailed_output)
                solve_with_multilevel_preconditioner_with_detailed_output<T,matrix_type,vector_type>(A,b,x_exact,x,false,eps_rel_residual,abs_residual, max_iter_iterations_used, abs_error,time,directory,matrix_name,output_directory,IP);
            else
                solve_with_multilevel_preconditioner<T,matrix_type,vector_type>(A,b,x_exact,x,false,eps_rel_residual,abs_residual, max_iter_iterations_used, abs_error,time,directory,matrix_name,IP);
            std::cout<<i<<" "<<std::flush;
        }
        std::cout<<std::endl;
        std::cout<<"*************************************************************************"<<std::endl;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"test_multilevel_preconditioner: "<<ippe.error_message()<<". Returning."<<std::endl;
        return;
    }
}


// tests preconditioner with artificial right hand side
template <class T, class matrix_type, class vector_type>
    void test_multilevel_preconditioner(iluplusplus_precond_parameter &IP, const matrix_type& A,
                Real begin_threshold, Real end_threshold, Integer testnumber, Real eps, Integer max_iter,
                Real& time, std::string directory, std::string matrix_name, bool write_detailed_output, std::string output_directory)
{
    try {
        if(non_fatal_error(!A.square_check(),"test_multilevel_preconditioner: Argument needs to be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        vector_type x_exact,b,x;
        Real eps_rel_residual, abs_residual, abs_error;
        Integer max_iter_iterations_used;
        linear_function<Real> drop;
        drop.set(0.0, begin_threshold, (Real)(testnumber-1),end_threshold);
        x_exact.resize(A.rows(),1.0);
        //x_exact.resize_set_natural_numbers(A.rows());
        A.matrix_vector_multiplication(ID,x_exact,b);
        std::cout<<"*************************************************************************"<<std::endl;
        A.print_info();
        std::cout<<"beginning with "<<matrix_name<<" using preconditioner number "<<IP.get_PRECON_PARAMETER()<<" with preprocessing "<<IP.get_PREPROCESSING().string()<<std::endl;
        std::cout<<"    ";
        for(Integer i=0;i<testnumber;i++){
            IP.set_threshold(drop(i));
            eps_rel_residual = eps;
            abs_residual = eps;
            max_iter_iterations_used = max_iter;
            abs_error = -100.0;  // dummy
            if (write_detailed_output)
                solve_with_multilevel_preconditioner_with_detailed_output<T,matrix_type,vector_type>(A,b,x_exact,x,true,eps_rel_residual,abs_residual, max_iter_iterations_used, abs_error,time,directory,matrix_name,output_directory,IP);
            else
                solve_with_multilevel_preconditioner<T,matrix_type,vector_type>(A,b,x_exact,x,true,eps_rel_residual,abs_residual, max_iter_iterations_used, abs_error,time,directory,matrix_name,IP);
            std::cout<<i<<" "<<std::flush;
        }
        std::cout<<std::endl;
        std::cout<<"*************************************************************************"<<std::endl;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"test_multilevel_preconditioner: "<<ippe.error_message()<<". Returning."<<std::endl;
        return;
    }
}


template <class T, class matrix_type, class vector_type>
    void test_multilevel_preconditioner_with_random_matrix(iluplusplus_precond_parameter &IP,
               Integer dim, Integer min_nnz, Integer max_nnz,
               Real begin_threshold, Real end_threshold, Integer testnumber, Real eps, Integer max_iter,
               Real& time, std::string directory, bool write_detailed_output, std::string output_directory)
{
    try {
        matrix_type A;
        std::string matrix_name = ("random-"+integertostring(dim)+"-"+integertostring(min_nnz)+"-"+integertostring(max_nnz));
        A.random(dim,dim,ROW,min_nnz,max_nnz);
        test_multilevel_preconditioner<T,matrix_type,vector_type>(IP,A,begin_threshold,end_threshold,testnumber,eps,max_iter,time,directory,matrix_name,write_detailed_output,output_directory);
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"test_multilevel_preconditioner_with_random_matrix: "<<ippe.error_message()<<". Returning."<<std::endl;
        return;
    }
}

} // end namespace iluplusplus

#endif
