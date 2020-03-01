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


#ifndef ITERATIVE_SOLVERS_IMPLEMENTATION_H
#define ITERATIVE_SOLVERS_IMPLEMENTATION_H

#include <ctime>

#include "declarations.h"
#include "preconditioner.h"

#include "preconditioner_implementation.h"


namespace iluplusplus {

//*************************************************************************************************************************************
// Complementary Functions                                                                                                            *
//*************************************************************************************************************************************


template<class matrix_type, class vector_type>
  void normalize_matrix(matrix_type& A){
    try {
        vector_type norms;
        norms.norm2_of_dim1(A,ROW);
        A.inverse_scale(norms,ROW);
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"normalize_matrix: "<<ippe.error_message()<<std::endl;
        throw;
    }
}

template<class matrix_type, class vector_type>
  void normalize_rows(matrix_type& A){
    try {
        vector_type norms;
        norms.norm2_of_dim1(A,ROW);
        A.inverse_scale(norms,ROW);
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"normalize_rows: "<<ippe.error_message()<<std::endl;
        throw;
    }
}

template<class matrix_type, class vector_type>
  void normalize_columns(matrix_type& A){
    try {
        vector_type norms;
        norms.norm2_of_dim1(A,COLUMN);
        A.inverse_scale(norms,COLUMN);
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"normalize_columns: "<<ippe.error_message()<<std::endl;
        throw;
    }
}


template<class matrix_type, class vector_type>
  void normalize_equations(matrix_type& A, vector_type& b){
    try {
        if(non_fatal_error(A.rows() != b.dimension(), "normalize_equations: number of rows of matrix does not equal number of entries in rhs.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        vector_type norms;
        norms.norm2_of_dim1(A,ROW);
        A.inverse_scale(norms,ROW);
        b.divide(norms);
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"normalize_equations: "<<ippe.error_message()<<std::endl;
        throw;
    }
}


template<class matrix_type, class vector_type>
  void normalize_equations(matrix_type& Arow, matrix_type& Acol, vector_type& b){
    try {
        if(non_fatal_error(Arow.rows() != b.dimension() || Acol.rows() != b.dimension(), "normalize_equations: number of rows of matrix does not equal number of entries in rhs.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        vector_type norms;
        norms.norm2_of_dim1(Arow,ROW);
        Arow.inverse_scale(norms,ROW);
        Acol.inverse_scale(norms,ROW);
        b.divide(norms);
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"normalize_equations: "<<ippe.error_message()<<std::endl;
        throw;
    }
}

template<class matrix_type, class vector_type>
  void normalize(matrix_type& A, vector_type& D_l, vector_type& D_r){
    try {
        D_l.norm2_of_dim1(A,COLUMN);
        A.inverse_scale(D_l,COLUMN);
        D_r.norm2_of_dim1(A,ROW);
        A.inverse_scale(D_r,ROW);
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"normalize_equations: "<<ippe.error_message()<<std::endl;
        throw;
    }
}



//*************************************************************************************************************************************
// CG                                                                                                                                 *
//*************************************************************************************************************************************


template<class T, class matrix_type, class vector_type>
    bool cg(const preconditioner<T,matrix_type,vector_type>& P,
             preconditioner_application1_type pa1,
             const matrix_type& A, const vector_type& b,vector_type& x,
             Integer min_iter, Integer& max_iter,
             Real& rel_tol, Real& abs_tol, bool use_0_as_starting_value)
   {
    try {
       if (P.compatibility_check(pa1,A,b) || !A.square_check()){
           std::cout<<"cg: incompatible dimensions; returning zero vector of dimension 0"<<std::endl;
           std::cout<<"Information on matrix:"<<std::endl;
           A.print_info();
           std::cout<<"Dimension of right hand side: "<<b.dimension()<<std::endl;
           std::cout<<"Information on preconditioner: Pre-image size: "<<P.pre_image_dimension()<<" image size: "<<P.image_dimension()<<std::endl;
           #ifdef VERBOSE
               P.print_info();
           #endif
           rel_tol = log10(-1.0);
           abs_tol = log10(-1.0);
           x.resize(0,0.0);
           return false;
       }
       // convert to actual measures
       rel_tol = std::exp(-rel_tol*std::log(10.0));
       abs_tol = std::exp(-abs_tol*std::log(10.0));
       // rel_tol = -log10(rel_tol);
       // abs_tol = -log10(-abs_tol);
       Integer problem_size = A.rows();
       vector_type r(problem_size);
       vector_type y(problem_size);
       vector_type p(problem_size);
       vector_type Ap(problem_size);
       T  alpha, beta, gamma_old, gamma_new;
       Integer iter;
       Real initial_res, res;
       // calculate initial residual r=L'(b-Ax), L' left part of preconditioner
       if (use_0_as_starting_value) {
           p = b;
       } else {
           if(non_fatal_error(x.dimension() != problem_size, "cg: starting value has wrong dimension")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           p.residual(ID,A,x,b);             //p = b-Ax
           P.preconditioned_starting_value(pa1,A,x,y);
       }
       P.preconditioned_rhs(pa1,A,p,r);  // r= L'(b-Ax), L' being the left part of the preconditioner
       // other initializations
       p=r;
       initial_res=r.norm2();
       res = initial_res;
       iter = 0;
       gamma_old = r*r;
       while( (((res/initial_res>rel_tol) ||  res > abs_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
       // while( ((res/initial_res>rel_tol)&&(iter<max_iter)) || (iter<min_iter) ){
           iter++;
           P.preconditioned_matrix_vector_multiplication(pa1,A,p,Ap);   // Ap = precond(A*p);
           alpha = gamma_old / (Ap*p);
           y.add_scaled(alpha, p);          // y = y + alpha*p
           r.add_scaled(-alpha, Ap);        // r = r - alpha*A*p
           gamma_new =r*r;
           beta = gamma_new/gamma_old;
           gamma_old = gamma_new;
           p.scale_add(beta, r);                        // p = beta*p + r
           res = r.norm2();
       }
       P.adapt_solution(pa1,A,y,x);     // y=R'x, R' an appropriate right part of the preconditioner
       max_iter = iter;
       rel_tol = -log10(res/initial_res);
       abs_tol = -log10(res);
       return true;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"cg: "<<ippe.error_message()<<std::endl;
        rel_tol = log10(-1.0);
        abs_tol = log10(-1.0);
        x.resize(0,0.0);
        return false;
    }
 }



//*************************************************************************************************************************************
// CGNR                                                                                                                               *
//*************************************************************************************************************************************


template<class T, class matrix_type, class vector_type>
    bool cgnr(const preconditioner<T,matrix_type,vector_type>& P,
             preconditioner_application1_type pa1,
             const matrix_type& A, const vector_type& b,vector_type& x,
             Integer min_iter, Integer& max_iter,
             Real& rel_tol, Real& abs_tol, bool use_0_as_starting_value)
   {
   try {
       if (P.compatibility_check(pa1,A,b) || !A.square_check()){
           std::cout<<"cgnr: incompatible dimensions; returning zero vector of dimension 0"<<std::endl;
           std::cout<<"Information on matrix:"<<std::endl;
           A.print_info();
           std::cout<<"Dimension of right hand side: "<<b.dimension()<<std::endl;
           std::cout<<"Information on preconditioner: Pre-image size: "<<P.pre_image_dimension()<<" image size: "<<P.image_dimension()<<std::endl;
           #ifdef VERBOSE
               P.print_info();
           #endif
           rel_tol = log10(-1.0);
           abs_tol = log10(-1.0);
           x.resize(0,0.0);
           return false;
       }
       // convert to actual measures
       rel_tol = std::exp(-rel_tol*std::log(10.0));
       abs_tol = std::exp(-abs_tol*std::log(10.0));
       Integer problem_size = A.rows();
       vector_type r(problem_size);
       vector_type z(problem_size);
       vector_type p(problem_size);
       vector_type w(problem_size);
       vector_type y(problem_size);
       T  alpha, beta, gamma_old, gamma_new;
       Integer iter;
       Real initial_res, res;
       // calculate initial residual r=L'(b-Ax), L' left part of preconditioner
       if (use_0_as_starting_value) {
           p = b;
       } else {
           if(non_fatal_error(x.dimension() != problem_size, "cgnr: starting value has wrong dimension")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           p.residual(ID,A,x,b);             //p = b-Ax
           P.preconditioned_starting_value(pa1,A,x,y);
       }
       P.preconditioned_rhs(pa1,A,p,r);  // r= L'(b-Ax), L' being the left part of the preconditioner
       // other initializations
       P.preconditioned_matrix_transposed_vector_multiplication(pa1,A,r,z);
       p=z;
       initial_res=r.norm2();
       res = initial_res;
       iter = 0;
       gamma_old = z*z;
       while( (((res/initial_res>rel_tol) ||  res > abs_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
       //while( ((res/initial_res>rel_tol)&&(iter<max_iter)) || (iter<min_iter) ){
           iter++;
           P.preconditioned_matrix_vector_multiplication(pa1,A,p,w);   // w = precond(A*p);
           alpha = gamma_old / (w*w);
           y.add_scaled(alpha, p);          // y = y + alpha*p
           r.add_scaled(-alpha, w);        // r = r - alpha*A*p
           P.preconditioned_matrix_transposed_vector_multiplication(pa1,A,r,z);
           gamma_new =z*z;
           beta = gamma_new/gamma_old;
           gamma_old = gamma_new;
           p.scale_add(beta, r);                        // p = beta*p + r
           res = r.norm2();
       }
       P.adapt_solution(pa1,A,y,x);     // y=R'x, R' an appropriate right part of the preconditioner
       max_iter = iter;
       rel_tol = -log10(res/initial_res);
       abs_tol = -log10(res);
       return true;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"cgnr: "<<ippe.error_message()<<std::endl;
        rel_tol = log10(-1.0);
        abs_tol = log10(-1.0);
        x.resize(0,0.0);
        return false;
    }
}


//*************************************************************************************************************************************
// CGNE                                                                                                                               *
//*************************************************************************************************************************************


template<class T, class matrix_type, class vector_type>
    bool cgne(const preconditioner<T,matrix_type,vector_type>& P,
             preconditioner_application1_type pa1,
             const matrix_type& A, const vector_type& b,vector_type& x,
             Integer min_iter, Integer& max_iter,
             Real& rel_tol, Real& abs_tol, bool use_0_as_starting_value)
   {
   try {
       if (P.compatibility_check(pa1,A,b) || !A.square_check()){
           std::cout<<"cgne: incompatible dimensions; returning zero vector of dimension 0"<<std::endl;
           std::cout<<"Information on matrix:"<<std::endl;
           A.print_info();
           std::cout<<"Dimension of right hand side: "<<b.dimension()<<std::endl;
           std::cout<<"Information on preconditioner: Pre-image size: "<<P.pre_image_dimension()<<" image size: "<<P.image_dimension()<<std::endl;
           #ifdef VERBOSE
               P.print_info();
           #endif
           rel_tol = log10(-1.0);
           abs_tol = log10(-1.0);
           x.resize(0,0.0);
           return false;
       }
       // convert to actual measures
       rel_tol = std::exp(-rel_tol*std::log(10.0));
       abs_tol = std::exp(-abs_tol*std::log(10.0));
       Integer problem_size = A.rows();
       vector_type r(problem_size);
       vector_type p(problem_size);
       vector_type Ap(problem_size);
       vector_type ATr(problem_size);
       vector_type w(problem_size);
       vector_type y(problem_size);
       T  alpha, beta, gamma_old, gamma_new;
       Integer iter;
       Real initial_res, res;
       // calculate initial residual r=L'(b-Ax), L' left part of preconditioner
       if (use_0_as_starting_value) {
           p = b;
       } else {
           if(non_fatal_error(x.dimension() != problem_size, "cgnr: starting value has wrong dimension")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           p.residual(ID,A,x,b);             //p = b-Ax
           P.preconditioned_starting_value(pa1,A,x,y);
       }
       P.preconditioned_rhs(pa1,A,p,r);  // r= L'(b-Ax), L' being the left part of the preconditioner
       // other initializations
       P.preconditioned_matrix_transposed_vector_multiplication(pa1,A,r,p);
       initial_res=r.norm2();
       res = initial_res;
       iter = 0;
       gamma_old = r*r;
       while( (((res/initial_res>rel_tol) ||  res > abs_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
       //while( ((res/initial_res>rel_tol)&&(iter<max_iter)) || (iter<min_iter) ){
           iter++;
           alpha = gamma_old / (p*p);
           P.preconditioned_matrix_vector_multiplication(pa1,A,p,Ap);   // Ap = precond(A*p);
           y.add_scaled(alpha, p);          // y = y + alpha*p
           r.add_scaled(-alpha, Ap);        // r = r - alpha*A*p
           gamma_new =r*r;
           beta = gamma_new/gamma_old;
           gamma_old = gamma_new;
           P.preconditioned_matrix_transposed_vector_multiplication(pa1,A,r,ATr);
           p.scaled_vector_addition(ATr,beta,r);                        // p = AT*r + beta*p 
           res = r.norm2();
       }
       P.adapt_solution(pa1,A,y,x);     // y=R'x, R' an appropriate right part of the preconditioner
       max_iter = iter;
       rel_tol = -log10(res/initial_res);
       abs_tol = -log10(res);
       return true;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"cgne: "<<ippe.error_message()<<std::endl;
        rel_tol = log10(-1.0);
        abs_tol = log10(-1.0);
        x.resize(0,0.0);
        return false;
    }
}



template<class T, class matrix_type, class vector_type>
    bool bicgstab(const preconditioner<T,matrix_type,vector_type>& P,
             preconditioner_application1_type pa1,
             const matrix_type& A, const vector_type& b,vector_type& x,
             Integer min_iter, Integer& max_iter,
             Real& rel_tol, Real& abs_tol, bool use_0_as_starting_value)
   {
   try {
       #ifdef IT_TIME
           clock_t time_1,time_2,time_3,time_4,time_5,time_6,time_7;
           Real time_prepost = 0.0;
           Real time_apply_precond=0.0;
           Real time_vector_ops=0.0;
           Real time_total=0.0;
           time_1 = clock();
       #endif
       #ifdef VERBOSE
           std::cout<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl;
           std::cout<<"     ***** bicgstab results: "<<std::endl;
           std::cout<<"     *****          matrix size         : "<<A.rows()<<std::endl;
       #endif
       if (P.compatibility_check(pa1,A,b) || !A.square_check()){
           std::cout<<"bicgstab: incompatible dimensions; returning zero vector of dimension 0"<<std::endl;
           std::cout<<"Information on matrix:"<<std::endl;
           A.print_info();
           std::cout<<"Dimension of right hand side: "<<b.dimension()<<std::endl;
           std::cout<<"Information on preconditioner: Pre-image size: "<<P.pre_image_dimension()<<" image size: "<<P.image_dimension()<<std::endl;
           #ifdef VERBOSE
               P.print_info();
           #endif
           rel_tol = log10(-1.0);
           abs_tol = log10(-1.0);
           x.resize(0,0.0);
           //throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           return false;
       }
       // convert to actual measures
       rel_tol = std::exp(-rel_tol*std::log(10.0));
       abs_tol = std::exp(-abs_tol*std::log(10.0));
       Integer problem_size = A.rows();
       vector_type r(problem_size, 0.0);
       vector_type y(problem_size, 0.0);
       vector_type p(problem_size, 0.0);
       vector_type s(problem_size, 0.0);
       vector_type Ap(problem_size, 0.0);
       vector_type As(problem_size, 0.0);
       vector_type r0star(problem_size);
       T omega, alpha, beta, dot_r_r0star;
       Integer iter;
       Real initial_res, res; 
       T product;  // will actually be real
       // calculate initial residual r=L'(b-Ax), L' left part of preconditioner
       if (use_0_as_starting_value) {
           r0star = b;
       } else {
           if(non_fatal_error(x.dimension() != problem_size, "bicgstab: starting value has wrong dimension")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           r0star.residual(ID,A,x,b);             //r0star = b-Ax
           P.preconditioned_starting_value(pa1,A,x,y);
       }
       P.preconditioned_rhs(pa1,A,r0star,r);  // r= L'(b-Ax), L' being the left part of the preconditioner
       // other initializations
       r0star=r;
       p=r;
       initial_res=r.norm2();
       res = initial_res;
       iter = 0;
       #ifdef IT_TIME
           time_6 = time_2 = clock();
           time_prepost += ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
       #endif
       while( (((res/initial_res>rel_tol) ||  res > abs_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
       //while( ((res/initial_res>rel_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
           #ifdef IT_TIME
               time_2 = clock();
           #endif
           iter++;
           P.preconditioned_matrix_vector_multiplication(pa1,A,p,Ap);   // Ap = precond(A*p)
           #ifdef IT_TIME
               time_3 = clock();
               time_apply_precond += ((Real)time_3-(Real)time_2)/(Real)CLOCKS_PER_SEC;
           #endif
           dot_r_r0star = r*r0star;
           alpha = dot_r_r0star / (Ap*r0star);
           s.scaled_vector_addition(r,-alpha,Ap);                          // s = r-alpha*Ap
           #ifdef IT_TIME
               time_4 = clock();
               time_vector_ops += ((Real)time_4-(Real)time_3)/(Real)CLOCKS_PER_SEC;
           #endif
           P.preconditioned_matrix_vector_multiplication(pa1,A,s,As);   // As = precond(A*s)
           #ifdef IT_TIME
               time_5 = clock();
               time_apply_precond += ((Real)time_5-(Real)time_4)/(Real)CLOCKS_PER_SEC;
           #endif
           product = (As*As);
           #ifdef DEBUG
               if(product<1e-16){std::cerr<<"bicgstab breakdown: s = 0"<<std::endl;};
           #endif
           omega = (As*s)/product;
           non_fatal_error(omega==0.0,"bicgstab: breakdown: omega = 0.");
           y.add_scaled(alpha, p);                                         // y = y+alpha*p
           y.add_scaled(omega, s);                                         // y = y+omega*s
           r.scaled_vector_addition(s,-omega,As);
           beta = ((r*r0star)/dot_r_r0star)*(alpha/omega);
           p.add_scaled(-omega, Ap);       // p = p -omega*A*p
           p.scale_add(beta,r);            // p = beta*p + r
           res = r.norm2();
           #ifdef IT_TIME
               time_6 = clock();
               time_vector_ops += ((Real)time_6-(Real)time_5)/(Real)CLOCKS_PER_SEC;
           #endif
       }
       P.adapt_solution(pa1,A,y,x);     // y=R'x, R' an appropriate right part of the preconditioner
       max_iter = iter;
       bool success = ( ( (res/initial_res<rel_tol) && res < abs_tol)  );
       rel_tol = -log10(res/initial_res);
       abs_tol = -log10(res);
       #ifdef VERBOSE
           std::cout<<"     *****          iterations:         : "<<iter<<std::endl;
           std::cout<<"     *****          absolute residual   : "<<abs_tol<<std::endl;
           std::cout<<"     *****          relative residual   : "<<rel_tol<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl<<std::endl;
       #endif
       #ifdef IT_TIME
           time_7 = clock();
           time_prepost += ((Real)time_7-(Real)time_6)/(Real)CLOCKS_PER_SEC;
           time_total = ((Real)time_7-(Real)time_1)/(Real)CLOCKS_PER_SEC;
           std::cout<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl;
           std::cout<<"     ***** bicgstab timing results: "<<std::endl;
           std::cout<<"     *****          total time:         : "<<time_total<<std::endl;
           std::cout<<"     *****          pre- postprocessing : "<<time_prepost<<std::endl;
           std::cout<<"     *****          preconditioned mult.: "<<time_apply_precond<<std::endl;
           std::cout<<"     *****          vector operations   : "<<time_vector_ops<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl<<std::endl;
       #endif
       return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"bicgstab: "<<ippe.error_message()<<std::endl;
        rel_tol = log10(-1.0);
        abs_tol = log10(-1.0);
        x.resize(0,0.0);
        return false;
    }
}



template<class T, class matrix_type, class vector_type>
    bool richardson(const preconditioner<T,matrix_type,vector_type>& P,
             preconditioner_application1_type pa1,
             const matrix_type& A, const vector_type& b,vector_type& x,
             Integer min_iter, Integer& max_iter,
             Real& rel_tol, Real& abs_tol, bool use_0_as_starting_value)
    {
    try {
       #ifdef IT_TIME
           clock_t time_1,time_2,time_3,time_4,time_5,time_6,time_7;
           Real time_prepost = 0.0;
           Real time_apply_precond=0.0;
           Real time_vector_ops=0.0;
           Real time_total=0.0;
           time_1 = clock();
       #endif
       #ifdef VERBOSE
           std::cout<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl;
           std::cout<<"     ***** richardson results: "<<std::endl;
           std::cout<<"     *****          matrix size         : "<<A.rows()<<std::endl;
       #endif
       if (P.compatibility_check(pa1,A,b) || !A.square_check()){
           std::cout<<"richardson: incompatible dimensions; returning zero vector of dimension 0"<<std::endl;
           std::cout<<"Information on matrix:"<<std::endl;
           A.print_info();
           std::cout<<"Dimension of right hand side: "<<b.dimension()<<std::endl;
           std::cout<<"Information on preconditioner: Pre-image size: "<<P.pre_image_dimension()<<" image size: "<<P.image_dimension()<<std::endl;
           #ifdef VERBOSE
               P.print_info();
           #endif
           rel_tol = log10(-1.0);
           abs_tol = log10(-1.0);
           x.resize(0,0.0);
           return false;
       }
       // convert to actual measures
       rel_tol = std::exp(-rel_tol*std::log(10.0));
       abs_tol = std::exp(-abs_tol*std::log(10.0));
       Integer problem_size = A.rows();
       vector_type r(problem_size, 0);
       vector_type z(problem_size, 0);
       Integer iter;
       Real initial_res, res;
       // calculate initial residual r=L'(b-Ax), L' left part of preconditioner
       if (use_0_as_starting_value) {
           P.preconditioned_rhs(pa1,A,b,r);  // r is preconditioned residual
       } else {
           if(non_fatal_error(x.dimension() != problem_size, "richardson: starting value has wrong dimension")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           P.preconditioned_residual(pa1,A,b,x,r);
           P.preconditioned_starting_value(pa1,A,x,z);
       }
       // other initializations
       initial_res=r.norm2();
       res = initial_res;
       iter = 0;
       #ifdef IT_TIME
           time_6 = time_2 = clock();
           time_prepost += ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
       #endif
       while( (((res/initial_res>rel_tol) ||  res > abs_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
       //while( ((res/initial_res>rel_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
           #ifdef IT_TIME
               time_2 = clock();
           #endif
           iter++;
           z.add(r);
           #ifdef IT_TIME
               time_3 = clock();
               time_vector_ops += ((Real)time_3-(Real)time_2)/(Real)CLOCKS_PER_SEC;
           #endif
           P.adapt_solution(pa1,A,z,x);
           P.preconditioned_residual(pa1,A,b,x,r);
           #ifdef IT_TIME
               time_4 = clock();
               time_apply_precond += ((Real)time_4-(Real)time_3)/(Real)CLOCKS_PER_SEC;
           #endif
           res = r.norm2();
           #ifdef IT_TIME
               time_5 = clock();
               time_vector_ops += ((Real)time_5-(Real)time_4)/(Real)CLOCKS_PER_SEC;
           #endif
       }
       max_iter = iter;
       bool success = ( ( (res/initial_res<rel_tol) && res < abs_tol)  );
       rel_tol = -log10(res/initial_res);
       abs_tol = -log10(res);
       #ifdef VERBOSE
           std::cout<<"     *****          iterations:         : "<<iter<<std::endl;
           std::cout<<"     *****          absolute residual   : "<<abs_tol<<std::endl;
           std::cout<<"     *****          relative residual   : "<<rel_tol<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl<<std::endl;
       #endif
       #ifdef IT_TIME
           time_7 = clock();
           time_prepost += ((Real)time_7-(Real)time_6)/(Real)CLOCKS_PER_SEC;
           time_total = ((Real)time_7-(Real)time_1)/(Real)CLOCKS_PER_SEC;
           std::cout<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl;
           std::cout<<"     ***** bicgstab timing results: "<<std::endl;
           std::cout<<"     *****          total time:         : "<<time_total<<std::endl;
           std::cout<<"     *****          pre- postprocessing : "<<time_prepost<<std::endl;
           std::cout<<"     *****          preconditioned mult.: "<<time_apply_precond<<std::endl;
           std::cout<<"     *****          vector operations   : "<<time_vector_ops<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl<<std::endl;
       #endif
       return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"richardson: "<<ippe.error_message()<<std::endl;
        rel_tol = log10(-1.0);
        abs_tol = log10(-1.0);
        x.resize(0,0.0);
        return false;
    }
}


//*************************************************************************************************************************************
// CGS                                                                                                                                *
//*************************************************************************************************************************************


template<class T, class matrix_type, class vector_type>
    bool cgs(const preconditioner<T,matrix_type,vector_type>& P,
             preconditioner_application1_type pa1,
             const matrix_type& A, const vector_type& b,vector_type& x,
             Integer min_iter, Integer& max_iter,
             Real& rel_tol, Real& abs_tol, bool use_0_as_starting_value)
   {
   try {
       if (P.compatibility_check(pa1,A,b) || !A.square_check()){
           std::cout<<"cgs: incompatible dimensions; returning zero vector of dimension 0"<<std::endl;
           std::cout<<"Information on matrix:"<<std::endl;
           A.print_info();
           std::cout<<"Dimension of right hand side: "<<b.dimension()<<std::endl;
           std::cout<<"Information on preconditioner: Pre-image size: "<<P.pre_image_dimension()<<" image size: "<<P.image_dimension()<<std::endl;
           #ifdef VERBOSE
               P.print_info();
           #endif
           rel_tol = log10(-1.0);
           abs_tol = log10(-1.0);
           x.resize(0,0.0);
           return false;
       }
       // convert to actual measures
       rel_tol = std::exp(-rel_tol*std::log(10.0));
       abs_tol = std::exp(-abs_tol*std::log(10.0));
       Integer problem_size = A.rows();
       vector_type y(problem_size, 0.0);
       vector_type r(problem_size, 0.0);
       vector_type r0star(problem_size, 0.0);
       vector_type p(problem_size, 0.0);
       vector_type q(problem_size, 0.0);
       vector_type h(problem_size, 0.0);
       vector_type u(problem_size, 0.0);
       vector_type uq(problem_size, 0.0);
       T alpha, beta, dot_r_r0star_old, dot_r_r0star_new;
       Integer iter;
       Real initial_res, res;
       // calculate initial residual r=L'(b-Ax), L' left part of preconditioner
       if (use_0_as_starting_value) {
           r0star = b;
       } else {
           if(non_fatal_error(x.dimension() != problem_size, "cgs: starting value has wrong dimension")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           r0star.residual(ID,A,x,b);             //r0star = b-Ax
           P.preconditioned_starting_value(pa1,A,x,y);
       }
       P.preconditioned_rhs(pa1,A,r0star,r);  // r= L'(b-Ax),  L' being the left part of the preconditioner
       // other initializations
       r0star=r; // now both r and r0star contain the correct residual
       p=r;
       u=r;
       initial_res=r.norm2();
       res = initial_res;
       dot_r_r0star_old =r*r0star;
       iter = 0;
       while( (((res/initial_res>rel_tol) ||  res > abs_tol) &&(iter<max_iter) )|| (iter<min_iter) ){
       //while( ((res/initial_res>rel_tol)&&(iter<max_iter)) || (iter<min_iter) ){
           iter++;
           P.preconditioned_matrix_vector_multiplication(pa1,A,p,h);   // Ap= h = precond(A*p)
           alpha = dot_r_r0star_old / (h*r0star);
           q.scaled_vector_addition(u,-alpha,h);
           uq.vector_addition(u,q);
           uq.scale(alpha);
           y.add(uq);
           P.preconditioned_matrix_vector_multiplication(pa1,A,uq,h);   // Auq = h = precond(A*uq)
           r.subtract(h);
           dot_r_r0star_new = r*r0star;
           beta = dot_r_r0star_new/dot_r_r0star_old;
           dot_r_r0star_old = dot_r_r0star_new;
           u.scaled_vector_addition(r,beta,q);   // u = r + beta*q
           h.scaled_vector_addition(q,beta,p);   // h = q + beta*p
           p.scaled_vector_addition(u,beta,h);   // p = u + beta*h = u+beta*(q+beta*p)
           res = r.norm2();
       }
       P.adapt_solution(pa1,A,y,x);     // y=R'x, R' an appropriate right part of the preconditioner
       max_iter = iter;
       bool success = ( ( (res/initial_res<rel_tol) && res < abs_tol)  );
       rel_tol = -log10(res/initial_res);
       abs_tol = -log10(res);
       return success;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"cgs: "<<ippe.error_message()<<std::endl;
        rel_tol = log10(-1.0);
        abs_tol = log10(-1.0);
        x.resize(0,0.0);
        return false;
    }
}



//*************************************************************************************************************************************
// GMRES
//*************************************************************************************************************************************

//*************************************************************************************************************************************
// Needed Functions                                                                                                                   *
//*************************************************************************************************************************************



template<class Real> void GeneratePlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn){
    if (dy == 0.0) {
        cs = 1.0;
        sn = 0.0;
     } else if (fabs(dy) > fabs(dx)) {
                Real temp = dx / dy;
                sn = 1.0 / sqrt( 1.0 + temp*temp );
                cs = temp * sn;
            } else {
                Real temp = dy / dx;
                cs = 1.0 / sqrt( 1.0 + temp*temp );
                sn = temp * cs;
            }
  }

template<class Real> void ApplyPlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn){
    Real temp  =  cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
  }

template<class matrix_type, class vector_type>
    void Update(vector_type &x, Integer k, matrix_type &H, const vector_type &s, std::vector<vector_type>& v)
    {
       Integer i,j;
       vector_type y(s, 0.0);
       // Backsolve:
       for (i = k; i >= 0; i--) {
           y[i] /= H(i,i);
           for (j = i - 1; j >= 0; j--){
               y[j] -= H(j,i) * y[i];   }
       }
       for (j = 0; j <= k; j++) {
           x.add_scaled(y[j],v[j]);   }
    }

#if 0           // cfh: currently disabled because of the matrix_dense reference
template<class T, class matrix_type, class vector_type>
     bool gmres(const preconditioner<T,matrix_type,vector_type>& P,
             preconditioner_application1_type pa1,
             const matrix_type &A, const vector_type &b, vector_type &x,
             Integer &restart, Integer min_iter, Integer &max_iter,
             Real& rel_tol, Real& abs_tol, bool use_0_as_starting_value)

   // m= restart, M=P
   {
   try {
       #ifdef VERBOSE
           std::cout<<std::endl;
           std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl;
           std::cout<<"     ***** gmres results: "<<std::endl;
           std::cout<<"     *****          matrix size         : "<<A.rows()<<std::endl;
       #endif
       if (P.compatibility_check(pa1,A,b) || !A.square_check()){
           std::cout<<"gmres: incompatible dimensions; returning zero vector of dimension 0"<<std::endl;
           std::cout<<"Information on matrix:"<<std::endl;
           A.print_info();
           std::cout<<"Dimension of right hand side: "<<b.dimension()<<std::endl;
           std::cout<<"Information on preconditioner: Pre-image size: "<<P.pre_image_dimension()<<" image size: "<<P.image_dimension()<<std::endl;
           #ifdef VERBOSE
               P.print_info();
           #endif
           rel_tol = log10(-1.0);
           abs_tol = log10(-1.0);
           x.resize(0,0.0);
           return false;
       }
       // convert to actual measures
       rel_tol = std::exp(-rel_tol*std::log(10.0));
       abs_tol = std::exp(-abs_tol*std::log(10.0));
       Integer size = A.rows();
       if(size < 1){x.resize(0,0.0); return false;}
       Real rel_resid, initial_res, res;
       Integer i, j = 1, k;
       vector_type s(restart+1, 0.0), cs(restart+1, 0.0), sn(restart+1, 0.0), w;
       matrix_dense<T> H(restart+1,restart+1,0.0);

       vector_type r,y;
       vector_type h;    // maybe use w

       P.preconditioned_rhs(pa1,A,b,r);
       initial_res = r.norm2();

       // calculate initial residual r=L'(b-Ax), L' left part of preconditioner
       if (use_0_as_starting_value) {
           h = b;
           y.resize(size,0.0);
       } else {
           if(non_fatal_error(x.dimension() != size, "gmres: starting value has wrong dimension")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
           h.residual(ID,A,x,b);             //h = b-Ax
           P.preconditioned_starting_value(pa1,A,x,y);
       }
       P.preconditioned_rhs(pa1,A,h,r);  // r= L'(b-Ax),  L' being the left part of the preconditioner
       if (initial_res == 0.0) initial_res = 1.0;
       res = r.norm2();
       rel_resid =res/initial_res;
       if ((rel_resid  < rel_tol)&&(res<abs_tol)&&(j>=min_iter)){
       //if ((rel_resid  < rel_tol)){
           rel_tol = -log10(rel_resid);
           abs_tol = -log10(res);
           max_iter = 0;
           #ifdef VERBOSE
               std::cout<<"     *****          iterations:         : "<<max_iter<<std::endl;
               std::cout<<"     *****          absolute residual   : "<<abs_tol<<std::endl;
               std::cout<<"     *****          relative residual   : "<<rel_tol<<std::endl;
               std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl<<std::endl;
           #endif
           return true;
       }
       std::vector<vector_type> v(restart+1);
       for(i=0;i<restart+1;i++) v[i].resize(size);
       while (j <= max_iter) {
           (v[0]).scale(1.0/res,r);
           s.set_all(0.0);
           s[0] = res;
           for (i = 0; i < restart && j <= max_iter; i++, j++) {
               P.preconditioned_matrix_vector_multiplication(pa1,A,v[i],w);
               for (k = 0; k <= i; k++) {
                   H(k, i) = w * v[k];
                   w.add_scaled(-H(k,i), v[k]);
               }
               H(i+1,i) = w.norm2();
               (v[i+1]).scale(1.0/H(i+1,i),w);
               for (k = 0; k < i; k++){
                   ApplyPlaneRotation(H(k,i), H(k+1,i), cs[k], sn[k]);
               }
               GeneratePlaneRotation(H(i,i), H(i+1,i), cs[i], sn[i]);
               ApplyPlaneRotation(H(i,i), H(i+1,i), cs[i], sn[i]);
               ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i]);
               res = fabs(s[i+1]);
               rel_resid = res / initial_res;
               if ((rel_resid  < rel_tol)&&(res<abs_tol)&&(j>=min_iter)) {
               //if ((rel_resid  < rel_tol)&&(j>=min_iter)) {
                   Update(y, i, H, s, v);
                   rel_tol = -log10(rel_resid);
                   abs_tol = -log10(res);
                   max_iter = j;
                   P.adapt_solution(pa1,A,y,x);
                   #ifdef VERBOSE
                       std::cout<<"     *****          iterations:         : "<<max_iter<<std::endl;
                       std::cout<<"     *****          absolute residual   : "<<abs_tol<<std::endl;
                       std::cout<<"     *****          relative residual   : "<<rel_tol<<std::endl;
                       std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl<<std::endl;
                   #endif
                   return true;
               }  //end if
           } // end for i,j
           Update(y, i - 1, H, s, v);
           rel_resid = res / initial_res;
           if ((rel_resid  < rel_tol)&&(res<abs_tol)&&(j>=min_iter)){
           //if (rel_resid < rel_tol&&(j>=min_iter)) {
               rel_tol = -log10(rel_resid);
               abs_tol = -log10(res);
               max_iter = j;
               P.adapt_solution(pa1,A,y,x);
               #ifdef VERBOSE
                   std::cout<<"     *****          iterations:         : "<<max_iter<<std::endl;
                   std::cout<<"     *****          absolute residual   : "<<abs_tol<<std::endl;
                   std::cout<<"     *****          relative residual   : "<<rel_tol<<std::endl;
                   std::cout<<"     ***** ***** ***** ***** ***** ***** *****"<<std::endl<<std::endl;
               #endif
               return true;
           }  // end if
       }  // end while
       P.adapt_solution(pa1,A,y,x);
       rel_tol = -log10(rel_resid);
       abs_tol = -log10(res);
       return false;
    }
    catch(iluplusplus_error ippe){
        std::cerr<<"gmres: "<<ippe.error_message()<<std::endl;
        rel_tol = log10(-1.0);
        abs_tol = log10(-1.0);
        x.resize(0,0.0);
        return false;
    }
}
#endif

} // end namespace iluplusplus

#endif

