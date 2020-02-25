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


#ifndef DECLARATIONS_H
#define DECLARATIONS_H


#include <new>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <stack>
#include <map>
#include <queue>
#include <vector>
#include <string>
#include <complex>


namespace iluplusplus {

//**************************************************************************************//
//               Typedef
//**************************************************************************************//

typedef double Real;                                     // will be used for those quantities that are always Real (e.g. norms)
typedef int Integer;                                     // for integers, particularly indices
typedef std::complex<Real> Complex;                      // declarations for complex numbers. Will only be used if set below for the Coeff_Field.
#ifndef ILUPLUSPLUS_USES_COMPLEX
    typedef Real Coeff_Field;                            // will be used for the coefficients of vectors and matrices
#endif
#ifdef ILUPLUSPLUS_USES_COMPLEX
    typedef Complex Coeff_Field;                         // will be used for the coefficients of vectors and matrices
#endif
typedef std::multimap<Real,Integer> Multimap;

#ifdef ILUPLUSPLUS_USES_METIS
    typedef Integer idxtype;   // only for use with METIS; METIS must be compiled with the same type as Integer.
#endif

//**************************************************************************************//
//               Constants
//**************************************************************************************//

const Real ln10 = log(10.0);
const Real COMPARE_EPS = 1e-8;  // tolerance for checking equality: used only in some exceptional cases, e.g. tolerance for checking if a diagonal element is 1 in an I-matrix, or if avoiding division by 0 is essential.

//**************************************************************************************//
//               Enumerated Types
//**************************************************************************************//

// types of errors
enum error_type {UNKNOWN_ERROR, INSUFFICIENT_MEMORY, INCOMPATIBLE_DIMENSIONS, ARGUMENT_NOT_ALLOWED, FILE_ERROR, OTHER_ERROR};

// types for matrices
enum orientation_type {ROW, COLUMN};                    // the orientation of a (sparse) matrix
enum matrix_usage_type {ID, TRANSPOSE};                 // determines if a given function uses a matrix argument as it is or transposed.
enum special_matrix_type{UNSTRUCTURED, LOWER_TRIANGULAR, UPPER_TRIANGULAR, PERMUTED_LOWER_TRIANGULAR, PERMUTED_UPPER_TRIANGULAR, DIAGONAL};

// types for preconditioners
enum preconditioner_application1_type {NONE, LEFT, RIGHT, SPLIT};    // the manner in which a preconditioner is to be applied to a system to yield an approximate inverse Q
enum perm_usage_type {NOPERM, PERM1, PERM2};                         // which permutation will be used if needed

// preprocessing types
enum preprocessing_type {
    TEST_ORDERING,
    NORMALIZE_COLUMNS,
    NORMALIZE_ROWS,
#ifdef ILUPLUSPLUS_USES_SPARSPAK
    REVERSE_CUTHILL_MCKEE_ORDERING,
#endif
    PQ_ORDERING,
    DYN_AV_PQ_ORDERING,
    SYMM_PQ,
    UNIT_OR_ZERO_DIAGONAL_SCALING,
    SPARSE_FIRST_ORDERING,
    MAX_WEIGHTED_MATCHING_ORDERING,
#ifdef ILUPLUSPLUS_USES_METIS
    METIS_NODE_ND_ORDERING,
#endif
#ifdef ILUPLUSPLUS_USES_PARDISO
    PARDISO_MAX_WEIGHTED_MATCHING_ORDERING,
#endif
    SYMM_MOVE_CORNER_ORDERING,
    SYMM_MOVE_CORNER_ORDERING_IM,
    SYMB_SYMM_MOVE_CORNER_ORDERING,
    SYMB_SYMM_MOVE_CORNER_ORDERING_IM,
    SP_SYMM_MOVE_CORNER_ORDERING,
    SP_SYMM_MOVE_CORNER_ORDERING_IM,
    WGT_SYMM_MOVE_CORNER_ORDERING,
    WGT_SYMM_MOVE_CORNER_ORDERING_IM,
    WGT2_SYMM_MOVE_CORNER_ORDERING,
    WGT2_SYMM_MOVE_CORNER_ORDERING_IM,
    DD_SYMM_MOVE_CORNER_ORDERING_IM
};

// types for iterative solvers
enum iterative_method_type {BICGSTAB, CG, CGS, GMRES, CGNR, CGNE};

// types of preconditioners
enum preconditioner_type {PC_ILUC, PC_ILUT, PC_ILUTP, PC_ILUCP, PC_ILUCDP, PC_ML_ILUCDP, PC_DLML_ILUCDP, PC_MG_ILUCDP, PC_NOPRECOND};

// types of data for analysis

enum data_type {SUCC_SOLVE, THRESHOLD, FILLIN, MEM_STORAGE, MEM_USED, MEM_ALLOCATED, ITERATIONS, ABS_ERROR, REL_RESIDUAL, ABS_RESIDUAL, SETUP_TIME, ITER_TIME, TOTAL_TIME, MATRIX_DIM, MATRIX_NNZ, LEVELS};
const data_type ITERATION_TIME = ITER_TIME;
const data_type PRECOND_TIME   = SETUP_TIME;

//**************************************************************************************//
//               Class Declarations and specific typedefs
//**************************************************************************************//

// classes functions.h

class iluplusplus_error;

// classes arrays.h

template<class T> class array;
class sorted_vector;

// classes for orderings.h

class preprocessing_sequence;

// classes for pmwm_declarations.h

struct dist;


// classes for matrix_sparse.h

// general declarations for templates
template<class T> class vector_dense;
template<class T> class vector_sparse_dynamic;
template<class T> class matrix_sparse;
template<class T> class matrix_dense;
template<class T> class matrix_oriented;
class index_list;

// classes for parameters.h

class precond_parameter;
class ILUCP_precond_parameter;
class ILUCDP_precond_parameter;
class iluplusplus_precond_parameter;

// classes for preconditioner.h

// abstract classes for preconditioners
template <class T, class matrix_type, class vector_type> class preconditioner;
template <class T, class matrix_type, class vector_type> class split_preconditioner;
template <class T, class matrix_type, class vector_type> class indirect_split_triangular_preconditioner;
template <class T, class matrix_type, class vector_type> class indirect_split_triangular_multilevel_preconditioner;
template <class T, class matrix_type, class vector_type> class indirect_split_pseudo_triangular_preconditioner;

// specific preconditioners
template <class T, class matrix_type, class vector_type> class NullPreconditioner;
template <class T, class matrix_type, class vector_type> class ILUCPreconditioner;
template <class T, class matrix_type, class vector_type> class ILUTPreconditioner;
template <class T, class matrix_type, class vector_type> class ILUTPPreconditioner;
template <class T, class matrix_type, class vector_type> class ILUCPPreconditioner;
template <class T, class matrix_type, class vector_type> class ILUCDPPreconditioner;
template <class T, class matrix_type, class vector_type> class multilevelILUCDPPreconditioner;

// classes for orderings.h

class preprocessing_sequence;


} // end namespace iluplusplus

#endif
