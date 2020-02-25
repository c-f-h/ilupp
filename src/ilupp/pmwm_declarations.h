/***************************************************************************
 *   Copyright (C) 2007 by Jan Mayer                                       *
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


/******************************************
 *    Author: Chandramowli Subramanian    *
 *    Begin:  December 2006               *
 *****************************************/


/************************************************************************************************************
 *                                                                                                          *
 * The function find_pmwm determines a perfect matching with maximum weight in a bipartite graph. This      *
 * function is based on the algorithm presented in the paper by Duff and Koster "On algorithms for          *
 * permuting large entries to the diagonal of a sparse matrix" (SIAM  Journal on Matrix Analysis and        *
 * Applications, 22(4):973�996, 2001). This matching defines a permutation for a matrix such that the       *
 * product of the diagonal entries of the permuted matrix is maximum in absolute value. Additionally this   *
 * function determines dual variables for scaling this permuted matrix to an I-matrix.                      *
 *                                                                                                          *
 ***********************************************************************************************************/


/************************************************************************************************************
 *                                                                                                          *
 *                                                                                                          *
 * REQUIREMENTS:                                                                                            *
 *                                                                                                          *
 *  sparse_matrix_class: stores a sparse matrix (or bipartite graph) in Compressed Sparse Row Format with   *
 *  fields data (of type T), pointer and index (of type Integer)                                            *
 *                                                                                                          *
 *  sparse_matrix_class functions:                                                                          *
 *      Integer dimension() const;              // Returns dimension                                        *
 *      Integer non_zeroes() const;             // Returns number of non-zero elements                      *
 *      T read_data(Integer j) const;           // Returns j-th element of the field data                   *
 *      Integer read_pointer(Integer j) const;	// Returns j-th element of the field pointer                *
 *      Integer read_index(Integer j) const;    // Returns j-th element of the field index                  *
 *                                                                                                          *
 *  permutation: field of Integers storing the permutation                                                  *
 *  permutation functions:                                                                                  *
 *      operator[](Integer i)                   // Access i-th element
 *      void resize_with_constant_value(Integer n, Integer d)                                               *
 *                                              // Resizes field to n and sets all elements to d            *
 *      void init()                             // Initializes permutation to identity                      *
 *                                                                                                          *
 *  vector: field of elements of type T                                                                     *
 *  vector functions:                                                                                       *
 *      Integer dimension() const;              // Returns dimension                                        *
 *      operator[](Integer i)                   // Access i-th element
 *      void resize(Integer n, T d);            // Resizes field to n and sets all elements to d            *
 *                                                                                                          *
 *                                                                                                          *
 ***********************************************************************************************************/



#ifndef PMWM_DECLARATIONS_H_
#define PMWM_DECLARATIONS_H_

#include <iostream>
#include <cmath>
#include <queue>
#include <algorithm>
#include <time.h>

#include "declarations.h"
#include "functions.h"
#include "function_class.h"
#include "parameters.h"
#include "orderings.h"
#include "sparse.h"


namespace iluplusplus {

//**************************************************************************************//
//               Function Declarations
//**************************************************************************************//

// finds perfect maximum weighted matching
template<class sparse_matrix_class, class permutation, class vector>
bool find_pmwm(const sparse_matrix_class& A, permutation& invPerm, permutation& perm, vector& inverse_row_scaling, vector& inverse_col_scaling);

// permutes columns in ascending order of non-zero elements
template<class sparse_matrix_class>
void column_perm(const sparse_matrix_class& A, std::vector<Integer>& col_perm);


//**************************************************************************************//
//               Structure dist
//**************************************************************************************//

// This structure stores the reduced distances from the root node to a column node "index" in 
// "value" and its unreduced weight c(i,j) with respect to the actual matching in "weight".
struct dist {
	Integer index;
	Real value;
	Real weight;
	bool operator<(const dist d) const {
		return (value<d.value);
	}
	bool operator>(const dist d) const {
		return (value>d.value);
	}
};


} // end namespace iluplusplus

#endif /*PMWM_DECLARATIONS_H_*/
