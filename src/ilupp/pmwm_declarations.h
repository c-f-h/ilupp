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
 *      Intger dimension() const;               // Returns dimension                                        *
 *      Integer get(Integer i) const;           // Returns i-th element of permutation (constant)           *
 *      Integer set(Integer i);                 // Returns i-th element of permutation (non-constant)       *
 *      void resize_with_constant_value(Integer n, Integer d)                                               *
 *                                              // Resizes field to n and sets all elements to d            *
 *      void init()                             // Initializes permutation to identity                      *
 *                                                                                                          *
 *  vector: field of elements of type T                                                                     *
 *  vector functions:                                                                                       *
 *      Integer dimension() const;              // Returns dimension                                        *
 *      T get(Integer i) const;                 // Returns i-th element of vector (constant)                *
 *      T& set(Integer i);                      // Returns i-th element of vector (non-constant)            *
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
#include "arrays.h"
#include "iohb.h"
#include "functions.h"
#include "function_class.h"
#include "parameters.h"
#include "orderings.h"
#include "sparse.h"


namespace iluplusplus {

//**************************************************************************************//
//               Class Declarations
//**************************************************************************************//

struct dist;
template<class sparse_matrix_class, class permutation> class sapTree;


//**************************************************************************************//
//               Function Declarations
//**************************************************************************************//

template<class sparse_matrix_class>
	void abs_max_row(const sparse_matrix_class& A, array<Real>& max);
// writes maximum absolute values per row in array max 

template<class sparse_matrix_class>
	void transform_and_copy_data(const sparse_matrix_class& A, array<Real>& comp, array<Real>& max);
// calculates transformation to minimization problem and stores new components in field comp

template<class sparse_matrix_class, class permutation, class vector>	
	bool find_pmwm(const sparse_matrix_class& A, permutation& invPerm, permutation& perm, vector& inverse_row_scaling, vector& inverse_col_scaling); 
// finds perfect maximum weighted matching

template<class sparse_matrix_class, class permutation>
	void column_perm(const sparse_matrix_class& A, permutation& col_perm);
// permutes columns in ascending order of non-zero elements


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


//**************************************************************************************//
//               Class sapTree (for storing shortest alternating path trees)
//**************************************************************************************//

template<class sparse_matrix_class, class permutation> class sapTree {
	
	private:
		
		// Attributes
		
		Integer root;	// root node (row)
		Real lsap; 	    // length of shortest augmenting path in the tree
		
		std::priority_queue< dist, std::vector<dist>, std::greater< dist > > cand_nodes; 
			// matched candidate nodes are stored in a priority queue with minimum element at the top
		
		vector_sparse_dynamic<Integer> checked_nodes;
			// contains column nodes whose shortest distances to the root node are known; is set to 1 if node is in B 
		
		array<Integer> row_pointer;
			// pointer array for row nodes, such that (i,mate_row(i)), (row_pointer(i),mate_row(i)) are consecutive edges towards the root 
			
		vector_sparse_dynamic<Real> reduced_dist;
			// stores reduced distances from the root node to a column node (indexed access)
			
		array<Real> cand_weights;
			// stores candidate weights; if the corresponding nodes are getting matched, these elements are copied to field weights
		
		array<Real> weights;
			// stores weights of matched column nodes, i.e. c(i,j), for updating duals
			
		
		// functions for allocating / deallocating memory for fields cand_weights, weights and row_pointer
		void destroy_fields();
		void destroy_resize_fields(Integer size);

				
	public:
		
		// Constructors
		sapTree();
		sapTree(const sapTree<sparse_matrix_class, permutation>& tree);
		~sapTree();
		
		// Operators
		sapTree<sparse_matrix_class, permutation>& operator = (const sapTree<sparse_matrix_class, permutation>& tree);
		
		// Accessing
		Integer get_root() const;
		Real get_lsap() const;
	
		
		// Manipulation

		void resize(Integer dim);  
			// resizes fields cand_weights and weights to dim
		
		void reset(Integer r);     
			// resets the sapTree, i.e. emptying cand_nodes, setting new root node r, and reseting checked_nodes, reduced_dist to zero
		
		void augment(permutation& mate_row, permutation& mate_col, Integer i, Integer j);
			// augments along edge (i,j)
		
		void dual_initialization(const sparse_matrix_class& A, const array<Real>& comp, array<Real>& u, array<Real>& v);
			// initialization of dual variables using heuristic
			
		void matching_initialization(const sparse_matrix_class& A, permutation& mate_row, permutation& mate_col, const array<Real>& comp, const array<Real>& u, const array<Real>& v);
			// determines an initial extreme matching using heuristically initialized dual variables
			
		void dual_update(const sparse_matrix_class& A, const permutation& mate_row, const permutation& mate_col, array<Real>& u, array<Real>& v, Integer isap, Integer jsap);
			// updates dual vectors u and v
		
		void find_sap(const sparse_matrix_class& A, const permutation& mate_col, const array<Real>& comp, const array<Real>& u, const array<Real>& v, Integer& isap, Integer& jsap);
			// procedure for finding a shortest augmenting path starting at the root node; writes edge (isap,jsap) for augmenting
};
		

} // end namespace iluplusplus

#endif /*PMWM_DECLARATIONS_H_*/
