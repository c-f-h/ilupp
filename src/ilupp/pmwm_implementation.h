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


#ifndef PMWM_IMPLEMENTATION_H_
#define PMWM_IMPLEMENTATION_H_

#include "pmwm_declarations.h"

namespace iluplusplus{

//**************************************************************************************//
//               Implemtation of Class sapTree
//**************************************************************************************//

//******************************************************************************************************************************
// Class sapTree: Private Functions                                                                                            *
//******************************************************************************************************************************

template<class sparse_matrix_class, class permuation>
void sapTree<sparse_matrix_class, permuation>::resize_fields(Integer size) {
    row_pointer.resize(size);
    cand_weights.resize(size);
    weights.resize(size);
}


//******************************************************************************************************************************
// Class sapTree: Constructors                                                                                                 *
//******************************************************************************************************************************

// Default constructor
template<class sparse_matrix_class, class permutation>
	sapTree<sparse_matrix_class, permutation>::sapTree() : cand_nodes(), checked_nodes(), reduced_dist() {
		root = -1;
		lsap = -1;
	}

//******************************************************************************************************************************
// Class sapTree: Accessing                                                                                                    *
//******************************************************************************************************************************

template<class sparse_matrix_class, class permutation>
	Integer sapTree<sparse_matrix_class, permutation>::get_root() const {
	return root;
}

template<class sparse_matrix_class, class permutation>
	Real sapTree<sparse_matrix_class, permutation>::get_lsap() const {
	return lsap;
}

//******************************************************************************************************************************
// Class sapTree: Manipulation                                                                                                 *
//******************************************************************************************************************************

template<class sparse_matrix_class, class permutation>
void sapTree<sparse_matrix_class, permutation>::resize(Integer dim) {
    checked_nodes.resize(dim);
    reduced_dist.resize(dim);
    resize_fields(dim);
}

template<class sparse_matrix_class, class permutation>
	void sapTree<sparse_matrix_class, permutation>::reset(Integer r) {
		root = r;
		lsap = -1;
		std::priority_queue< dist, std::vector<dist>, std::greater< dist > > empty;
		cand_nodes = empty;
		checked_nodes.zero_reset();
		reduced_dist.zero_reset();
	}


template<class sparse_matrix_class, class permutation>
void sapTree<sparse_matrix_class, permutation>::augment(permutation& mate_row, permutation& mate_col, Integer i, Integer j) {

#ifdef VERBOSE
    std::cout << "augmenting along edge " << i << " " << j << "\n";
#endif /*VERBOSE*/

    Integer k;
    mate_col.set(j) = i;
    while (i != root) {
        weights[j] = cand_weights[j];
        k = mate_row.get(i);
        mate_row.set(i) = j;
        j = k;
        i = row_pointer[i];
        mate_col.set(j) = i;
    }
    mate_row.set(root) = j;
    weights[j] = cand_weights[j];

#ifdef VERBOSE
    std::cout << "New mate_row: " << std::endl << mate_row << std::endl;
    std::cout << "New mate_col: " << std::endl << mate_col << "\n" << std::endl;
#endif /*VERBOSE*/
}

template<class sparse_matrix_class, class permutation>
void sapTree<sparse_matrix_class, permutation>::dual_initialization(const sparse_matrix_class& A,  const array<Real>& comp, array<Real>& u, array<Real>& v) {
    Integer i, row, col;
    Integer dim = A.dimension();
    Integer nz = A.non_zeroes();

    for (i = 0; i < dim; i++) {
        v[i] = -1; u[i] = -1;
    }

    // find minimum weights per column
    for (i = 0; i < nz; i++) {
        col = A.read_index(i);
        if (v[col] > comp[i] || v[col] == -1) v[col] = comp[i];
    }

    // find minimum per row
    for (row = 0; row < dim; row++) {
        for (i = A.read_pointer(row); i < A.read_pointer(row+1); i++) {
            if ( (u[row] > comp[i] - v[A.read_index(i)] ) || u[row] == -1 ) u[row] = comp[i] - v[A.read_index(i)];
        }
    }

#ifdef VERBOSE
    std::cout << "Duals initialized: " << std::endl;
    for (Integer k = 0; k < A.dimension(); k++) std::cout<<"u("<<k<<") = "<<u[k]<<"\tv("<<k<<") = "<<v[k]<<std::endl;
#endif /*VERBOSE*/
}

template<class sparse_matrix_class, class permutation>
void sapTree<sparse_matrix_class, permutation>::matching_initialization(const sparse_matrix_class& A, permutation& mate_row, permutation& mate_col, const array<Real>& comp, const array<Real>& u, const array<Real>& v) {
    Integer i, row, col;
    Integer dim = A.dimension();

    // scans in the neighborhood of unmatched row nodes for an unmatched column node whose reduced weight is zero
    for (row = 0; row < dim; row++) {
        for (i = A.read_pointer(row); i < A.read_pointer(row+1); i++) {
            col = A.read_index(i);
            if ( mate_col.get(col) == -1 && (comp[i] - u[row] - v[col] == 0) ) {
                mate_row.set(row) = col; mate_col.set(col) = row;
                weights[col] = comp[i];
                break;
            }
        }
    }

    // scans in the neighborhood of unmatched row nodes for a matched column node whose reduced weight is zero;
    // sets set row_new = mate_col(j) and continues the search in the neighborhood of rode_new like above
    Integer row_new, col_new;
    for (row = 0; row < dim; row++) {
        if (mate_row.get(row) == -1) {
            for (i = A.read_pointer(row); i < A.read_pointer(row+1); i++) {
                col = A.read_index(i);
                if ( mate_col.get(col) != -1 && (comp[i] - u[row] - v[col] == 0) ) {
                    row_new = mate_col.get(col);
                    for (Integer j = A.read_pointer(row_new); j < A.read_pointer(row_new+1); j++) {
                        col_new = A.read_index(j);
                        if ( mate_col.get(col_new) == -1 && (comp[j] - u[row_new] - v[col_new] == 0) ) {
                            mate_row.set(row) = col; mate_col.set(col) = row;
                            mate_row.set(row_new) = col_new; mate_col.set(col_new) = row_new;
                            weights[col_new] = comp[j]; weights[col] = comp[i];
                            break;
                        }
                    }
                }
                if (mate_row.get(row) != -1) break;
            }
        }
    }

#ifdef VERBOSE
    std::cout << "Matching initialized: " << std::endl;
    std::cout << "mate_row: " << std::endl << mate_row << std::endl;
    std::cout << "mate_col: " << std::endl << mate_col << std::endl;
#endif /*VERBOSE*/
}

template<class sparse_matrix_class, class permutation>
void sapTree<sparse_matrix_class, permutation>::dual_update(const sparse_matrix_class& A, const permutation& mate_row, const permutation& mate_col, array<Real>& u, array<Real>& v, Integer isap, Integer jsap) {
    Integer col;
    Integer index_checked_nodes;

    // updating elements of v
    for (index_checked_nodes = 0; index_checked_nodes < checked_nodes.non_zeroes(); index_checked_nodes++) {
        col = checked_nodes.get_pointer(index_checked_nodes);
        v[col] = v[col] + reduced_dist.read(col) - lsap;
    }

    // updating elements of u (in shortest augmenting path)
    Integer i = isap;
    Integer j = jsap;
    while (i != root) {
        checked_nodes[j] = 0;
        u[i] = weights[j] - v[j];
        i = row_pointer[i]; j = mate_row.get(i);
    }
    checked_nodes[j] = 0; u[root] = weights[j] - v[j];

    // updating elements of u (in checked_nodes)
    for (index_checked_nodes = 0; index_checked_nodes < checked_nodes.non_zeroes(); index_checked_nodes++) {
        col = checked_nodes.get_pointer(index_checked_nodes);
        u[mate_col.get(col)] = weights[col] - v[col];
    }

#ifdef VERBOSE
    std::cout << "Duals updated: " << std::endl;
    for (Integer k = 0; k < A.dimension(); k++) std::cout<<"u("<<k<<") = "<<u[k]<<"\tv("<<k<<") = "<<v[k]<<std::endl;
#endif /*VERBOSE*/
}

template<class sparse_matrix_class, class permutation>
void sapTree<sparse_matrix_class, permutation>::find_sap(const sparse_matrix_class& A, const permutation& mate_col,  const array<Real>& comp, const array<Real>& u, const array<Real>& v, Integer& isap, Integer& jsap) {
    Real dnew;
    Real weight;
    dist temp_dist;     // temporary distance, for pushing struct dist into heap cand_nodes
    dist min_dist;      // top dist elememt of cand_nodes

    Integer i = root;   // actual treated row node
    Integer j;          // actual treated column node

    Integer j_min;      // matched column node with minimum distance to the root node
    isap = -1;
    jsap = -1;          // (isap,jsap) : last edge of shortest augmenting path
    Real lsp = 0;       // length of shortest path from the root node to any node in cand_nodes

    Integer index_data; // indes of the actual weight in the field data;

    while(true) {

        for (index_data = A.read_pointer(i); index_data < A.read_pointer(i+1); index_data++) {

            j = A.read_index(index_data);
            // if col node j is not in checked_nodes...
            if (checked_nodes.get_occupancy(j) == -1) {

                weight = comp[index_data];
                dnew = lsp + weight - u[i] - v[j];

#ifdef VERBOSE
                std::cout << "u(" << i << ") = " << u[i] << std::endl;
                std::cout << "v(" << j << ") = " << v[j] << std::endl;
                std::cout << "dnew = " << dnew << std::endl;
#endif /*VERBOSE*/

                // if a new shortest alternating path is found...
                if (lsap == -1 || dnew < lsap) {

                    // if node j is unmatched, lsap is shortest augmenting path so far
                    if ( mate_col.get(j) == -1 ) {
                        lsap = dnew;
                        cand_weights[j] = weight;

                        // stores isap, jsap for augmenting
                        jsap = j;
                        isap = i;
                    }

                    // if node j is matched, push distance into the heap cand_nodes
                    else if (reduced_dist.get_occupancy(j) == -1 || dnew < reduced_dist.read(j)) {
                        reduced_dist[j] = dnew;
                        row_pointer[mate_col.get(j)] = i;
                        temp_dist.index = j; temp_dist.value = dnew; temp_dist.weight = weight;

                        cand_nodes.push(temp_dist);
                    }
                }
            }
        }

        if (cand_nodes.empty()) break;

        do {
            min_dist = cand_nodes.top();
            j_min = min_dist.index;
            cand_nodes.pop();
        } while(checked_nodes.get_occupancy(j_min) != -1 && !cand_nodes.empty());

        if (cand_nodes.empty() && checked_nodes.get_occupancy(j_min) != -1) break;

        lsp = min_dist.value;

        if (lsap != -1 && lsap <= lsp) break;
        cand_weights[j_min] = min_dist.weight;
        checked_nodes[j_min] = 1;
        i = mate_col.get(j_min);
    }
}

//**************************************************************************************//
//               Function Implementations
//**************************************************************************************//

// writes maximum absolute values per row in array max
template<class sparse_matrix_class>
void abs_max_row(const sparse_matrix_class& A, array<Real>& max) {
    for (Integer row = 0; row < A.dimension(); row++) {
        for (Integer j = A.read_pointer(row); j < A.read_pointer(row+1); j++) {
            if ( max.get(row) < fabs(A.read_data(j)) ) max.set(row) = fabs(A.read_data(j));
        }
    }
}

// calculates transformation to minimization problem and stores new components in field comp
template<class sparse_matrix_class>
void transform_and_copy_data(const sparse_matrix_class& A, array<Real>& comp, array<Real>& max) {
    abs_max_row<sparse_matrix_class>(A, max);
    Integer row = 0;
    for (Integer i = 0; i < A.non_zeroes(); i++) {
        while ( i >= A.read_pointer(row+1) ) row++;
        comp[i] = std::log( max.get(row) / fabs(A.read_data(i)) );
    }
}

template<class sparse_matrix_class, class permutation, class vector>
bool find_pmwm(const sparse_matrix_class& A, permutation& mate_row, permutation& mate_col, vector& inverse_row_scaling, vector& inverse_col_scaling) {

#ifdef PMWM_TIME
    clock_t init_fields_start, init_fields_end, init_match_start, init_match_end, reset_start, reset_end, sap_start, sap_end, augment_start, augment_end, dual_start, dual_end, scale_start, scale_end;
    double init_fields_total, init_match_total, reset_total, sap_total, augment_total, dual_total, scale_total;
    scale_total = init_fields_total = init_match_total = reset_total = sap_total = augment_total = dual_total = 0;
#endif /*PMWM_TIME*/

    Integer jsap;
    Integer isap;
    array<Real> u;              // row duals
    array<Real> v;              // col duals
    array<Real> comp;           // transformed components of matrix A
    sapTree<sparse_matrix_class, permutation> tree;
    array<Real> abs_max_per_row;
    Integer dim = A.dimension();


#ifdef PMWM_TIME
    init_fields_start = clock();
#endif /*PMWM_TIME*/

    // initializes fields
    u.erase_resize_data_field(dim);
    v.erase_resize_data_field(dim);
    comp.erase_resize_data_field(A.non_zeroes());
    mate_row.resize_with_constant_value(dim, -1);
    mate_col.resize_with_constant_value(dim, -1);
    inverse_row_scaling.resize(dim,0);
    inverse_col_scaling.resize(dim,0);
    tree.resize(dim);

    // calculates transformed weights
    abs_max_per_row.resize(dim, 0);
    transform_and_copy_data<sparse_matrix_class> (A, comp, abs_max_per_row);

#ifdef PMWM_TIME
    init_fields_end = clock();
    init_fields_total = ((double)init_fields_end - (double)init_fields_start)/(double)CLOCKS_PER_SEC;
#endif /*PMWM_TIME*/

#ifdef PMWM_TIME
    init_match_start = clock();
#endif /*PMWM_TIME*/

    // initializes matching and duals using heuristic
    tree.dual_initialization(A, comp, u, v);
    tree.matching_initialization(A, mate_row, mate_col, comp, u, v);

#ifdef PMWM_TIME
    init_match_end = clock();
    init_match_total = ((double)init_match_end-(double)init_match_start)/(double)CLOCKS_PER_SEC;
#endif /*PMWM_TIME*/


    for (Integer r = 0; r < dim; r++) {
        if (mate_row.get(r) == -1) {

#ifdef PMWM_TIME
            reset_start = clock();
#endif /*PMWM_TIME*/

            tree.reset(r);	// resets tree

#ifdef PMWM_TIME
            reset_end = clock();
            reset_total += ((double)reset_end-(double)reset_start)/(double)CLOCKS_PER_SEC;
#endif /*PMWM_TIME*/


#ifdef PMWM_TIME
            sap_start = clock();
#endif /*PMWM_TIME*/

            tree.find_sap(A, mate_col, comp, u, v, isap, jsap);	// finds shortest augmenting path starting at root node r

#ifdef PMWM_TIME
            sap_end = clock();
            sap_total += ((double)sap_end-(double)sap_start)/(double)CLOCKS_PER_SEC;
#endif /*PMWM_TIME*/

            // if no shortest augmenting path is found, return false
            if (tree.get_lsap() == -1 || jsap == -1) {
#ifdef DEBUG
                std::cerr << "find_pmwm: No perfect matching found." << std::endl;
#endif
                for (Integer s = 0; s < dim; s++) {
                    inverse_row_scaling.set(s) = (Real) 1;
                    inverse_col_scaling.set(s) = (Real) 1;
                    mate_row.init();
                    mate_col.init();
                }
                return false;
            }



#ifdef PMWM_TIME
            augment_start = clock();
#endif /*PMWM_TIME*/

            tree.augment(mate_row, mate_col, isap, jsap);	// augments along (isap,jsap)

#ifdef PMWM_TIME
            augment_end = clock();
            augment_total += ((double)augment_end-(double)augment_start)/(double)CLOCKS_PER_SEC;
#endif /*PMWM_TIME*/


#ifdef PMWM_TIME
            dual_start = clock();
#endif /*PMWM_TIME*/

            tree.dual_update(A, mate_row, mate_col, u, v, isap, jsap);	// updates duals

#ifdef PMWM_TIME
            dual_end = clock();
            dual_total += ((double)dual_end-(double)dual_start)/(double)CLOCKS_PER_SEC;
#endif /*PMWM_TIME*/
        }
    }

#ifdef PMWM_TIME
    scale_start = clock();
#endif /*PMWM_TIME*/

    // calculates vectors for scaling
    for (Integer s = 0; s < dim; s++) {
        inverse_row_scaling.set(s) = abs_max_per_row.get(s) / std::exp(u[s]);
        inverse_col_scaling.set(s) = std::exp(-v[s]);
    }

#ifdef PMWM_TIME
    scale_end = clock();
    scale_total = ((double)scale_end-(double)scale_start)/(double)CLOCKS_PER_SEC;
#endif /*PMWM_TIME*/

    // deallocates reserved memory

#ifdef PMWM_TIME
    std::cout << "*********** RUNNING TIMES ****************" << std::endl;
    std::cout << "Running time for initializing fields: " << init_fields_total << std::endl;
    std::cout << "Running time for initializing matching using heuristic: " << init_match_total << std::endl;
    std::cout << "Total running time for reseting tree: " << reset_total << std::endl;
    std::cout << "Total running time for finding shortest augmenting paths: " << sap_total << std::endl;
    std::cout << "Total running time for augmenting: " << augment_total << std::endl;
    std::cout << "Total running time for updating duals: " << dual_total << std::endl;
    std::cout << "Running time for calculating scaling vectors: " << scale_total << std::endl;
    std::cout << "******************************************\n" << std::endl;
#endif /*PWMW_TIME*/

    return true;
}

template<class sparse_matrix_class, class permutation>
void column_perm(const sparse_matrix_class& A, permutation& col_perm) {
    Integer dim;
    vector_dense<Integer> nnz_per_col;
    index_list indices;

    dim = A.dimension();
    nnz_per_col.resize(dim,0);
    indices.resize(dim);
    col_perm.resize_with_constant_value(dim,-1);

    for (Integer j = 0; j < A.non_zeroes(); j++)
        nnz_per_col.set(A.read_index(j))++;

    nnz_per_col.quicksort(indices,0,dim-1);

    if (nnz_per_col.get(0) != 0) {
        for (Integer j = 0; j < dim; j++) {
            col_perm.set(j) = indices.get(j);
        }
    }
    else col_perm.init();
}

} // end namespace iluplusplus
#endif /*PMWM_IMPLEMENTATION_H_*/
