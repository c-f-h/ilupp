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

#ifndef ORDERINGS_H
#define ORDERINGS_H


//#include <math.h>
#include <cmath>
#include <iostream>

#include "declarations.h"
#include "arrays.h"
#include "functions.h"

namespace iluplusplus {

#ifdef ILUPLUSPLUS_USES_SPARSPAK
    int PERMRCM(int nr, int nc, int nnz, int* ia, int* ja, int *p);
    template <class T> void RCM(const matrix_sparse<T>&, index_list&);
    template <class T> void RCM(const matrix_sparse<T>&, index_list&, int, int);
#endif
void setup_symmetric_graph(Integer, Integer*, Integer*, Integer*, Integer*);
template <class int_type> void setup_symmetric_graph(int_type dim, int_type* pointer, int_type* indices, int_type* sym_pointer, int_type* sym_indices, bool shift_index);
void setup_symmetric_graph_without_diag(Integer, Integer*, Integer*, Integer*, Integer*);
template <class T> void setup_symmetric_graph(const matrix_sparse<T> &, Integer* , Integer* );
template <class T> void setup_symmetric_graph_without_diag(const matrix_sparse<T> &, Integer* , Integer* );
#ifdef ILUPLUSPLUS_USES_METIS
    template <class T> void METIS_NODE_ND(const matrix_sparse<T>&, Integer* ,  Integer* );
    template <class T> void METIS_NODE_ND(const matrix_sparse<T>&, index_list&,  index_list& );
    extern "C" {void METIS_NodeND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);} // requires metis library
#endif


/***********************************************************************************

class preprocessing_sequence

************************************************************************************/


class preprocessing_sequence : public std::vector<preprocessing_type> {
    public:
        // compatibility with custom array<>
        preprocessing_type get(Integer k) const { return (*this)[k]; }
        preprocessing_type& set(Integer k)  { return (*this)[k]; }

        void set_test_new();  // only for testing purposes
        void set_none();
        void set_normalize();
        void set_PQ();
        //void set_MAX_WEIGHTED_MATCHING_ORDERING();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG();
        //void set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM();
        void set_SPARSE_FIRST();
        void set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING();
        void set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG();
        void set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
        void set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING();
        //void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_PQ();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_PQ();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM();
#ifdef ILUPLUSPLUS_USES_METIS
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_METIS();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_METIS();
        void set_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ();
        void set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI();
        void set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI();
        void set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI();
#endif  // end PARDISO & METIS available
#ifdef ILUPLUSPLUS_USES_SPARSPAK
        void set_RCM();
        void set_PQ_RCM();
#endif
#ifdef ILUPLUSPLUS_USES_METIS  // requires only METIS
        void set_METIS_NODE_ND_ORDERING();
        void set_PQ_METIS_NODE_ND_ORDERING();
#endif
#ifdef ILUPLUSPLUS_USES_PARDISO
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_PQ();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_PQ();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM();
#endif
#if defined(ILUPLUSPLUS_USES_METIS) && defined(ILUPLUSPLUS_USES_PARDISO)
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI();
        void set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI();
        void set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI();
#endif  // end PARDISO & METIS available
#if defined(ILUPLUSPLUS_USES_METIS) && defined(ILUPLUSPLUS_USES_PARDISO) && defined(ILUPLUSPLUS_USES_SPARSPAK)
        // only for testing purposes of program... makes no sense for preconditioning
        void set_test();
#endif
        void print() const;
        std::string string() const;
        std::string string_with_hyphen() const;
};  // end class preconditioning_sequence


} // end namespace iluplusplus

#endif

