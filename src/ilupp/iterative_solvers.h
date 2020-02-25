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


#ifndef ITERATIVE_SOLVERS_H
#define ITERATIVE_SOLVERS_H

#include <ctime>

#include "declarations.h"
#include "preconditioner.h"


namespace iluplusplus {

template <class matrix_type, class vector_type> void normalize_matrix(matrix_type&);
template <class matrix_type, class vector_type> void normalize_rows(matrix_type&);
template <class matrix_type, class vector_type> void normalize_columns(matrix_type&);
template <class matrix_type, class vector_type> void normalize_equations(matrix_type&, vector_type&);
template <class matrix_type, class vector_type> void normalize_equations(matrix_type&, matrix_type&, vector_type&);
template <class matrix_type, class vector_type> void normalize(matrix_type&, vector_type&, vector_type&);

template <class T, class matrix_type, class vector_type> bool cg(const preconditioner<T,matrix_type,vector_type>&,
             preconditioner_application1_type, const matrix_type&, const vector_type&,vector_type&,
             Integer, Integer&, Real&, Real&, bool);
template <class T, class matrix_type, class vector_type> bool cgnr(const preconditioner<T,matrix_type,vector_type>&,
             preconditioner_application1_type, const matrix_type&, const vector_type&, vector_type&,
             Integer, Integer&, Real&, Real&, bool);
template <class T, class matrix_type, class vector_type> bool cgne(const preconditioner<T,matrix_type,vector_type>&,
             preconditioner_application1_type, const matrix_type&, const vector_type&, vector_type&,
             Integer, Integer&, Real&, Real&, bool);
template <class T, class matrix_type, class vector_type> bool bicgstab(const preconditioner<T,matrix_type,vector_type>&,
             preconditioner_application1_type pa1, const matrix_type&,
             const vector_type&,vector_type&, Integer, Integer&,Real&, Real&, bool);
template<class T, class matrix_type, class vector_type> bool richardson(const preconditioner<T,matrix_type,vector_type>&,
             preconditioner_application1_type, const matrix_type&, const vector_type&, vector_type&,
             Integer, Integer&, Real&, Real&, bool);
template <class T, class matrix_type, class vector_type> bool cgs(const preconditioner<T,matrix_type,vector_type>&,
             preconditioner_application1_type, const matrix_type&, const vector_type&, vector_type&,
             Integer, Integer&, Real&, Real&, bool);


template <class Real> void GeneratePlaneRotation(Real&, Real&, Real&, Real&);
template <class Real> void ApplyPlaneRotation(Real&, Real&, Real&, Real&);

template <class matrix_type, class vector_type>
void Update(vector_type&, Integer, matrix_type&, const vector_type&, std::vector<vector_type>&);

//template <class Real> inline Real abs(Real);
template <class T, class matrix_type, class vector_type> bool gmres(const preconditioner<T,matrix_type,vector_type>&,
             preconditioner_application1_type, const matrix_type&, const vector_type&, vector_type&,
             Integer&, Integer, Integer&, Real&, Real&, bool);

} // end namespace iluplusplus 


#endif

