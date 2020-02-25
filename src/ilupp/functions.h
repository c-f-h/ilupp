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

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <string>

#include "declarations.h"

namespace iluplusplus {

orientation_type other_orientation(orientation_type);
matrix_usage_type other_usage(matrix_usage_type);

template<class VT, class VInt>
VT permute_vec(const VT& vec, const VInt& perm)
{
#ifdef DEBUG
    if (x.dimension() != perm.dimension()) {
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    VT result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        result[i] = vec[perm[i]];
    return result;
}

template <class T>
void fill_identity(std::vector<T>& v)
{
    std::iota(v.begin(), v.end(), 0);
}

template <class T>
void make_identity(std::vector<T>& v, size_t n)
{
    v.resize(n);
    fill_identity(v);
}

float fabs(float x)                              { return std::abs(x); }
double fabs(double x)                            { return std::abs(x); }
long double fabs(long double x)                  { return std::abs(x); }
float fabs(std::complex<float> x)                { return std::abs(x); }
double fabs(std::complex<double> x)              { return std::abs(x); }
long double fabs(std::complex<long double> x)    { return std::abs(x); }

template<class T> inline Real absvalue_squared(T x)     { return std::abs(x * std::conj(x)); }
template<class T> inline T sqr(T x)                     { return x * x; }
inline void fatal_error(bool, const std::string);
inline bool non_fatal_error(bool, const std::string);

std::string booltostring(bool);
Integer RoundRealToInteger(Real);
std::string string(data_type);
std::string cap_string(data_type);
std::string string(preprocessing_type);
std::string long_string(preprocessing_type);

template<class T> bool equal_to_zero(T);
bool equal_to_zero(Real);
template<class T> bool equal(T, T);
bool equal(Real, Real);

#ifndef ILUPLUSPLUS_USES_SPARSPAK
// if SPARSPAK is used, the file f2c.h is included, which provides macros for max and min
using std::min;
using std::max;
#endif

// The class iluplusplus_error                                                                                         *

class iluplusplus_error {
       private:
           error_type error;
       public:
           iluplusplus_error();
           iluplusplus_error(error_type E);
           error_type& set();
           error_type get() const;
           error_type read() const;
           void print() const;
           std::string error_message() const;
    };


} // end namespace iluplusplus

#endif
