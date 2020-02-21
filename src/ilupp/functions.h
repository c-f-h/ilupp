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
template<class T> inline void switchnumbers(T&, T&);
template<class T> inline void interchange(T&, T&);

template<class T> inline T conj(T);                             //conjugate; adjust for complex numbers
template<> inline float conj(float);
template<> inline double conj(double);
template<> inline long double conj(long double);
template<> inline std::complex<float>  conj(std::complex<float>);
template<> inline std::complex<double> conj(std::complex<double>);
template<> inline std::complex<long double> conj(std::complex<long double>);

template<class T> inline Real real(T);                             //conjugate; adjust for complex numbers
template<> inline Real real(float);
template<> inline Real real(double);
template<> inline Real real(long double);
template<> inline Real real(std::complex<float>);
template<> inline Real real(std::complex<double>);
template<> inline Real real(std::complex<long double>);

inline float fabs(std::complex<float>);
inline double fabs(std::complex<double>);
inline long double fabs(std::complex<long double>);

template<class T> inline Real absvalue_squared(T);
template<class T> inline T sqr(T);
inline void fatal_error(bool, const std::string);
inline bool non_fatal_error(bool, const std::string);
Integer bin(Integer,Integer);
std::string replace_underscore_with_backslash_underscore(std::string);
std::string replace_underscore_with_double_backslash_underscore(std::string oldstring);
std::string booltostring(bool);
Integer RoundRealToInteger(Real);
std::string integertostring(Integer);
std::string integertostring_with_spaces(Integer k);
std::string string(data_type);
std::string cap_string(data_type);
std::string string(preprocessing_type);
std::string long_string(preprocessing_type);
template<class T> bool equal_to_zero(T);
bool equal_to_zero(Real);
template<class T> bool equal(T, T);
bool equal(Real, Real);
#ifndef ILUPLUSPLUS_USES_SPARSPAK    // if SPARSPAK is used, the file f2c.h is included, which provides macros for max and min
    template<class T> T max(T, T);
    Integer max(Integer, Integer);
    Real max(Real, Real);
    template<class T> T min(T, T);
    Integer min(Integer, Integer);
    Real min(Real, Real);
#endif

// The class iluplusplus_error                                                                                         *

class iluplusplus_error {
       private:
           error_type error;
       public:
           iluplusplus_error();
           iluplusplus_error(error_type E);
           ~iluplusplus_error();
           iluplusplus_error(const iluplusplus_error& E);
           iluplusplus_error& operator = (const iluplusplus_error& E);
           error_type& set();
           error_type get() const;
           error_type read() const;
           void print() const;
           std::string error_message() const;
    };


} // end namespace iluplusplus

#endif
