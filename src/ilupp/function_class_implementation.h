/***************************************************************************
 *   Copyright (C) 2005 by Jan Mayer                                       *
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


#ifndef FUNCTION_CLASS_IMPLEMENTATION_H
#define FUNCTION_CLASS_IMPLEMENTATION_H


#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <iostream>

#include "functions.h"
#include "function_class.h"

#include "functions_implementation.h"

namespace iluplusplus {


template<class field> function<field>::~function(){}

/***********************************************************************************************/

template<class field>  constant_function<field>::constant_function(){value = 0.0;}

template<class field> constant_function<field>::constant_function(field y){value = y;}

template<class field> field constant_function<field>::operator()(field x) const {return value;}

template<class field> constant_function<field>::constant_function(const constant_function &f){value=f.value;}

template<class field> constant_function<field>& constant_function<field>::operator = (const constant_function &f){
    value=f.value;
    return *this;
  }
template<class field> constant_function<field>::~constant_function(){}

template<class field> field constant_function<field>::get_value() const {return value;}


/***********************************************************************************************/


template<class field> linear_function<field>::linear_function(){slope = 1.0; y_intercept = 0.0;}

template<class field> linear_function<field>::linear_function(field m, field b){slope = m; y_intercept = b;}

template<class field> linear_function<field>::linear_function(field x1, field y1, field x2, field y2){
                slope = (y2-y1)/(x2-x1);
                y_intercept = y1-slope*x1;
            }

template<class field> field linear_function<field>::operator()(field x) const {return slope*x+y_intercept;}

template<class field> linear_function<field>::linear_function(const linear_function &f){
                slope=f.slope;
                y_intercept=f.y_intercept;
            }

template<class field> linear_function<field>& linear_function<field>::operator = (const linear_function &f){
                slope=f.slope;
                y_intercept=f.y_intercept;
                return *this;
            }

template<class field> linear_function<field>::~linear_function(){}

template<class field> field linear_function<field>::get_slope() const {return slope;}

template<class field> field linear_function<field>::get_y_intercept() const {return y_intercept;}

template<class field> void linear_function<field>::set_slope(field m)  {slope = m;}

template<class field> void linear_function<field>::set_y_intercept(field b) {y_intercept = b;}

template<class field> void linear_function<field>::set(field m, field b) {slope=m; y_intercept=b;}

template<class field> void linear_function<field>::set(field x1, field y1, field x2, field y2){
                if(x1==x2){
                    std::cerr<<"linear_function::set: WARNING: x-values are equal. Setting constant function whose value is average of y-values."<<std::endl;
                    slope = 0.0;
                    y_intercept = 0.5*(y1+y2);
                } else {
                    slope = (y2-y1)/(x2-x1);
                    y_intercept = y1-slope*x1;
                }
            }

} // end namespace iluplusplus

#endif

