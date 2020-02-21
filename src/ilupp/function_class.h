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

#ifndef FUNCTION_CLASS_H
#define FUNCTION_CLASS_H


#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <iostream>
#include "declarations.h"
#include "functions.h"

namespace iluplusplus {


template<class field> class function
    {
        public:
            virtual field operator()(field x) const = 0;
            virtual ~function();
    };




template<class field> class constant_function : public function <field>
    {
        private:
            field value;
        public:
            constant_function();
            constant_function(field y);
            virtual field operator()(field x) const;
            constant_function(const constant_function &f);
            constant_function& operator = (const constant_function &f);
            virtual ~constant_function();
            field get_value() const;
    };

template<class field> class linear_function : public function <field>
    {
        private:
            field slope;
            field y_intercept;
        public:
            linear_function();
            linear_function(field m, field b);
            linear_function(field x1, field y1, field x2, field y2);
            virtual field operator()(field x) const;
            linear_function(const linear_function &f);
            linear_function& operator = (const linear_function &f);
            virtual ~linear_function();
            field get_slope() const;
            field get_y_intercept() const;
            void set_slope(field m);
            void set_y_intercept(field b);
            void set(field m, field b);
            void set(field x1, field y1, field x2, field y2);
    };


} // end namespace iluplusplus


#endif
