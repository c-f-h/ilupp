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


#ifndef ARRAYS_H
#define ARRAYS_H


#include <new>
#include <iostream>
#include <map>

#include "declarations.h"
#include "functions.h"


namespace iluplusplus {


//************************************************************************************************************************
//                                                                                                                       *
//         The class array                                                                                               *
//                                                                                                                       *
//************************************************************************************************************************



template<class T> class array
   {
       private:
          std::vector<T> data;
       public:
        // constructors & destructors
          array();
          array(Integer m);
          array(Integer m, T t);
       // Functions, Manipulation, Information
          Integer dimension() const;
          void print_info() const;
      // Accessing elements
          T& operator[](Integer j);
          const T& operator[](Integer j) const;
       // Assignment
          void set_all(T d);
          void erase_resize_data_field(Integer newsize);  // resizes only if newsize is different
          void resize(Integer newsize);
          void resize(Integer newsize, T d);
          void destroy_resize_data_field(Integer newsize); // destroy and then resizes, even if newsize is the same.
          void enlarge_dim_keep_data(Integer newdim);
          void destroy();
          Real memory() const;
   };


template<class T> std::istream& operator >> (std::istream& is, array<T> &x);
template<class T> std::ostream& operator << (std::ostream& os, const array<T> &x);


//************************************************************************************************************************
//                                                                                                                       *
//         The sorted vector                                                                                             *
//                                                                                                                       *
//************************************************************************************************************************
//
// list: a vector sorted by natural ordering of reals
// pointers: iterators for accessing every element directly
// used: indicates if a particular index is being used

 class sorted_vector
  {
      private:
          Multimap list;
          array<Multimap::iterator> pointers;
          std::vector<bool> used;
      public:
          sorted_vector();
          sorted_vector(Integer max_size);
          void resize(Integer max_size);
          Real read(Integer j) const;
          void insert(Integer pos, Real val);
          void add(Integer pos, Real val);
          Integer index_min() const;
          Integer index_max() const;
          Real read_min() const;
          Real read_max() const;
          void remove_min();
          void remove_max();
          void remove(Integer k);
          void print() const;
          void print_list() const;
          Real memory() const;
  };

}

#endif
