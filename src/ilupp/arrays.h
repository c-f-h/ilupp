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

namespace iluplusplus {

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
          std::vector<Multimap::iterator> pointers;
          std::vector<bool> used;
      public:
          sorted_vector();
          sorted_vector(Integer max_size);
          sorted_vector(const sorted_vector&) = delete;
          sorted_vector& operator=(const sorted_vector&) = delete;

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
