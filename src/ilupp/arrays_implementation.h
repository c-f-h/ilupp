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

#ifndef ARRAYS_IMPLEMENTATION_H
#define ARRAYS_IMPLEMENTATION_H


#include <iostream>
#include <map>

#include "functions.h"
#include "arrays.h"

#include "functions_implementation.h"


namespace iluplusplus {


//************************************************************************************************************************
//                                                                                                                       *
//         The implementation of the class sorted_vector                                                                 *
//                                                                                                                       *
//************************************************************************************************************************

sorted_vector::sorted_vector(){}

sorted_vector::sorted_vector(Integer max_size){
     pointers.resize(max_size);
     used.resize(max_size,true);
     list.clear();
     Integer k;
     for(k=0;k<max_size;k++)
         pointers[k]=list.insert(Multimap::value_type(0.0,k));
}


void sorted_vector::resize(Integer max_size){
     pointers.resize(max_size);
     used.resize(max_size,true);
     list.clear();
     Integer k;
     for(k=0;k<max_size;k++)
         pointers[k] = list.insert(Multimap::value_type(0.0,k));
}


Real sorted_vector::read(Integer j) const {
     if(used[j])
         return pointers[j]->first;
     else {
         std::cerr<<"sorted_vector::read: entry with given index is not being used. Returning 0.0"<<std::endl;
         return 0.0;
     }
}

void sorted_vector::insert(Integer pos, Real val){
     if(used[pos])
         list.erase(pointers[pos]);
     pointers[pos] = list.insert(Multimap::value_type(val,pos));
     used[pos]=true;
}

void sorted_vector::remove(Integer k){
    if(used[k])
        list.erase(pointers[k]);
    used[k]=false;
}

Integer sorted_vector::index_min() const {
         Multimap::const_iterator p; 
         p = list.begin();
         if(!list.empty()) return p->second;
         else {
            std::cerr<<"sorted_vector::index_min: list is empty. Returning -1"<<std::endl;
            return -1;
         }
}

Integer sorted_vector::index_max() const {
         Multimap::const_iterator p; 
         p = list.end();
         if(!list.empty()) return (--p)->second;
         else {
            std::cerr<<"sorted_vector::index_max: list is empty. Returning -1"<<std::endl;
            return -1;
         }
}

Real sorted_vector::read_min() const {
    Multimap::const_iterator p; 
    p = list.begin();
    if(!list.empty()){ 
        return p->first;
    } else {
        std::cerr<<"sorted_vector::read_min: list is empty. Returning 0"<<std::endl;
        return 0.0;
    }
}

Real sorted_vector::read_max() const {
    Multimap::const_iterator p; 
    p = list.end();
    if(!list.empty()){ 
        return (--p)->first;
    } else {
        std::cerr<<"sorted_vector::read_max: list is empty. Returning 0"<<std::endl;
        return 0.0;
    }
}


void sorted_vector::remove_min(){
    Multimap::iterator p; 
    p = list.begin();
    if(!list.empty()){ 
        used[p->second]=false;
        list.erase(p);
    } else {
        std::cerr<<"sorted_vector::remove_min: list is empty."<<std::endl;
    }
}

void sorted_vector::remove_max(){
    Multimap::iterator p = list.end();
    if(!list.empty()){ 
        used[--p->second]=false;
        list.erase(p);
    } else {
        std::cerr<<"sorted_vector::remove_max: list is empty. Returning 0"<<std::endl;
    }
}

void sorted_vector::add(Integer pos, Real val){
    insert(pos,val+read(pos));
}


void sorted_vector::print() const{
    for(size_t i=0;i<pointers.size();i++)
        if(used[i]) std::cout<<pointers[i]->first<<std::endl;
        else std::cout<<"undefined"<<std::endl;
}



void sorted_vector::print_list() const {
    Multimap::const_iterator p;
    for(p=list.begin();p!=list.end();p++)
        std::cout<<p->second<<": "<<p->first<<std::endl;
}

Real sorted_vector::memory() const {
    // no way to get memory used of vector<bool>
    return (Real)  used.size() * (sizeof(Real)+sizeof(Integer)) /* + used.memory() */ + memsize(pointers);
}

}

#endif


