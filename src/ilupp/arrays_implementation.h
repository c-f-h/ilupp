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
     pointers.erase_resize_data_field(max_size);
     used.resize(max_size,true);
     list.clear();
     Integer k;
     for(k=0;k<max_size;k++) pointers[k]=list.insert(Multimap::value_type(0.0,k));
}


void sorted_vector::resize(Integer max_size){
     pointers.erase_resize_data_field(max_size);
     used.resize(max_size,true);
     list.clear();
     Integer k;
     for(k=0;k<max_size;k++) pointers[k]=list.insert(Multimap::value_type(0.0,k));
}


Real sorted_vector::read(Integer j) const {
     if(used[j]) return pointers[j]->first;
     else {
         std::cerr<<"sorted_vector::read: entry with given index is not being used. Returning 0.0"<<std::endl;
         return 0.0;
     }
}

void sorted_vector::insert(Integer pos, Real val){
     if(used[pos]) list.erase(pointers[pos]);
     pointers[pos] = list.insert(Multimap::value_type(val,pos));
     used[pos]=true;
}

void sorted_vector::remove(Integer k){
    if(used[k]) list.erase(pointers[k]);
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
    for(Integer i=0;i<pointers.dimension();i++) 
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
    return (Real)  used.size() * (sizeof(Real)+sizeof(Integer)) /* + used.memory() */ + pointers.memory();
}

//************************************************************************************************************************
//                                                                                                                       *
//         The implementation of the class array                                                                         *
//                                                                                                                       *
//************************************************************************************************************************

template<class T> array<T>::array() {}

template<class T> void array<T>::erase_resize_data_field(Integer newsize){
    if (newsize != data.size()) {
        data.clear();
        data.resize(newsize);
    }
}

template<class T> void array<T>::destroy_resize_data_field(Integer newsize){
    data.clear();
    data.resize(newsize);
}

template<class T> void array<T>::enlarge_dim_keep_data(Integer newdim){
    data.resize(newdim);
}

template<class T> array<T>::array(Integer m)
    : data(m)
{
}

template<class T> array<T>::array(Integer m, T t)
    : data(m, t)
{
}

template<class T> void array<T>::destroy() {
    data.clear();
 }

template<class T> Integer array<T>::dimension() const {       // returns dimension of the vector
    return data.size();
}

template<class T> void array<T>::print_info() const {
    std::cout<<"An array of dimension "<<data.size()<<std::endl;
}

template<class T>  Real array<T>::memory() const {
    return (Real)data.capacity() * sizeof(T);
}


//*************************************************************************************************************************************
// Class array: Accessing Elements                                                                                                    *
//*************************************************************************************************************************************


template<class T> const T& array<T>::get(Integer j) const {
     #ifdef DEBUG
         if(j<0||j>=data.size()){
             std::cerr<<"array::get: index out of range. Accessing an element with index "<<j<<" in a array having size "<<data.size()<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     return data[j];
  }

template<class T> T& array<T>::operator[](Integer j){
     #ifdef DEBUG
         if(j<0||j>=data.size()){
             std::cerr<<"array::operator[]: index out of range. Accessing an element with index "<<j<<" in a array having size "<<data.size()<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
    return data[j];
  }

template<class T> const T& array<T>::operator[](Integer j) const {
     #ifdef DEBUG
         if(j<0||j>=data.size()){
             std::cerr<<"array::operator[] const: index out of range. Accessing an element with index "<<j<<" in a array having size "<<data.size()<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
    return data[j];
  }

template<class T> T& array<T>::set(Integer j){
     #ifdef DEBUG
         if(j<0||j>=data.size()){
             std::cerr<<"array::set: index out of range. Accessing an element with index "<<j<<" in a array having size "<<data.size()<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
    return data[j];
  }

//*************************************************************************************************************************************
// Class array: Input / Output                                                                                                 *
//*************************************************************************************************************************************


template<class T> std::istream& operator >> (std::istream& is, array<T> &x) {
     std::cout<<"The components of the array having size "<<x.dimension()<<std::endl;
     for(Integer i=0;i<x.dimension();i++) is >> x[i];
     return is;
 }

template<class T> std::ostream& operator << (std::ostream& os, const array<T> &x) {
    os<<std::endl;
    for(Integer i=0;i<x.dimension();i++) os << x[i] << std::endl;
    os<<std::endl;
    return os;
 }

//*************************************************************************************************************************************
// Class array: Generating special arrays                                                                                     *
//*************************************************************************************************************************************


template<class T> void array<T>::set_all(T d){
     for(Integer k=0; k<data.size(); k++) data[k]=d;
  }

template<class T> void array<T>::resize(Integer newsize){
     erase_resize_data_field(newsize);
  }

template<class T> void array<T>::resize(Integer newsize, T d){
     erase_resize_data_field(newsize);
     set_all(d);
  }

}

#endif


