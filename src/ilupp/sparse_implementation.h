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


#ifndef SPARSE_IMPLEMENTATION_H
#define SPARSE_IMPLEMENTATION_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <stack>
#include <map>
#include <queue>
#include <vector>


#include "declarations.h"

#include "arrays.h"
#include "functions.h"
#include "orderings.h"
#include "pmwm_declarations.h"


#include "functions_implementation.h"
#include "parameters_implementation.h"
#include "orderings_implementation.h"
#include "pmwm_implementation.h"

#include "ILUT.hpp"
#include "ILUTP.hpp"
#include "ILUC.hpp"

#ifdef ILUPLUSPLUS_USES_PARDISO
#include "pardiso_unsym_interface.h"
#endif


// Scalar-Product does not include complex conjugation, needs to be adapted for complex numbers
// Hence, currently the scalar product, norm2, norm2squared,norm2_along_orientation, etc. are incorrect for complex numbers.

namespace iluplusplus {

//************************************************************************************************************************
//                                                                                                                       *
//         The implementation of the class vector_dense                                                                  *
//                                                                                                                       *
//************************************************************************************************************************

//*************************************************************************************************************************************
// Class vector_dense: private functions                                                                                              *
//*************************************************************************************************************************************

template<class T> void vector_dense<T>::erase_resize_data_field(Integer newsize){
    if (size != newsize){
        if (data != 0){ delete [] data; data = 0;}
        if(newsize > 0){
            data = new T[newsize];
            size = newsize;
        } else {
            size = 0; data = 0;
        }  // end if size>0
    }  // end if size != newsize
}

//*************************************************************************************************************************************
// Class vector_dense: Constructor, Destructor, etc.                                                                                  *
//*************************************************************************************************************************************

template<class T> vector_dense<T>::vector_dense(){
    size = 0; data = 0;
}


/*
template<class T> vector_dense<T>::vector_dense(Integer m) {
    size = m;
    data   = new (std::nothrow) T[m];
    if (data == 0){
        std::cerr<<"vector_dense::vector_dense: "<<ippe.error_message()<<std::endl;
        exit(1);
    }
    for(Integer i=0;i<size;i++) data[i]=0;
 }
*/

template<class T> vector_dense<T>::vector_dense(Integer m) {
    size = 0; data = 0; // initialization needed, so that erase_resize_data_field actually can check if resizing needs to take place.
    erase_resize_data_field(m);
    for(Integer i=0;i<size;i++) data[i]=0;
 }


template<class T> vector_dense<T>::vector_dense(Integer m, T t) {
    size = 0; data = 0;
    erase_resize_data_field(m);
    for(Integer i=0;i<size;i++) data[i]=t;
 }

template<class T> vector_dense<T>::vector_dense(const vector_dense& x) {
    size = 0; data = 0;
    erase_resize_data_field(x.size);
    for(Integer i=0;i<size;i++) data[i]=x.data[i];
 }

template<class T> vector_dense<T>::vector_dense(Integer m, T* _data, bool _non_owning)
    : size(m), data(_data), non_owning(_non_owning)
{
}

template<class T> vector_dense<T>::~vector_dense() {
    if (!non_owning && data != 0) delete [] data;
    data = 0;
 }

//*************************************************************************************************************************************
// Class vector_dense: Basic functions                                                                                                *
//*************************************************************************************************************************************

template<class T> void vector_dense<T>::scale(T d){
     for(Integer i=0;i<size;i++) data[i]*=d;
  }

template<class T> void vector_dense<T>::scale(T d, const vector_dense<T>& v){
     resize_without_initialization(v.size);
     for(Integer i=0;i<size;i++) data[i]= d*v.data[i];
  }

template<class T> void vector_dense<T>::add(T d){
     for(Integer i=0;i<size;i++) data[i]+=d;
  }

template<class T> void vector_dense<T>::add(T d, const vector_dense<T>& v){
     resize_without_initialization(v.size);
     for(Integer i=0;i<size;i++) data[i]= d+v.data[i];
  }

template<class T> void vector_dense<T>::power(Real c){
     for(Integer i=0;i<size;i++) data[i] = std::exp(c*std::log(fabs(data[i])));
}

template<class T> void vector_dense<T>::scale_at_end(const vector_dense<T>& v){
     Integer offset = size-v.size;
     #ifdef DEBUG
         if(non_fatal_error((offset<0),"vector_dense<T>::scale_at_end: dimension of *this must be larger than dimension of argument.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     for(Integer i=0;i<v.size;i++) data[i+offset] *= v.data[i];
  }

template<class T> void vector_dense<T>::scale_at_end_and_project(const vector_dense<T>& v, const vector_dense<T>& scale){
    Integer offset = v.size-scale.size;
    resize_without_initialization(scale.dimension());
#ifdef DEBUG
    if(non_fatal_error((offset<0),"vector_dense<T>::inverse_scale_at_end: dimension of *this must be larger than dimension of argument.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    for(Integer i=0;i<v.size;i++) data[i]= v.data[i+offset] * scale.data[i];
}

template<class T> void vector_dense<T>::inverse_scale_at_end(const vector_dense<T>& v){
     Integer offset = size-v.size;
     #ifdef DEBUG
         if(non_fatal_error((offset<0),"vector_dense<T>::inverse_scale_at_end: dimension of *this must be larger than dimension of argument.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     for(Integer i=0;i<v.size;i++) data[i+offset] /= v.data[i];
  }

template<class T> void vector_dense<T>::inverse_scale_at_end_and_project(const vector_dense<T>& v, const vector_dense<T>& scale){
    Integer offset = v.size-scale.size;
    resize_without_initialization(scale.dimension());
#ifdef DEBUG
    if(non_fatal_error((offset<0),"vector_dense<T>::inverse_scale_at_end: dimension of *this must be larger than dimension of argument.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    for(Integer i=0;i<v.size;i++) data[i]= v.data[i+offset] / scale.data[i];
}


template<class T> void vector_dense<T>::add(const vector_dense<T> &v){
     #ifdef DEBUG
         if(non_fatal_error((v.size != size),"vector_dense<T>::add: the vectors have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     for(Integer i=0;i<size;i++) data[i]+=v.data[i];
  }

template<class T> void vector_dense<T>::subtract(const vector_dense<T> &v){
     #ifdef DEBUG
         if(non_fatal_error((v.size != size),"vector_dense<T>::subtract: the vectors have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     for(Integer i=0;i<size;i++) data[i]-=v.data[i];
  }

template<class T> void vector_dense<T>::multiply(const vector_dense<T> &v){
     #ifdef DEBUG
         if(non_fatal_error((v.size != size),"vector_dense<T>::multiply: the vectors have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     for(Integer i=0;i<size;i++) data[i]*=v.data[i];
  }


template<class T> T vector_dense<T>::product() const {
     T product = 1.0;
     for(Integer i=0;i<size;i++) product *= data[i];
     return product;
  }


template<class T> void vector_dense<T>::divide(const vector_dense<T> &v){
     #ifdef DEBUG
         if(non_fatal_error((v.size != size),"vector_dense<T>::divide: the vectors have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     for(Integer i=0;i<size;i++) data[i]/=v.data[i];
  }

template<class T> void vector_dense<T>::invert(){
     for(Integer i=0;i<size;i++) data[i] = 1.0/data[i];
  }

template<class T> void vector_dense<T>::invert(const vector_dense<T> &v){
    resize_without_initialization(v.size);
    for(Integer i=0;i<size;i++) data[i] = 1.0/v.data[i];
  }


template<class T> void vector_dense<T>::add_scaled(T alpha, const vector_dense<T> &v){
     #ifdef DEBUG
         if(v.size != size){
             std::cerr<<"vector_dense<T>::add_scaled: the vectors have incompatible dimensions."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     for(Integer i=0;i<size;i++) data[i]+=alpha*v.data[i];
  }

template<class T> void vector_dense<T>::scale_add(T alpha, const vector_dense<T> &v){
     #ifdef DEBUG
         if(non_fatal_error((v.size != size),"vector_dense<T>::add_scaled: the vectors have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     for(Integer i=0;i<size;i++) data[i] = alpha*data[i]+v.data[i];
  }

template<class T> void vector_dense<T>::vector_addition(const vector_dense<T> &v, const vector_dense<T> &w){
#ifdef DEBUG
    if(v.size != w.size){
        std::cerr<<"vector_dense<T>::addition: the addends have incompatible dimensions."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    if(size != v.size){
        resize(v.size, 0.0);
    } else {
        set_all(0.0);
    }
    for(Integer i=0;i<size;i++) data[i]=v.data[i]+w.data[i];
  }

template<class T> void vector_dense<T>::scaled_vector_addition(const vector_dense<T> &v, T alpha, const vector_dense<T> &w){
#ifdef DEBUG
    if(v.size != w.size){
        std::cerr<<"vector_dense<T>::addition: the addends have incompatible dimensions."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    if(size != v.size){
        resize(v.size, 0.0);
    } else {
        set_all(0.0);
    }
    for(Integer i=0;i<size;i++) data[i]=v.data[i]+(alpha*w.data[i]);
  }


template<class T> void vector_dense<T>::scaled_vector_addition(T alpha, const vector_dense<T> &v, const vector_dense<T> &w){
#ifdef DEBUG
    if(v.size != w.size){
        std::cerr<<"vector_dense<T>::addition: the addends have incompatible dimensions."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    if(size != v.size){
        resize(v.size, 0.0);
    } else {
        set_all(0.0);
    }
    for(Integer i=0;i<size;i++) data[i] = (alpha*v.data[i]) + w.data[i];
  }

template<class T> void vector_dense<T>::scaled_vector_subtraction(T alpha, const vector_dense<T> &w, const vector_dense<T> &v){
#ifdef DEBUG
    if(v.size != w.size){
        std::cerr<<"vector_dense<T>::addition: the addends have incompatible dimensions."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    if(size != v.size){
        resize(v.size, 0.0);
    } else {
        set_all(0.0);
    }
    for(Integer i=0;i<size;i++) data[i]=(alpha*w.data[i])-v.data[i];
  }

template<class T> void vector_dense<T>::vector_subtraction(const vector_dense<T> &v, const vector_dense<T> &w){
#ifdef DEBUG
    if(v.size != w.size){
        std::cerr<<"vector_dense<T>::subtraction: the arguments have incompatible dimensions."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    if(size != v.size){
        resize(v.size, 0.0);
    } else {
        set_all(0.0);
    }
    for(Integer i=0;i<size;i++) data[i]=v.data[i]-w.data[i];
  }


template<class T> void vector_dense<T>::residual(matrix_usage_type use, const matrix_sparse<T> &A, const vector_dense<T> &x, const vector_dense<T> &b){
    if(non_fatal_error(((use==ID) && ((A.columns() != x.dimension()) || (A.rows() != b.dimension())) ), "vector_dense::residual: incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(((use==TRANSPOSE) && ( (A.rows() != x.dimension()) || (A.columns() != b.dimension()) ) ), "vector_dense::residual: incompatible dimensions."  )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    *this=b;
    Integer i,j;
    if ( ( (A.orient() == ROW)&&(use == ID) ) || ( (A.orient() == COLUMN)&&(use == TRANSPOSE) )  )
        for(i=0;i<A.read_pointer_size()-1;i++)
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++)
                data[i]-=A.read_data(j)*x.data[A.read_index(j)];
    else
        for(i=0;i<A.read_pointer_size()-1;i++)
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++)
                data[A.read_index(j)]-=A.read_data(j)*x.data[i];
}


template<class T> void vector_dense<T>::extract(const matrix_sparse<T> &A, Integer m){
    if(non_fatal_error(m+1>A.pointer_size, "vector_dense<T>::extract: the matrix does not have this column or row.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    resize(A.dim_against_orientation(),0.0);
    for(Integer i=A.read_pointer(m); i<A.read_pointer(m+1); i++){
        data[A.read_index(i)] = A.read_data(i);
    }

  }

template<class T> void vector_dense<T>::extract(const vector_dense<T>& x,Integer begin,Integer end) {
    Integer newsize = end-begin;
#ifdef DEBUG
    if(newsize<0||end>x.dimension()||begin<0){ std::cout<<"vector_dense<T>::extract: extraction range "<<begin<<" to "<<end<<" not possible."<<std::endl; throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);}
#endif
    resize_without_initialization(newsize);
    for(Integer k=0;k<newsize;k++) set(k) = x[k+begin];

}

template<class T> void vector_dense<T>::extract_from_matrix_update(T d, const matrix_sparse<T> &A, Integer k){
     if (k+1 >= A.get_pointer_size()){
         std::cerr<<"vector_dense<T>::extract_from_matrix_update: the matrix does not have a column/row "<<k<<"."<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     }
     if (size != A.dim_against_orientation()){
         std::cerr<<"vector_dense<T>::extract_from_matrix_update: the vector and matrix have incompatible dimensions."<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     }
     for(Integer i=A.get_pointer(k); i<A.get_pointer(k+1); i++){
         data[A.get_index(i)] += d*A.get_data(i);
     }
  }

//*************************************************************************************************************************************
// Class vector_dense: Operators                                                                                                      *
//*************************************************************************************************************************************

template<class T> vector_dense<T> vector_dense<T>::operator * (T k) const {           // Multiplication with a scalar
    vector_dense<T> y;
    y.resize_without_initialization(size);
    for(Integer i=0;i<size;i++) y.data[i]=k*data[i];
    return y;
  }

template<class T> vector_dense<T> vector_dense<T>::operator + (vector_dense const &v) const {
    vector_dense<T> y;
#ifdef DEBUG
    if (size != v.size){
        std::cerr<<"vector_dense<T>::operator + : Dimension error"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    y.resize_without_initialization(size);
    for(Integer i=0;i<size;i++) y.data[i]=data[i]+v.data[i];
    return y;
  }

template<class T> vector_dense<T> vector_dense<T>::operator - (vector_dense const &v) const {
    vector_dense<T> y;
#ifdef DEBUG
    if (size != v.size){
        std::cerr<<"vector_dense<T>::operator - : Dimension error"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    y.resize_without_initialization(size);
    for(Integer i=0;i<size;i++) y.data[i]=data[i]-v.data[i];
    return y;
  }

template<class T> T vector_dense<T>::operator * (vector_dense const &v) const {   // Scalar-Product
     #ifdef DEBUG
         if (size != v.size){
             std::cerr<<"vector_dense<T>::operator * (scalar product): Dimension error. Dimension of arguments are: "<<size<<" and "<<v.size<<"."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     T h = (T) 0;
     for (Integer i=0;i<size;i++)
         h += data[i] * std::conj(v.data[i]);
     return h;
  }

template<class T> vector_dense<T>& vector_dense<T>::operator =(const vector_dense<T>& x){   // Assignment-Operator
    if(this == &x) return *this;
    erase_resize_data_field(x.size);
    for(Integer i=0;i<size;i++) data[i]=x.data[i];
    return *this;
}

template<class T> void vector_dense<T>::copy_and_destroy(vector_dense<T>& v){
      size=v.size;
      v.size=0;
      if(data != 0) delete [] data;
      data = v.data;
      v.data = 0;
  }

template<class T> void vector_dense<T>::setup(Integer n, T* data_array){
    #ifdef DEBUG
         std::cerr<<"vector_dense::setup: WARNING: making a vector using pointers. This is not recommended. You are responsible for making sure that this does not result in a segmentation fault!"<<std::endl<<std::flush;
    #endif
    size = n;
    data = data_array;
}

template<class T> void vector_dense<T>::free(Integer& n, T*& data_array){
    #ifdef DEBUG
         std::cerr<<"vector_dense::free: WARNING: not freeing memory and destructor will not free memory. This is not recommended. You are responsible for freeing the memory using the pointer that this function is returning."<<std::endl<<std::flush;
    #endif
    n = size;
    size = 0;
    data_array = data;
    data = 0;
}

template<class T> void vector_dense<T>::null_vector_keep_data(){
    #ifdef DEBUG
         std::cerr<<"vector_dense::null_vector_keep_data: WARNING: not freeing memory and destructor will not free memory. This is not recommended. You are responsible for freeing memory somehow...."<<std::endl<<std::flush;
    #endif
    size = 0;
    data = 0;
}

template<class T> void vector_dense<T>::interchange(vector_dense<T>& v){
    std::swap(v.data,data);
    std::swap(v.size,size);
}


template<class T> void vector_dense<T>::interchange(T*& newdata, Integer& newsize){
#ifdef DEBUG
    std::cerr<<"vector_dense::interchange: WARNING: making a vector using pointers. This is not recommended. You are responsible for making sure that this does not result in a segmentation fault!"<<std::endl<<std::flush;
#endif
    std::swap(data,newdata);
    std::swap(size,newsize);
}


template<class T> void vector_dense<T>::switch_entry(Integer i, Integer j){
     #ifdef DEBUG
         if((i>=size)||(j>=size)){
             std::cerr << "vector_dense::switch_entry: out of domain error: size of vector "<<size<<" entries to be switched: "<<i<<" "<<j<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     T h;
     h = data[i];
     data[i] = data[j];
     data[j] = h;
  }

template<class T> void vector_dense<T>::switch_entry(Integer i, Integer j, T& h){
     #ifdef DEBUG
         if((i>=size)||(j>=size)){
             std::cerr << "vector_dense::switch_entry: out of domain error: size of vector "<<size<<" entries to be switched: "<<i<<" "<<j<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     h = data[i];
     data[i] = data[j];
     data[j] = h;
  }





template<class T> void vector_dense<T>::write(std::string filename) const {
    std::ofstream file(filename.c_str());
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    for (Integer i=0;i<size;i++) file<<data[i]<<std::endl;
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    file.close();
  }

template<class T> void vector_dense<T>::append(std::string filename) const {
    std::ofstream file(filename.c_str(), std::ios_base::app);
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    for (Integer i=0;i<size;i++) file<<data[i]<<std::endl;
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    file.close();
  }

template<class T> void  vector_dense<T>::write_with_indices(std::string filename) const {
    std::ofstream file(filename.c_str());
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    for (Integer i=0;i<size;i++) file<<i<<"\t"<<data[i]<<std::endl;
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    file.close();
  }

template<class T> void  vector_dense<T>::append_with_indices(std::string filename, Integer shift) const {
    std::ofstream file(filename.c_str(), std::ios_base::app);
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    for (Integer i=0;i<size;i++) file<<i+shift<<"\t"<<data[i]<<std::endl;
    if(non_fatal_error(!file.good(),"vector_dense::write: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    file.close();
  }


template <class T> void vector_dense<T>::quicksort(index_list& list){
    quicksort(list,0,dimension()-1);
}

template <class T> void vector_dense<T>::quicksort(){
    quicksort(0,dimension()-1);
}

template <class T> void vector_dense<T>::quicksort(Integer left, Integer right){
     #ifdef DEBUG
         if(left < 0){
             std::cout<<"vector_dense::quicksort: smallest sorting index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(right > dimension()-1){
             std::cout<<"vector_dense::quicksort: largest index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer i,j;
     T m;
     if(left<right){
         m=data[left];
         i=left;
         j=right;
         while(i<=j){
             while(data[i]<m) i++;
             while(data[j]>m) j--;
             if(i<=j){
                 switch_entry(i,j);
                 i++;
                 j--;
             }
         }
         vector_dense<T>::quicksort(left,j);
         vector_dense<T>::quicksort(i,right);
     }
  }

template <class T> void vector_dense<T>::quicksort(index_list& list, Integer left, Integer right){
     #ifdef DEBUG
         if(dimension() != list.dimension()){
             std::cout<<"vector_dense::quicksort: vector dimension and index list dimension are not equal."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(left < 0){
             std::cout<<"vector_dense::quicksort: smallest sorting index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(right > dimension()-1){
             std::cout<<"vector_dense::quicksort: largest index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer i,j;
     T m;
     if(left<right){
         m=data[left];
         i=left;
         j=right;
         while(i<=j){
             while(data[i]<m) i++;
             while(data[j]>m) j--;
             if(i<=j){
                 switch_entry(i,j);
                 list.switch_index(i,j);
                 i++;
                 j--;
             }
         }
         vector_dense<T>::quicksort(list,left,j);
         vector_dense<T>::quicksort(list,i,right);
     }
  }

template <class T> void vector_dense<T>::sort(index_list& list, Integer left, Integer right, Integer m){
     Integer i,j,mid,a_list;
     T a,help;
     #ifdef DEBUG
         Integer n = right-left+1; // number of elements to be sorted
         if(m<0||m>n){
             std::cerr<<"vector_dense::sort: choosing the largest "<<m<<" out of "<<n<<" elements in not possible."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer k = right-m+1;
     while(true){
         if(right<=left+1){
             if(right==left+1 && data[right]<data[left]){
                 switch_entry(left,right,help);
                 list.switch_index(left,right);
             }  // end if
             break;
         } else {
             mid = (left+right)/2;
             switch_entry(mid,left+1,help);
             list.switch_index(mid,left+1);
             if(data[left]>data[right]){
                 switch_entry(left,right,help);
                 list.switch_index(left,right);
             }
             if(data[left+1]>data[right]){
                 switch_entry(left+1,right,help);
                 list.switch_index(left+1,right);
             }
             if(data[left]>data[left+1]){
                 switch_entry(left,left+1,help);
                 list.switch_index(left,left+1);
             }
             i=left+1;
             j=right;
             a=data[left+1];
             a_list=list[left+1];
             while(true){
                 do i++; while(data[i]<a);
                 do j--; while(data[j]>a);
                 if(j<i) break;
                 switch_entry(i,j,help);
                 list.switch_index(i,j);
             } // end while
             data[left+1]=data[j];
             list[left+1]=list[j];
             data[j]=a;
             list[j]=a_list;
             if (j>=k) right = j-1;
             if (j<=k) left = i;
         }// end if
     } // end while
  }


template<class T> void vector_dense<T>::take_largest_elements_by_abs_value(index_list& list, Integer n) const {
    Integer offset = size-n;
    Integer i;
    vector_dense<Real> input_abs;   // will store the absolute values of input
    index_list complete_list;
    list.resize_without_initialization(n);
    complete_list.resize(size);
    input_abs.absvalue(*this);                   // make vector containing abs. value of input
    //input_abs.quicksort(complete_list,0,size-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
    input_abs.sort(complete_list,0,size-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
    // we need the indices of the largest elements in ascending order. To get this order, we sort here.
    complete_list.quicksort(offset,size-1);
    for (i=0;i<n;i++) list[i]=complete_list[offset+i];
  }

template<class T> void vector_dense<T>::take_largest_elements_by_abs_value_with_threshold(index_list& list, Integer n, Real tau) const {
    Real norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
    for(i=0;i<size;i++) norm += absvalue_squared(data[i]);
    norm=sqrt(norm);
    for(i=0;i<size;i++){
        if(fabs(data[i]) > norm*tau){
            input_abs.data[number_elements_larger_tau]=fabs(data[i]);
            complete_list[number_elements_larger_tau]=i;
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we need the indices of the largest elements in ascending order. To get this order, we sort here.
        complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
  }


template<class T> void vector_dense<T>::take_largest_elements_by_abs_value_with_threshold(Real& norm_input, index_list& list, const index_list& perm, Integer n, Real tau, Integer from, Integer to) const {
    // list will contain at most n elements of *this having relative absolute value (in 2-norm) of at least tau.
    // Only those elements indexed by "from" to "to", including "from", excluding "to" are accessed using "perm".
    // indices in list refer to "perm"; the index of the largest element will always be at the end.
    norm_input = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer pos_larg_element=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error(((from<0)||(to>size)), "vector_dense::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(from>to || size==0){
        std::cerr<<"vector_dense<T>::take_largest_elements_by_abs_value_with_threshold: arguments out of range. Returning empty list."<<std::endl;
        list.resize_without_initialization(0);
        return;
    }
#endif
    if(n==0){
        list.resize_without_initialization(0);
        return;
    }
    for(i=from;i<to;i++) norm_input += absvalue_squared(data[perm[i]]);
    norm_input=sqrt(norm_input);
    for(i=from;i<to;i++){
        if(fabs(data[perm[i]]) > norm_input*tau){
            input_abs.data[number_elements_larger_tau]=fabs(data[perm[i]]);
            complete_list[number_elements_larger_tau]=i; // do not need perm[i]
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau==0){
        list.resize(0);
        return;
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        pos_larg_element=offset;
        for(i=offset+1;i<number_elements_larger_tau;i++)
            if(input_abs.data[i]>input_abs.data[pos_larg_element])
                pos_larg_element=i;
        complete_list.switch_index(pos_larg_element,number_elements_larger_tau-1);
        //complete_list.quicksort(offset,number_elements_larger_tau-2);  //  not really necessary for most applications, keep largest element at end
        list.resize_without_initialization(n);
        for (i=0;i<list.dimension();i++) list[i]=complete_list[offset+i];
        // pos_larg_element=n-1;
    } else {
        pos_larg_element=0;
        if(number_elements_larger_tau>0)
            for(i=1;i<number_elements_larger_tau;i++)
                if(input_abs[i]>input_abs[pos_larg_element])
                    pos_larg_element=i;
        complete_list.switch_index(pos_larg_element,list.dimension()-1);
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }  // end if
  }

template<class T> void vector_dense<T>::insert_at_end(const vector_dense<T>& v){
    #ifdef DEBUG
        if(non_fatal_error((dimension() < v.dimension()), "vector_dense::insert_at_end: cannot insert, vector too large.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    Integer k;
    Integer offset = dimension() - v.dimension();
    for(k=0;k<v.dimension();k++) set(k+offset) = v[k];
}

template<class T> void vector_dense<T>::insert(const vector_dense<T>& b, Integer position, T value){
    if(non_fatal_error(position<0 || position > b.dimension(),"vector_dense<T>::insert: trying to insert at a position that is not possible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    erase_resize_data_field(b.dimension()+1);
    Integer i;
    for(i=0;i<position;i++) set(i) = b[i];
    set(position) = value;
    for(i=position;i<b.dimension();i++) set(i+1) = b[i];
}


template<class T> void vector_dense<T>::permute(const vector_dense<T>& x, const index_list& perm){
#ifdef DEBUG
    if(non_fatal_error((x.size!= perm.dimension()), "vector_dense::permute: permutation and vector must have same dimension.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    erase_resize_data_field(x.size);
    for(Integer k=0;k<size;k++)
        (*this)[k] = x[perm[k]];
}

template<class T> void vector_dense<T>::permute(const index_list& perm){
    vector_dense<T> H;
#ifdef DEBUG
    if(non_fatal_error((size!= perm.dimension()), "vector_dense::permute: permutation and vector must have same dimension.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    H.permute(*this,perm);
    interchange(H);
}

template<class T> void vector_dense<T>::permute_at_end(const vector_dense<T>& x, const index_list& perm){
    Integer k;
    Integer offset = x.size - perm.dimension();
#ifdef DEBUG
    if(non_fatal_error((x.size < perm.dimension()), "vector_dense::permute_at_end: dimension of permutation is too large.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    erase_resize_data_field(x.size);
    for(k=0;k<offset;k++) set(k) = x[k]; 
    for(k=0;k<perm.dimension();k++) set(k+offset) = x[perm[k]+offset];

}

template<class T> void vector_dense<T>::permute_at_end(const index_list& perm){
    vector_dense<T> H;
#ifdef DEBUG
    if(non_fatal_error((size < perm.dimension()), "vector_dense::permute_at_end: dimension of permutation is too large.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    H.permute_at_end(*this,perm);
    interchange(H);
}


template<class T> void vector_dense<T>::take_weighted_largest_elements_by_abs_value_with_threshold(Real& norm_input,index_list& list, const index_list& perm, const vector_dense<Real>& weights, Integer n, Real tau, Integer from, Integer to) const {
    // list will contain at most n elements of *this having relative absolute value (in 2-norm) of at least tau.
    // Only those elements indexed by "from" to "to", including "from", excluding "to" are accessed using "perm".
    // indices in list refer to "perm"; the index of the largest element will always be at the end.
    norm_input = 0.0;
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer pos_larg_element=0;
    Real value_larg_element,value;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error(((from<0)||(to>size)), "vector_dense::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(from>to || size==0){
        std::cerr<<"vector_dense::take_largest_elements_by_abs_value_with_threshold: arguments out of range: returning empty list"<<std::endl;
        list.resize_without_initialization(0);
        return;
    }
#endif
    if(n==0){
        list.resize_without_initialization(0);
        return;
    }
    for(i=from;i<to;i++) norm_input += absvalue_squared(weights.read(i)*data[perm[i]]);
    norm_input=sqrt(norm_input);
    for(i=from;i<to;i++){
        product = weights.read(i)*fabs(data[perm[i]]);
        if(product > norm_input*tau){
            input_abs.data[number_elements_larger_tau]=product;
            complete_list[number_elements_larger_tau]=i; // do not need perm[i]
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau==0){
        list.resize(0);
        return;
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        //pos_larg_element=complete_list[number_elements_larger_tau-1];
        //complete_list.quicksort(offset,number_elements_larger_tau-2);  //  not really necessary for most applications, keep largest element at end
        list.resize_without_initialization(n);
        for (i=0;i<list.dimension();i++) list[i]=complete_list[offset+i];
        // pos_larg_element=n-1;
    } else {
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }  // end if
    if(list.dimension()>0){
        pos_larg_element=0;
        value_larg_element=fabs(data[perm[0]]);
        for(i=1;i<list.dimension();i++){
            value=fabs(data[perm[i]]);
            if(value>value_larg_element){
                pos_larg_element=i;
                value_larg_element=value;
            } // end if
        } // end for
        list.switch_index(pos_larg_element,list.dimension()-1);
    } // end if
  }


template<class T> void vector_dense<T>::take_largest_elements_by_abs_value_with_threshold(index_list& list, Integer n, Real tau, Integer from, Integer to) const {
    Real norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error(((from<0)||(to>size)), "vector_dense::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    for(i=from;i<to;i++) norm += absvalue_squared(data[i]);
    norm=sqrt(norm);
    for(i=from;i<to;i++){
        if(fabs(data[i]) > norm*tau){
            input_abs.data[number_elements_larger_tau]=fabs(data[i]);
            complete_list[number_elements_larger_tau]=i;
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we need the indices of the largest elements in ascending order. To get this order, we sort here.
        complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
  }

template<class T> void vector_dense<T>::take_largest_elements_by_abs_value_with_threshold(Real& norm,index_list& list, Integer n, Real tau, Integer from, Integer to) const {
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error(((from<0)||(to>size)), "vector_dense::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    for(i=from;i<to;i++) norm += absvalue_squared(data[i]);
    norm=sqrt(norm);
    for(i=from;i<to;i++){
        if(fabs(data[i]) > norm*tau){
            input_abs.data[number_elements_larger_tau]=fabs(data[i]);
            complete_list[number_elements_larger_tau]=i;
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we need the indices of the largest elements in ascending order. To get this order, we sort here.
        complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
  }

template<class T> void vector_dense<T>::take_weighted_largest_elements_by_abs_value_with_threshold(Real& norm,index_list& list, const vector_dense<T>& weight, Integer n, Real tau, Integer from, Integer to) const {
    T product;
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error(((from<0)||(to>size)), "vector_dense::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    for(i=from;i<to;i++) norm += absvalue_squared(weight.read(i)*data[i]);
    norm=sqrt(norm);
    for(i=from;i<to;i++){
        product=fabs(weight.read(i)*data[i]);
        if(product > norm*tau){
            input_abs.data[number_elements_larger_tau]=product;
            complete_list[number_elements_larger_tau]=i;
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we need the indices of the largest elements in ascending order. To get this order, we sort here.
        complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
  }

//*************************************************************************************************************************************
// Class vector_dense: Functions                                                                                                      *
//*************************************************************************************************************************************

template<class T> Integer vector_dense<T>::dimension() const {       // returns dimension of the vector
    return size;
  }

template<class T> Integer vector_dense<T>::dim() const {       // returns dimension of the vector
    return size;
  }

template<class T> void vector_dense<T>::absvalue(){             // overwrites the vector with its absolute value, elementwise
    for(Integer i=0;i<size;i++) data[i]=fabs(data[i]);
  }

template<class T> void vector_dense<T>::absvalue(const vector_dense<T>& v){   // (*this) contains the absolute values of v elementwise.
    erase_resize_data_field(v.size);
    for(Integer i=0;i<size;i++) data[i]=fabs(v.data[i]);
}


template<class T> void  vector_dense<T>::absvalue(const vector_dense<T>& v, Integer begin, Integer n){
    if(non_fatal_error((begin+n>v.size),"vector_dense::absvalue: dimensions are  are incompatible")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    erase_resize_data_field(n);
    for(Integer i=0;i<n;i++) data[i]=fabs(v.data[i+begin]);
  }

template<class T> void vector_dense<T>::value(const T* values, Integer begin, Integer n){   // (*this) contains n elements of the field data from begin.
    erase_resize_data_field(n);
    for(Integer i=begin;i<begin+n;i++) data[i]=values[i];
  }

template<class T> void vector_dense<T>::absvalue(const T* values, Integer begin, Integer n){   // (*this) contains n absolute values of the field data from begin.
    erase_resize_data_field(n);
    for(Integer i=begin;i<begin+n;i++) data[i]=fabs(values[i]);
  }

template<class T> void vector_dense<T>::insert_value(const matrix_oriented<T> A, Integer begin_matrix, Integer n, Integer begin_vector){   // (*this) contains n absolute values of the matrix from begin.
    if(non_fatal_error(((begin_vector+n>size)||(begin_matrix+n>A.size)),"vector_dense::insert_value: vector/matrix are incompatible")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    for(Integer i=0;i<n;i++) data[begin_vector+i]=A.data[begin_matrix+i];
  }

template<class T> void vector_dense<T>::insert_absvalue(const matrix_oriented<T> A, Integer begin_matrix, Integer n, Integer begin_vector){   // (*this) contains n absolute values of the matrix from begin.
    if(non_fatal_error(((begin_vector+n>size)||(begin_matrix+n>A.size)),"vector_dense::insert_value: vector/matrix are incompatible")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    for(Integer i=0;i<n;i++) data[begin_vector+i]=fabs(A.data[begin_matrix+i]);
  }


template<class T> Real vector_dense<T>::norm1() const {          // returns the 1-norm of a vector
    Real norm = 0.0;
    for(Integer i=0;i<size;i++) norm += fabs(data[i]);
    return norm;
  }

template<class T> Real vector_dense<T>::norm2() const {          // returns the 2-norm of a vector
    return sqrt(norm2_squared());
  }

template<class T> Real vector_dense<T>::norm2_squared() const {          // returns the 2-norm of a vector
     Real h = (T) 0;
     for (Integer i=0;i<size;i++)
         h += std::abs(std::conj(data[i]) * data[i]);
     return h;
  }


template<> Real vector_dense<Real>::norm2_squared() const {          // returns the 2-norm of a vector
     Real h = (Real) 0;
     for (Integer i=0;i<size;i++) h += sqr(data[i]);
     return h;
  }

template<> Real vector_dense<Complex>::norm2_squared() const {          // returns the 2-norm of a vector
     Real h = (Real) 0;
     for (Integer i=0;i<size;i++) h += norm(data[i]);
     return h;
  }

template<class T> Real vector_dense<T>::norm_max() const {       // returns the maximum-norm of a vector
    Real norm = 0.0;
    for(Integer i=0;i<size;i++) norm = max(norm,fabs(data[i]));
    return norm;
  }

template<class T> T vector_dense<T>::sum_over_elements() const {          // returns the sum over the elements
    if (size != 0){
       T sum = data[0];
       for(Integer i=1;i<size;i++) sum += data[i];
       return sum;
    } else {
        std::cerr<<"vector_dense::max_over_elements: vector has size 0. Sum does not exist."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
  }

template<class T> T vector_dense<T>::max_over_elements() const {          // returns the sum over the elements
    if (size != 0){
       T maximum = data[0];
          for(Integer i=1;i<size;i++) maximum = max(maximum,data[i]);
       return maximum;
    } else {
       std::cerr<<"vector_dense::max_over_elements: vector has size 0. Maximum does not exist."<<std::endl;
       throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
  }


template<class T> T vector_dense<T>::min_over_elements() const {          // returns the minimum over the elements of a vector
    if (size != 0){
       T minimum = data[0];
       for(Integer i=1;i<size;i++) minimum = min(minimum,data[i]);
       return minimum;
    } else {
        std::cerr<<"vector_dense::max_over_elements: vector has size 0. Minimum does not exist."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
  }

template<class T> T vector_dense<T>::min_over_elements_ignore_negative() const {          // returns the minimum over the elements of a vector
    if (size != 0){
       bool found_non_negative = false;
       T minimum = -1.0;
       for(Integer i=0;i<size;i++){
           if (found_non_negative){ 
               if (data[i]>=0.0) minimum = min(minimum,data[i]);
           } else {
               if(data[i]>= 0.0){
                   minimum = data[i];
                   found_non_negative = true;
               }
           }
       }
       if(found_non_negative) return minimum;
       else {
           std::cerr<<"vector_dense::max_over_elements_ignore_negative: No non-negative element exists. Return Inf."<<std::endl;
           return std::exp(10000.0);
       }
    } else {
          std::cerr<<"vector_dense::max_over_elements_ignore_negative: vector has size 0. Minimum does not exist."<<std::endl;
          throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
  }

template<class T> T vector_dense<T>::min_over_elements_ignore_negative(Integer& ind) const {          // returns the minimum over the elements of a vector
    if (size != 0){
       bool found_non_negative = false;
       ind = -1;
       T minimum = -1.0;
       for(Integer i=0;i<size;i++){
           if (found_non_negative){ 
               if (data[i]>=0.0){
                   if(data[i]<minimum){
                       minimum = data[i];
                       ind = i;
                    }
               }
           } else {
               if(data[i]>= 0.0){
                   minimum = data[i];
                   ind = i;
                   found_non_negative = true;
               }
           }
       }
       if(found_non_negative) return minimum;
       else {
           std::cerr<<"vector_dense::max_over_elements_ignore_negative: No non-negative element exists. Return Inf and negative index"<<std::endl;
           return std::exp(10000.0);
       }
    } else {
          std::cerr<<"vector_dense::max_over_elements_ignore_negative: vector has size 0. Minimum does not exist."<<std::endl;
          throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
  }


template<class T> void vector_dense<T>::min_rows(const matrix_dense<T>& A){
    if(A.columns() == 0 || A.rows() == 0){
        std::cerr<<"matrix_dense<T>::min_rows: matrix must have positive dimension"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    resize(A.rows());
    Integer i,j;
    for(i=0;i<A.rows();i++) set(i) = A.read(i,0); 
    for(i=0;i<A.rows();i++)
        for(j=0;j<A.columns();j++)
            if(A.read(i,j) < get(i)) set(i) = A.read(i,j);
}

template<class T> void vector_dense<T>::max_rows(const matrix_dense<T>& A){
    if(A.columns() == 0 || A.rows() == 0){
        std::cerr<<"matrix_dense<T>::max_rows: matrix must have positive dimension"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    resize(A.rows());
    Integer i,j;
    for(i=0;i<A.rows();i++) set(i) = A.read(i,0); 
    for(i=0;i<A.rows();i++)
        for(j=0;j<A.columns();j++)
            if(A.read(i,j) > get(i)) set(i) = A.read(i,j);
}

template<class T> void vector_dense<T>::min_columns(const matrix_dense<T>& A){
    if(A.columns() == 0 || A.rows() == 0){
        std::cerr<<"matrix_dense<T>::min_columns: matrix must have positive dimension"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    resize(A.columns());
    Integer i,j;
    for(j=0;j<A.columns();j++) set(j) = A.read(0,j); 
    for(j=0;j<A.columns();j++)
        for(i=0;i<A.rows();i++)
            if(A.read(i,j) < get(j)) set(j) = A.read(i,j);
}

template<class T> void vector_dense<T>::max_columns(const matrix_dense<T>& A){
    if(A.columns() == 0 || A.rows() == 0){
        std::cerr<<"matrix_dense<T>::max_columns: matrix must have positive dimension"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    resize(A.columns());
    Integer i,j;
    for(j=0;j<A.columns();j++) set(j) = A.read(0,j); 
    for(j=0;j<A.columns();j++)
        for(i=0;i<A.rows();i++)
            if(A.read(i,j) > get(j)) set(j) = A.read(i,j);
}


template<class T> bool vector_dense<T>::zero_check(Integer k){
    #ifdef DEBUG
         if(k<0||k>=size){
             std::cerr<<"vector_dense::zero_check: index out of range."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
    #endif
    if(data[k]==0) return true;
    else return false;
  }

template<class T> void vector_dense<T>::shortest_vector_point_line(const vector_dense<T>& r, const vector_dense<T>& p, const vector_dense<T>& t){
    *this = p - r;
    T factor = ((*this)*t) / (t*t);
    add_scaled(-factor,t);
}

template<class T> Real vector_dense<T>::distance_point_to_line(const vector_dense<T>& p, const vector_dense<T>& t) const {
    vector_dense<T> shortest_vector;
    shortest_vector.shortest_vector_point_line(*this,p,t);
    return shortest_vector.norm2();
}


//*************************************************************************************************************************************
// Class vector_dense: Accessing Elements                                                                                             *
//*************************************************************************************************************************************


template<class T> T vector_dense<T>::get(Integer j) const {
     #ifdef DEBUG
         if(j<0||j>=size){
             std::cerr<<"vector_dense::get: index out of range. Accessing an element with index "<<j<<" in a vector having size "<<size<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     return data[j];
  }

template<class T> T vector_dense<T>::read(Integer j) const {
     #ifdef DEBUG
         if(j<0||j>=size){
             std::cerr<<"vector_dense::read: index out of range. Accessing an element with index "<<j<<" in a vector having size "<<size<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     return data[j];
  }

template<class T> T& vector_dense<T>::operator[](Integer j){
     #ifdef DEBUG
         if(j<0||j>=size){
             std::cerr<<"vector_dense::operator[]: index out of range. Accessing an element with index "<<j<<" in a vector having size "<<size<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
    return data[j];
  }

template<class T> const T& vector_dense<T>::operator[](Integer j) const {
     #ifdef DEBUG
         if(j<0||j>=size){
             std::cerr<<"vector_dense::operator[]: index out of range. Accessing an element with index "<<j<<" in a vector having size "<<size<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
    return data[j];
  }

template<class T> T& vector_dense<T>::set(Integer j){
     #ifdef DEBUG
         if(j<0||j>=size){
             std::cerr<<"vector_dense::set: index out of range. Accessing an element with index "<<j<<" in a vector having size "<<size<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
    return data[j];
  }

//*************************************************************************************************************************************
// Class vector_dense: Input / Output                                                                                                 *
//*************************************************************************************************************************************


template<class T> std::istream& operator >> (std::istream& is, vector_dense<T> &x) {
     std::cout<<"The components of the vector_dense having size "<<x.size<<std::endl;
     for(Integer i=0;i<x.dimension();i++) is >> x[i];
     return is;
 }

template<class T> std::ostream& operator << (std::ostream& os, const vector_dense<T> &x) {
    os<<std::endl;
    for(Integer i=0;i<x.dimension();i++) os << x[i] << std::endl;
    os<<std::endl;
    return os;
 }

template<class T> void vector_dense<T>::print_info() const {
    std::cout<<"A vector of dimension "<<size<<std::endl;
  }

//*************************************************************************************************************************************
// Class vector_dense: Generating special vectors                                                                                     *
//*************************************************************************************************************************************


template<class T> void vector_dense<T>::unitvector(Integer j){
     Integer k;
     for(k=0;k<size;k++) data[k]=0.0;
     data[j]=1.0;
  }

template<class T> void vector_dense<T>::set_all(T d){
     for(Integer k=0; k<size; k++) data[k]=d;
  }

template<class T> void vector_dense<T>::set_natural_numbers(){
     for(Integer k=0; k<size; k++) data[k] = (T) k;
  }

template<class T> void vector_dense<T>::resize(Integer newsize){
    erase_resize_data_field(newsize);
    set_all((T) 0.0);
  }

template<class T> void vector_dense<T>::resize(Integer newsize, T d){
    erase_resize_data_field(newsize);
    set_all(d);
  }

template<class T> void vector_dense<T>::resize_set_natural_numbers(Integer newsize){
    erase_resize_data_field(newsize);
    set_natural_numbers();
  }

template<class T> void vector_dense<T>::resize_without_initialization(Integer newsize){
    erase_resize_data_field(newsize);
  }


template<class T> void vector_dense<T>::norm2_of_dim1(const matrix_sparse<T>& A, orientation_type o) {
    Integer vector_size,i;
    Integer j;
    if (o==ROW) vector_size = A.rows();
    else vector_size = A.columns();
    resize(vector_size,0.0);
    if(o==A.orient())
        for(i=0;i<A.read_pointer_size()-1;i++)
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++)
                data[i]+=absvalue_squared(A.read_data(j));
    else
        for(j=0;j<A.read_pointer(A.read_pointer_size()-1);j++)
            data[A.read_index(j)]+=absvalue_squared(A.read_data(j));
    for(i=0;i<vector_size;i++) data[i]=sqrt(data[i]);
  }


template<class T> void vector_dense<T>::norm1_of_dim1(const matrix_sparse<T>& A, orientation_type o) {
    Integer vector_size,i;
    Integer j;
    if (o==ROW) vector_size = A.rows();
    else vector_size = A.columns();
    resize(vector_size,0.0);
    if(o==A.orientation)
        for(i=0;i<A.read_pointer_size()-1;i++)
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++)
                data[i]+=fabs(A.read_data(j));
    else
        for(j=0;j<A.read_pointer(A.read_pointer_size()-1);j++)
            data[A.read_index(j)]+=fabs(A.read_data(j));
  }


template<class T>  Real vector_dense<T>::memory() const{
    return (Real) (sizeof(T)*size + sizeof(Integer));
}



//*************************************************************************************************************************************
// Global functions used for vector_dense<T>                                                                                          *
//*************************************************************************************************************************************


 // sorts the elements of vector dense in list, accessing occurs by permutation.
 // the elements of v will be sorted in such a manner that they occur in the same order
 // as in the inverse of permutation
void quicksort(index_list& v, index_list& list, const index_list& permutation, Integer left, Integer right){
     #ifdef DEBUG
         if(list.dimension() != v.dimension() || list.dimension() != permutation.dimension()){
             std::cout<<"vector_dense::quicksort: vector, list or permutation dimension are not equal."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(left < 0){
             std::cout<<"vector_dense::quicksort: smallest sorting index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(right > v.dimension()-1){
             std::cout<<"vector_dense::quicksort: largest index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer i,j;
     Integer m;
     if(left<right){
         m=permutation[v[left]];
         i=left;
         j=right;
         while(i<=j){
             while(permutation[v[i]]<m) i++;
             while(permutation[v[j]]>m) j--;
             if(i<=j){
                 v.switch_index(i,j);
                 list.switch_index(i,j);
                 i++;
                 j--;
             }
         }
         quicksort(v,list,permutation,left,j);
         quicksort(v,list,permutation,i,right);
     }
  }

// same as above, but no list is returned.
void quicksort(index_list& v, const index_list& permutation, Integer left, Integer right){
     #ifdef DEBUG
         if(v.dimension() != permutation.dimension()){
             std::cout<<"vector_dense::quicksort: vector and permutation dimension are not equal."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(left < 0){
             std::cout<<"vector_dense::quicksort: smallest sorting index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(right > v.dimension()-1){
             std::cout<<"vector_dense::quicksort: largest index must be positive."<<std::endl<<std::flush;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer i,j;
     Integer m;
     if(left<right){
         m=permutation[v[left]];
         i=left;
         j=right;
         while(i<=j){
             while(permutation[v[i]]<m) i++;
             while(permutation[v[j]]>m) j--;
             if(i<=j){
                 v.switch_index(i,j);
                 i++;
                 j--;
             }
         }
         quicksort(v,permutation,left,j);
         quicksort(v,permutation,i,right);
     }
  }

//************************************************************************************************************************
//                                                                                                                       *
//         The implementation of the class vector_sparse_dynamic                                                         *
//                                                                                                                       *
//************************************************************************************************************************

//*************************************************************************************************************************************
// Class vector_dense: Constructor, Destructor, etc.                                                                                  *
//*************************************************************************************************************************************


template<class T> vector_sparse_dynamic<T>::vector_sparse_dynamic(){
    size=0; nnz=0;
  }

template<class T> void vector_sparse_dynamic<T>::erase_resize_data_fields(Integer m) {
    if(size != m) {
        nnz = 0;
        data.resize(m);
        occupancy.resize(m);
        pointer.resize(m);
        size = m;
    }
}

template<class T> void vector_sparse_dynamic<T>::resize(Integer m) {
    erase_resize_data_fields(m);
    zero_set();
}


template<class T> vector_sparse_dynamic<T>::vector_sparse_dynamic(Integer m) {
    size=0;
    nnz=0;
    resize(m);
 }

//*************************************************************************************************************************************
// Class vector_sparse_dynamic: Basic functions                                                                                       *
//*************************************************************************************************************************************

template<class T> Integer vector_sparse_dynamic<T>::dimension() const {
    return size;
  }

template<class T> Integer vector_sparse_dynamic<T>::dim() const {
    return size;
  }

template<class T> Integer vector_sparse_dynamic<T>::non_zeroes() const {
    return nnz;
  }

template<class T> T& vector_sparse_dynamic<T>::operator[](Integer j){
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::operator[]: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if (occupancy[j]<0) {
         occupancy[j]=nnz;
         pointer[nnz]=j;
         nnz++;
         data[occupancy[j]] = static_cast<T>(0);
         return data[occupancy[j]];
     } else {
         return data[occupancy[j]];
     }
  }

template<class T> const T& vector_sparse_dynamic<T>::operator[](Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::operator[]: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if (occupancy[j]<0) {
         return (T) 0;
     } else {
         return data[occupancy[j]];
     }
  }

template<class T> void vector_sparse_dynamic<T>::zero_set(Integer j){
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::zero_set: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if(occupancy[j]>=0){
         data[occupancy[j]] = static_cast<T>(0);
         occupancy[j]=-1;
     }
  }

template<class T> Integer vector_sparse_dynamic<T>::get_occupancy(Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::get_occupancy: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     return occupancy[j];
  }

template<class T> T vector_sparse_dynamic<T>::get_data(Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::get_data: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     return data[j];
  }

template<class T> Integer vector_sparse_dynamic<T>::get_pointer(Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::get_pointer: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     return pointer[j];
  }
template<class T> bool vector_sparse_dynamic<T>::zero_check(Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::zero_check: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     return (occupancy[j]<0);
  }

template<class T> bool vector_sparse_dynamic<T>::non_zero_check(Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::non_zero_check: out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     return (occupancy[j]>=0);
  }

template<class T>  T vector_sparse_dynamic<T>::read(Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>=size){
            std::cout<<"vector_sparse_dynamic<T>::read(): out of range. Trying to access "<<j<<" in a vector having size "<<size<<std::endl;  
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if (occupancy[j] < 0)
         return static_cast<T>(0);
     else
         return data[occupancy[j]];
}

template<class T> void vector_sparse_dynamic<T>::zero_reset() {
    for (Integer i=0; i<nnz; ++i)
        occupancy[pointer[i]] = -1;
    nnz = 0;
}

template<class T> void vector_sparse_dynamic<T>::zero_set(){
    for (Integer i=0; i<size; i++)
        occupancy[i] = -1;
    nnz = 0;
}


template<class T> void vector_sparse_dynamic<T>::print_non_zeroes() const {
    Integer k;
    std::cout<<"A vector of dimension "<<size<<std::endl;
    for(k=0;k<nnz;k++) std::cout<<"index "<<pointer[k]<<": "<<data[k]<<std::endl;
}

template<class T> vector_dense<T> vector_sparse_dynamic<T>::expand() const {
     vector_dense<T> z(size);
     for(Integer i=0;i<nnz;i++) z[pointer[i]]=data[i];
     return z;
  }

template<class T> Real vector_sparse_dynamic<T>::norm1() const {
     Real z=0.0;
     for(Integer i=0;i<nnz;i++) z += std::abs(data[i]);
     return z;
  }

template<class T> Real vector_sparse_dynamic<T>::norm2() const {
     Real z=0.0;
     for(Integer i=0;i<nnz;i++) z += sqr(std::abs(data[i]));
     return sqrt(z);
  }


template<class T> Real vector_sparse_dynamic<T>::norm_max() const {
     Real z=0.0;
     for(Integer i=0;i<nnz;i++) z=max(std::abs(data[i]),z);
     return z;
  }

template<class T> T vector_sparse_dynamic<T>::abs_max() const {
     Real max=0.0;
     Real z;
     for(Integer i=0;i<nnz;i++) if(std::abs(data[i])>max){max=std::abs(data[i]); z=data[i];}
     return z;
  }

template<class T> T vector_sparse_dynamic<T>::abs_max(Integer& pos) const {
     pos=-1;
     Real max = 0.0;
     T z = 0.0;
     for(Integer i=0;i<nnz;i++)
         if(std::abs(data[i])>max){
             max=std::abs(data[i]);
             z=data[i]; 
             pos=pointer[i];
     }
     return z;
  }


template<class T> void vector_sparse_dynamic<T>::scale(T d) {
     for(Integer i=0; i<nnz; i++)
         data[i] *= d;
  }

template<class T> T vector_sparse_dynamic<T>::operator * (const vector_sparse_dynamic<T>& y) const { // scalar product
    if(non_fatal_error(dimension() != y.dimension(), "vector_sparse_dynamic::operator * : dimensions must be equal.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(dimension()==0) return 0.0;
    T scalar_product=0.0;
    Integer k;
    for(k=0;k<nnz;k++){
        if (y.occupancy[pointer[k]]>=0) scalar_product += data[k]*y.data[y.occupancy[pointer[k]]];
    }
    return scalar_product;
}

template<class T> T vector_sparse_dynamic<T>::scalar_product_pos_factors(const vector_sparse_dynamic<T>&y) const { // scalar product using only positive summands
    if(non_fatal_error(dimension() != y.dimension(), "vector_sparse_dynamic::operator * : dimensions must be equal.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(dimension()==0) return 0.0;
    T scalar_product=0.0;
    Integer k;
    for(k=0;k<nnz;k++){
        if (y.occupancy[pointer[k]]>=0 && data[k]*y.data[y.occupancy[pointer[k]]]>0) scalar_product += data[k]*y.data[y.occupancy[pointer[k]]];
    }
    return scalar_product;

}

template<class T> void vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold(Real& norm, index_list& list, Integer n, Real tau) const {
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
    for(i=0;i<nnz;i++){
        norm += absvalue_squared(data[i]);
    }
    norm=sqrt(norm);
    for(i=0;i<nnz;i++){
        if(std::abs(data[i]) > norm*tau){
            input_abs[number_elements_larger_tau] = std::abs(data[i]);
            complete_list[number_elements_larger_tau]=pointer[i];
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we need the indices of the largest elements in ascending order. To get this order, we sort here.
        complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        complete_list.quicksort(0,number_elements_larger_tau-1);
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
}

template<class T> void vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold_largest_last(Real& norm, index_list& list, Integer n, Real tau) const {
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer pos_larg_el;
    Real val_larg_el;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
    for(i=0;i<nnz;i++){
        norm += absvalue_squared(data[i]);
    }
    norm=sqrt(norm);
    for(i=0;i<nnz;i++){
        if(std::abs(data[i]) > norm*tau){
            input_abs[number_elements_larger_tau]=std::abs(data[i]);
            complete_list[number_elements_larger_tau]=pointer[i];
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we do not need the indices of the largest elements in ascending order. To get this order, we sort here.
        //complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        //complete_list.quicksort(0,number_elements_larger_tau-1);
        pos_larg_el=0;
        val_larg_el=0.0;
        for(i=0;i<number_elements_larger_tau;i++){
            if(input_abs.read(i)>val_larg_el){
                pos_larg_el=i;
                val_larg_el=input_abs.read(i);
            }
        }
        if(number_elements_larger_tau>0) complete_list.switch_index(pos_larg_el,number_elements_larger_tau-1);
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
}

template<class T> void vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold_pivot_last(Real& norm, index_list& list, Integer n, Real tau, Integer pivot_position, Real perm_tol) const {
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer pos_larg_el;
    Real val_larg_el=0.0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
    for(i=0;i<nnz;i++){
        if(std::abs(data[i])>val_larg_el) val_larg_el=std::abs(data[i]);
        norm += absvalue_squared(data[i]);
    }
    if(val_larg_el*perm_tol>std::abs(read(pivot_position))){ // do pivoting
        norm=sqrt(norm);
        for(i=0;i<nnz;i++){
            if(std::abs(data[i])> norm*tau){
                input_abs[number_elements_larger_tau]=std::abs(data[i]);
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        }
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we do not need the indices of the largest elements in ascending order. To get this order, we sort here.
            //complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
        } else {
            pos_larg_el=0;
            val_larg_el=0.0;
            for(i=0;i<number_elements_larger_tau;i++){
                if(input_abs.read(i)>val_larg_el){
                    pos_larg_el=i;
                    val_larg_el=input_abs.read(i);
                }
            }
            if(number_elements_larger_tau>0) complete_list.switch_index(pos_larg_el,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
        }
    } else {    // don't pivot
        if(read(pivot_position)==0){
            list.resize_without_initialization(0);
            return;
        }
        norm=sqrt(norm);
        for(i=0;i<nnz;i++){
            if(std::abs(data[i])> norm*tau && pointer[i] != pivot_position){
                input_abs[number_elements_larger_tau]=std::abs(data[i]);
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        }
        if(number_elements_larger_tau > n-1){
            offset=number_elements_larger_tau-n+1;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            list.resize_without_initialization(n);
            for (i=0;i<n-1;i++) list[i]=complete_list[offset+i];
            list[n-1]=pivot_position;
        } else {
            //complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau+1);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
            list[number_elements_larger_tau]=pivot_position;
        }
    }  // end if "to pivot or not to pivot"
}



template<class T> void vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold_pivot_last(Real& norm, index_list& list, Integer n, Real tau, Integer pivot_position) const {
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer pos_larg_el;
    Real val_larg_el=0.0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
    for(i=0;i<nnz;i++){
        if(std::abs(data[i])>val_larg_el) val_larg_el=std::abs(data[i]);
        norm += absvalue_squared(data[i]);
    }
    norm=sqrt(norm);
    for(i=0;i<nnz;i++){
        if(std::abs(data[i])> norm*tau){
            input_abs[number_elements_larger_tau]=std::abs(data[i]);
            complete_list[number_elements_larger_tau]=pointer[i];
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we do not need the indices of the largest elements in ascending order. To get this order, we sort here.
        //complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        //complete_list.quicksort(0,number_elements_larger_tau-1);
        pos_larg_el=0;
        val_larg_el=0.0;
        for(i=0;i<number_elements_larger_tau;i++){
            if(input_abs.read(i)>val_larg_el){
                pos_larg_el=i;
                val_larg_el=input_abs.read(i);
            }
        }
        if(number_elements_larger_tau>0) complete_list.switch_index(pos_larg_el,number_elements_larger_tau-1);
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
}


template<class T> void vector_sparse_dynamic<T>::take_weighted_largest_elements_by_abs_value_with_threshold_pivot_last(Real& norm, index_list& list, const vector_dense<Real>& weights, Integer n, Real tau, Integer pivot_position, Real perm_tol) const {
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer pos_larg_el;
    Real val_larg_el=0.0;
    Real product;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
    for(i=0;i<nnz;i++){
        if(std::abs(data[i])>val_larg_el) val_larg_el=std::abs(data[i]);
        norm += absvalue_squared(weights.read(pointer[i])*data[i]);
    }
    if(val_larg_el*perm_tol>std::abs(read(pivot_position))){ // do pivoting
        norm=sqrt(norm);
        for(i=0;i<nnz;i++){
            product=std::abs(data[i])*weights.read(pointer[i]);
            if(product> norm*tau){
                input_abs[number_elements_larger_tau]=product;
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        }
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we do not need the indices of the largest elements in ascending order. To get this order, we sort here.
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
        } else {
            pos_larg_el=0;
            val_larg_el=0.0;
            for(i=0;i<number_elements_larger_tau;i++){
                if(input_abs.read(i)>val_larg_el){
                    pos_larg_el=i;
                    val_larg_el=input_abs.read(i);
                }
            }
            if(number_elements_larger_tau>0) complete_list.switch_index(pos_larg_el,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
        }
    } else {    // don't pivot
        if(read(pivot_position)==0){
            list.resize_without_initialization(0);
            return;
        }
        norm=sqrt(norm);
        for(i=0;i<nnz;i++){
            if(std::abs(data[i])*weights.read(pointer[i])> norm*tau && pointer[i] != pivot_position){
                input_abs[number_elements_larger_tau]=std::abs(data[i]);
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        }
        if(number_elements_larger_tau > n-1){
            offset=number_elements_larger_tau-n+1;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we do not need the indices of the largest elements in ascending order. To get this order, we sort here.
            list.resize_without_initialization(n);
            for (i=0;i<n-1;i++) list[i]=complete_list[offset+i];
            list[n-1]=pivot_position;
        } else {
            //complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau+1);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
            list[number_elements_larger_tau]=pivot_position;
        }
    }  // end if "to pivot or not to pivot"
}

template<class T> void vector_sparse_dynamic<T>::take_single_weight_largest_elements_by_abs_value_with_threshold_pivot_last(index_list& list, vector_dense<Real>& weights, Integer n, Real tau, Integer pivot_position, Real perm_tol) const
{
    Real norm=0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer pos_larg_el=0;
    Real val_larg_el=0.0;
    Real product,weight;
    Real xiplus, ximinus;
    index_list complete_list;
    vector_dense<Real> input_abs;
        if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
        if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
        for(i=0;i<nnz;i++){
            if(std::abs(data[i])>val_larg_el){
                val_larg_el=std::abs(data[i]);
                pos_larg_el=pointer[i]; // position in *this
            }
            norm += absvalue_squared(data[i]);
        }
        if(val_larg_el*perm_tol>std::abs(read(pivot_position))){ // do pivoting
            xiplus=1.0+weights[pos_larg_el];
            ximinus=-1.0+weights[pos_larg_el];
            if(std::abs(xiplus)<std::abs(ximinus)) weights[pos_larg_el] = ximinus/read(pos_larg_el);
            else weights[pos_larg_el] = xiplus/read(pos_larg_el);
            weight=std::abs(weights[pos_larg_el]);
            for(i=0;i<nnz;i++){
                product=std::abs(data[i]*weight);
                if(product/std::abs(read(pos_larg_el))>tau){
                    input_abs[number_elements_larger_tau]=product;
                    complete_list[number_elements_larger_tau]=pointer[i];
                    number_elements_larger_tau++;
                }
            }
            if(number_elements_larger_tau==0){
                for(i=0;i<nnz;i++){
                    input_abs[i]=std::abs(data[i]*weight);
                    complete_list[i]=pointer[i];
                }   // end for
                number_elements_larger_tau=nnz;
            } // end if
            if(number_elements_larger_tau > n){
                offset=number_elements_larger_tau-n;
                input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
                // we do not need the indices of the largest elements in ascending order. To get this order, we sort here.
                list.resize_without_initialization(n);
                for (i=0;i<n;i++) list[i]=complete_list[offset+i];
            } else {
                pos_larg_el=0;
                val_larg_el=0.0;
                for(i=0;i<number_elements_larger_tau;i++){
                    if(std::abs(read(complete_list[i]))>val_larg_el){
                        val_larg_el=std::abs(read(complete_list[i]));
                        pos_larg_el=i;                 // position in complete_list
                   }
                }
                if(number_elements_larger_tau>0) complete_list.switch_index(pos_larg_el,number_elements_larger_tau-1);
                list.resize_without_initialization(number_elements_larger_tau);
                for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
            }
        } else {    // don't pivot
            if(read(pivot_position)==0.0){
                list.resize_without_initialization(0);
                #ifdef DEBUG
                    std::cerr<<"vector_sparse_dynamic<T>::take_single_weight_largest_elements_by_abs_value_with_threshold_pivot_last: returning an empty list. Pivoting was not performed."<<std::endl;
                #endif
                return;
            }
            xiplus=1.0+weights[pivot_position];
            ximinus=-1.0+weights[pivot_position];
            if(std::abs(xiplus)<std::abs(ximinus)) weights[pivot_position] = ximinus/read(pivot_position);
            else weights[pivot_position] = xiplus/read(pivot_position);
            weight=weights[pivot_position];
            for(i=0;i<nnz;i++){
                if(std::abs(data[i]*weight/read(pivot_position)) > tau && pointer[i] != pivot_position){
                    input_abs[number_elements_larger_tau]=std::abs(data[i]);
                    complete_list[number_elements_larger_tau]=pointer[i];
                    number_elements_larger_tau++;
                }
            }
            if(number_elements_larger_tau > n-1){
                offset=number_elements_larger_tau-n+1;
                input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
                // we do not need the indices of the largest elements in ascending order. To get this order, we sort here.
                list.resize_without_initialization(n);
                for (i=0;i<n-1;i++) list[i]=complete_list[offset+i];
                list[n-1]=pivot_position;
            } else {
                //complete_list.quicksort(0,number_elements_larger_tau-1);
                list.resize_without_initialization(number_elements_larger_tau+1);
                for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
                list[number_elements_larger_tau]=pivot_position;
            }
        }  // end if "to pivot or not to pivot"
}

template<class T> void vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold(Real& norm, index_list& list, Integer n, Real tau, Integer from, Integer to) const
{
    norm = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
    #ifdef DEBUG
        if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    for(i=0;i<nnz;i++){
        if(from<=pointer[i] && pointer[i]<to)
            norm += absvalue_squared(data[i]);
    }
    norm=sqrt(norm);
    for(i=0;i<nnz;i++){
        if(from<=pointer[i] && pointer[i]<to && std::abs(data[i])> norm*tau){
            input_abs[number_elements_larger_tau]=std::abs(data[i]);
            complete_list[number_elements_larger_tau]=pointer[i];
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
           // we need the indices of the largest elements in ascending order. To get this order, we sort here.
        complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        complete_list.quicksort(0,number_elements_larger_tau-1);
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
}


template<class T> void vector_sparse_dynamic<T>::take_weighted_largest_elements_by_abs_value_with_threshold(Real& norm, index_list& list, const vector_dense<Real>& weights, Integer n, Real tau, Integer from, Integer to) const {
    norm = 0.0;
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    for(i=0;i<nnz;i++){
        if(from<=pointer[i] && pointer[i]<to)
            norm += absvalue_squared(weights.read(pointer[i])*data[i]);
    }
    norm=sqrt(norm);
    for(i=0;i<nnz;i++){
        product = weights.read(pointer[i])* std::abs(data[i]);
        if(from<=pointer[i] && pointer[i]<to && product > norm*tau){
            input_abs[number_elements_larger_tau]=product;
            complete_list[number_elements_larger_tau]=pointer[i];
            number_elements_larger_tau++;
        }
    }
    if(number_elements_larger_tau > n){
        offset=number_elements_larger_tau-n;
        input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
        // we need the indices of the largest elements in ascending order. To get this order, we sort here.
        complete_list.quicksort(offset,number_elements_larger_tau-1);
        list.resize_without_initialization(n);
        for (i=0;i<n;i++) list[i]=complete_list[offset+i];
    } else {
        complete_list.quicksort(0,number_elements_larger_tau-1);
        list.resize_without_initialization(number_elements_larger_tau);
        for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
    }
  }

template<class T> void vector_sparse_dynamic<T>::take_single_weight_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter& IP, index_list& list, Real weight, Integer n, Real tau, Integer from, Integer to) const {
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            product = weight * std::abs(data[i]);
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        list.resize_without_initialization(nnz-offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            product = weight * std::abs(data[i]);
            if(from<=pointer[i] && pointer[i]<to && (product >= tau) ){
                input_abs[number_elements_larger_tau]=product;
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
        }
    } // end WEIGHTED_DROPPING
  }

template<class T> void vector_sparse_dynamic<T>::take_single_weight_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter& IP, index_list& list, index_list& rejected_list, Real weight, Integer n, Real tau, Integer from, Integer to) const {
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer rejected_number=0;
    index_list complete_list, complete_rejected_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (complete_rejected_list.dimension() != size) complete_rejected_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            product = weight * std::abs(data[i]);
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        complete_list.quicksort(0,offset-1);
        list.resize_without_initialization(nnz-offset);
        rejected_list.resize_without_initialization(offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
        for (i=0;i<offset;i++) rejected_list[i]=complete_list[i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            if(from<=pointer[i] && pointer[i]<to){  // in the proper range
                product = weight * std::abs(data[i]);
                if(product >= tau){  // mark to keep
                    input_abs[number_elements_larger_tau]=product;
                    complete_list[number_elements_larger_tau]=pointer[i];
                    number_elements_larger_tau++;
                } else { // mark as rejected
                    complete_rejected_list[rejected_number]=pointer[i];
                    rejected_number++;
                }
            }  // end in proper range
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
            rejected_list.resize_without_initialization(nnz-n); // note: nnz-n >= number_elements_larger_tau-n > 0) (note: nnz = number_elements_larger_tau + rejected_number)
            for(i=0;i<number_elements_larger_tau-n;i++) rejected_list[i] = complete_list[i];  // copy indices of rejected elements larger than tau
            for(i=0;i<rejected_number;i++) rejected_list[number_elements_larger_tau-n+i] = complete_rejected_list[i]; // copy indices of rejected elements less than tau
            rejected_list.quicksort(0,nnz-n-1); // sort all
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
            complete_rejected_list.quicksort(0,rejected_number-1);
            rejected_list.resize_without_initialization(rejected_number);
            for(i=0;i<rejected_number;i++) rejected_list[i]=complete_rejected_list[i];
        }
    } // end WEIGHTED_DROPPING
  }



template<class T> void vector_sparse_dynamic<T>::take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter& IP, index_list& list, const vector_dense<Real>& weights, Real weight, Integer n, Real tau, Integer from, Integer to) const  {
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic:::take_single_weight_weighted_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            //product = (weight + weights.read(pointer[i])) * std::abs(data[i]);
            product = max(weight,weights.read(pointer[i])) * std::abs(data[i]);
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        list.resize_without_initialization(nnz-offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            //product = (weight + weights.read(pointer[i])) * std::abs(data[i]);
            product = max(weight,weights.read(pointer[i])) * std::abs(data[i]);
            if(from<=pointer[i] && pointer[i]<to && (product >= tau) ){
                input_abs[number_elements_larger_tau]=product;
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
        }
    } // end WEIGHTED_DROPPING
  }

template<class T> void vector_sparse_dynamic<T>::take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter& IP, index_list& list, index_list& rejected_list, const vector_dense<Real>& weights, Real weight, Integer n, Real tau, Integer from, Integer to) const  {
#ifdef DEBUG
    std::cout<<"vector_sparse_dynamic::take_single_weight_weighted_largest_elements_by_abs_value_with_threshold: this routine has not been tested.... "<<std::endl<<std::flush;
#endif
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer rejected_number=0;
    index_list complete_list, complete_rejected_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (complete_rejected_list.dimension() != size) complete_rejected_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "take_single_weight_weighted_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            //product = (weight + weights.read(pointer[i])) * std::abs(data[i]);
            product = max(weight,weights.read(pointer[i])) * std::abs(data[i]);
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        complete_list.quicksort(0,offset-1);
        list.resize_without_initialization(nnz-offset);
        rejected_list.resize_without_initialization(offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
        for (i=0;i<offset;i++) rejected_list[i]=complete_list[i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            //product = (weight + weights.read(pointer[i])) * std::abs(data[i]);
            product = max(weight,weights.read(pointer[i])) * std::abs(data[i]);
            if(from<=pointer[i] && pointer[i]<to){ // in right range
                if(product >= tau){ // mark to keep
                    input_abs[number_elements_larger_tau]=product;
                    complete_list[number_elements_larger_tau]=pointer[i];
                    number_elements_larger_tau++;
                } else {  // mark to reject
                    complete_rejected_list[rejected_number]=pointer[i];
                    rejected_number++;
                }
            }
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
            rejected_list.resize_without_initialization(nnz-n); // note: nnz-n >= number_elements_larger_tau-n > 0) (note: nnz = number_elements_larger_tau + rejected_number)
            for(i=0;i<number_elements_larger_tau-n;i++) rejected_list[i] = complete_list[i];  // copy indices of rejected elements larger than tau
            for(i=0;i<rejected_number;i++) rejected_list[number_elements_larger_tau-n+i] = complete_rejected_list[i]; // copy indices of rejected elements less than tau
            rejected_list.quicksort(0,nnz-n-1); // sort all
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
            complete_rejected_list.quicksort(0,rejected_number-1);
            rejected_list.resize_without_initialization(rejected_number);
            for(i=0;i<rejected_number;i++) rejected_list[i]=complete_rejected_list[i];
        }
    } // end WEIGHTED_DROPPING
  }


template<class T> void vector_sparse_dynamic<T>::take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter &IP, index_list& list, Real weight, Integer n, Real tau, Integer from, Integer to, Integer vector_index, Integer max_pos_drop) const {
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
            if(pointer[i]<=max_pos_drop) 
                product = weight * std::abs(data[i]) * IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/size);
            //product = max(weight,TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]) * std::abs(data[i]);
            //else  product = weight * std::abs(data[i]);
            else  product = weight * std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        list.resize_without_initialization(nnz-offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
            if(pointer[i]<=max_pos_drop) 
                //product = max(weight,TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]) * std::abs(data[i]); // Abstand von der Diagonalen
                product = weight * std::abs(data[i]) * IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/size); // Abstand von der Diagonalen
            //else  product = weight * std::abs(data[i]);
            else  product =weight * std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
            if(from<=pointer[i] && pointer[i]<to && (product >= tau) ){
                input_abs[number_elements_larger_tau]=product;
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
        }
    } // end WEIGHTED_DROPPING
}


template<class T> void vector_sparse_dynamic<T>::take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter &IP, index_list& list, index_list& rejected_list, Real weight, Integer n, Real tau, Integer from, Integer to, Integer vector_index, Integer max_pos_drop) const {
#ifdef DEBUG
    std::cout<<"vector_sparse_dynamic::take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold: note: this routine has not been tested..."<<std::endl<<std::flush;
#endif
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer rejected_number=0;
    index_list complete_list, complete_rejected_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (complete_rejected_list.dimension() != size) complete_rejected_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
            if(pointer[i]<=max_pos_drop) 
                product = weight * std::abs(data[i]) * IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/size);
            //product = max(weight,TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]) * std::abs(data[i]);
            //else  product = weight * std::abs(data[i]);
            else  product = weight * std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        complete_list.quicksort(0,offset-1);
        list.resize_without_initialization(nnz-offset);
        rejected_list.resize_without_initialization(offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
        for (i=0;i<offset;i++) rejected_list[i]=complete_list[i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            if(from<=pointer[i] && pointer[i]<to){  // if in proper range
                //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
                if(pointer[i]<=max_pos_drop)
                    //product = max(weight,TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]) * std::abs(data[i]); // Abstand von der Diagonalen
                    product = weight * std::abs(data[i]) * IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/size); // Abstand von der Diagonalen
                //else  product = weight * std::abs(data[i]);
                else  product =weight * std::abs(data[i]);
                //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
                if(product >= tau){ // mark to keep
                    input_abs[number_elements_larger_tau]=product;
                    complete_list[number_elements_larger_tau]=pointer[i];
                    number_elements_larger_tau++;
                } else { // mark to reject
                    complete_rejected_list[rejected_number]=pointer[i];
                    rejected_number++;
                }
            } // end of proper range
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
            rejected_list.resize_without_initialization(nnz-n); // note: nnz-n >= number_elements_larger_tau-n > 0) (note: nnz = number_elements_larger_tau + rejected_number)
            for(i=0;i<number_elements_larger_tau-n;i++) rejected_list[i] = complete_list[i];  // copy indices of rejected elements larger than tau
            for(i=0;i<rejected_number;i++) rejected_list[number_elements_larger_tau-n+i] = complete_rejected_list[i]; // copy indices of rejected elements less than tau
            rejected_list.quicksort(0,nnz-n-1); // sort all
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
            complete_rejected_list.quicksort(0,rejected_number-1);
            rejected_list.resize_without_initialization(rejected_number);
            for(i=0;i<rejected_number;i++) rejected_list[i]=complete_rejected_list[i];
        }
    } // end WEIGHTED_DROPPING
}


template<class T> void vector_sparse_dynamic<T>::take_single_weight_bw_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter &IP, index_list& list, Real weight, Integer n, Real tau, Integer from, Integer to, Integer vector_index,Integer bandwidth, Integer max_pos_drop) const {
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    index_list complete_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_single_weight_bw_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
            if(pointer[i]<=max_pos_drop) 
                product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i])*IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/bandwidth);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size];
            //product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i]);
            //else  product = weight * std::abs(data[i]);
            else  product = weight * std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        list.resize_without_initialization(nnz-offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
            if(pointer[i]<=max_pos_drop) 
                product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i])*IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/bandwidth);
            //product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]; // Abstand von der Diagonalen
            //product = weight * std::abs(data[i]) * ; // Abstand von der Diagonalen
            //else  product = weight * std::abs(data[i]);
            else  product =weight * std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
            if(from<=pointer[i] && pointer[i]<to && (product >= tau) ){
                input_abs[number_elements_larger_tau]=product;
                complete_list[number_elements_larger_tau]=pointer[i];
                number_elements_larger_tau++;
            }
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
        }
    }
}

template<class T> void vector_sparse_dynamic<T>::take_single_weight_bw_largest_elements_by_abs_value_with_threshold(const iluplusplus_precond_parameter &IP, index_list& list, index_list& rejected_list, Real weight, Integer n, Real tau, Integer from, Integer to, Integer vector_index,Integer bandwidth, Integer max_pos_drop) const {
#ifdef DEBUG
    std::cout<<"vector_sparse_dynamic::take_single_weight_bw_largest_elements_by_abs_value_with_threshold: this routine has not been tested yet..."<<std::endl<<std::flush;
#endif
    Real product = 0.0;
    Integer offset=0;
    Integer i;
    Integer number_elements_larger_tau=0;
    Integer rejected_number=0;
    index_list complete_list, complete_rejected_list;
    vector_dense<Real> input_abs;
    Real sum = 0.0;
    if (complete_list.dimension() != size) complete_list.resize_without_initialization(size);
    if (complete_rejected_list.dimension() != size) complete_rejected_list.resize_without_initialization(size);
    if (input_abs.dimension() != size) input_abs.erase_resize_data_field(size);
#ifdef DEBUG
    if(non_fatal_error((from<0 || to>size), "vector_sparse_dynamic::take_single_weight_bw_largest_elements_by_abs_value_with_threshold: sorting range is not permitted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    if(IP.get_SUM_DROPPING()){
        for(i=0;i<nnz;i++){
            //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
            if(pointer[i]<=max_pos_drop) 
                product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i])*IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/bandwidth);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size];
            //product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i]);
            //else  product = weight * std::abs(data[i]);
            else  product = weight * std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs.read(i);
            if (sum > tau){
                offset = i;
                break;
            }
        }
        if(nnz-offset>n) offset=nnz-n;
        complete_list.quicksort(offset,nnz-1);
        list.resize_without_initialization(nnz-offset);
        rejected_list.resize_without_initialization(offset);
        for (i=0;i<nnz-offset;i++) list[i]=complete_list[offset+i];
        for (i=0;i<offset;i++) rejected_list[i]=complete_list[i];
    } // end SUM_DROPPING
    if(IP.get_WEIGHTED_DROPPING()){
        for(i=0;i<nnz;i++){ // mark elements to be kept
            //product = max(std::abs(data[i]),weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]);
            if(pointer[i]<=max_pos_drop) 
                product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i])*IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(pointer[i]-vector_index))/bandwidth);
            //product = (abs(pointer[i]-vector_index)>bandwidth) ?  0.0 : weight*std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]-vector_index))/size]; // Abstand von der Diagonalen
            //product = weight * std::abs(data[i]) * ; // Abstand von der Diagonalen
            //else  product = weight * std::abs(data[i]);
            else  product =weight * std::abs(data[i]);
            //product = weight * std::abs(data[i]) * TABLE_POSITIONAL_WEIGHTS[(SIZE_TABLE_POS_WEIGHTS*abs(pointer[i]))/size];
            if((from<=pointer[i] && pointer[i]<to) ){ // in proper range
                if(product >= tau){ // mark to keep
                    input_abs[number_elements_larger_tau]=product;
                    complete_list[number_elements_larger_tau]=pointer[i];
                    number_elements_larger_tau++;
                } else {  // mark to reject
                    complete_rejected_list[rejected_number]=pointer[i];
                    rejected_number++;
                }
            }  // end proper range
        } // end marking elements
        if(number_elements_larger_tau > n){
            offset=number_elements_larger_tau-n;
            input_abs.sort(complete_list,0,number_elements_larger_tau-1,n);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            // we need the indices of the largest elements in ascending order. To get this order, we sort here.
            complete_list.quicksort(offset,number_elements_larger_tau-1);
            list.resize_without_initialization(n);
            for (i=0;i<n;i++) list[i]=complete_list[offset+i];
            rejected_list.resize_without_initialization(nnz-n); // note: nnz-n >= number_elements_larger_tau-n > 0) (note: nnz = number_elements_larger_tau + rejected_number)
            for(i=0;i<number_elements_larger_tau-n;i++) rejected_list[i] = complete_list[i];  // copy indices of rejected elements larger than tau
            for(i=0;i<rejected_number;i++) rejected_list[number_elements_larger_tau-n+i] = complete_rejected_list[i]; // copy indices of rejected elements less than tau
            rejected_list.quicksort(0,nnz-n-1); // sort all
        } else {
            complete_list.quicksort(0,number_elements_larger_tau-1);
            list.resize_without_initialization(number_elements_larger_tau);
            for(i=0;i<number_elements_larger_tau;i++) list[i]=complete_list[i];
            complete_rejected_list.quicksort(0,rejected_number-1);
            rejected_list.resize_without_initialization(rejected_number);
            for(i=0;i<rejected_number;i++) rejected_list[i]=complete_rejected_list[i];
        }
    } // end WEIGHTED_DROPPING
}

template<class T> void vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold(Real& norm_input_L, Real& norm_input_U, index_list& list_L, index_list& list_U, const index_list& invperm, Integer n_L, Integer n_U, Real tau_L,  Real tau_U, Integer mid) const {
    norm_input_L = 0.0;
    norm_input_U = 0.0;
    Integer i;
    Integer number_elements_larger_tau_L=0;
    Integer number_elements_larger_tau_U=0;
    Integer offset=0;
    Integer pos_larg_element=0;
    index_list complete_list_L;
    index_list complete_list_U;
    vector_dense<Real> input_abs_L;
    vector_dense<Real> input_abs_U;
    if (complete_list_L.dimension() != size) complete_list_L.resize_without_initialization(size);
    if (complete_list_U.dimension() != size) complete_list_U.resize_without_initialization(size);
    if (input_abs_L.dimension() != size) input_abs_L.erase_resize_data_field(size);
    if (input_abs_U.dimension() != size) input_abs_U.erase_resize_data_field(size);
#ifdef DEBUG
    if(size==0){
        std::cerr<<"vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold: size=0: returning empty list"<<std::endl;
        list_L.resize_without_initialization(0);
        list_U.resize_without_initialization(0);
        return;
    }
#endif
    for(i=0;i<nnz;i++)
        if (invperm[pointer[i]]<mid) norm_input_L += absvalue_squared(data[i]);
        else norm_input_U += absvalue_squared(data[i]);
    norm_input_L=sqrt(norm_input_L);
    norm_input_U=sqrt(norm_input_U);
    for(i=0;i<nnz;i++){
        if(invperm[pointer[i]]<mid){
            if(std::abs(data[i]) > norm_input_L*tau_L){
                input_abs_L[number_elements_larger_tau_L]=std::abs(data[i]);
                //complete_list_L[number_elements_larger_tau_L]=invperm[pointer[i]];
                complete_list_L[number_elements_larger_tau_L]=pointer[i];
                number_elements_larger_tau_L++;
            }
        } else {
            if(std::abs(data[i]) > norm_input_U*tau_U){
                input_abs_U[number_elements_larger_tau_U]=std::abs(data[i]);
                //complete_list_U[number_elements_larger_tau_U]=invperm[pointer[i]]; // the true index in w
                complete_list_U[number_elements_larger_tau_U]=pointer[i]; // the true index in w
                number_elements_larger_tau_U++;
            }

        }
    }
    if(number_elements_larger_tau_L==0){
        list_L.resize(0);
    } else {
        if(number_elements_larger_tau_L > n_L){
            offset=number_elements_larger_tau_L-n_L;
            input_abs_L.sort(complete_list_L,0,number_elements_larger_tau_L-1,n_L);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            list_L.resize_without_initialization(n_L);
            for (i=0;i<list_L.dimension();i++) list_L[i]=complete_list_L[offset+i];
        } else {
            list_L.resize_without_initialization(number_elements_larger_tau_L);
            for(i=0;i<number_elements_larger_tau_L;i++) list_L[i]=complete_list_L[i];
            //list_L.switch_index(pos_larg_element,list_L.dimension()-1);
        }  // end if
    } // end if
    if(number_elements_larger_tau_U==0){
        list_U.resize(0);
    } else {
        if(number_elements_larger_tau_U > n_U){
            offset=number_elements_larger_tau_U-n_U;
            //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            input_abs_U.sort(complete_list_U,0,number_elements_larger_tau_U-1,n_U);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            pos_larg_element=offset;
            for(i=offset+1;i<number_elements_larger_tau_U;i++)
                if(input_abs_U.read(i)>input_abs_U.read(pos_larg_element))
                    pos_larg_element=i;
            complete_list_U.switch_index(pos_larg_element,number_elements_larger_tau_U-1);
            list_U.resize_without_initialization(n_U);
            for (i=0;i<list_U.dimension();i++) list_U[i]=complete_list_U[offset+i];
        } else {
            pos_larg_element=0;
            //if(number_elements_larger_tau_U>0)  //
            for(i=1;i<number_elements_larger_tau_U;i++)
                if(input_abs_U.read(i)>input_abs_U.read(pos_larg_element))
                    pos_larg_element=i;
            complete_list_U.switch_index(pos_larg_element,number_elements_larger_tau_U-1);            
            list_U.resize_without_initialization(number_elements_larger_tau_U);
            for(i=0;i<number_elements_larger_tau_U;i++) list_U[i]=complete_list_U[i];            
        }  // end if
    } // end if
}


template<class T> void vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold(Real& norm_input_L, Real& norm_input_U, index_list& list_L, index_list& list_U, const index_list& invperm, Integer n_L, Integer n_U, Real tau_L,  Real tau_U, Integer mid, Real piv_tol) const {
    norm_input_L = 0.0;
    bool pivoting;
    Real val_larg_element=0.0;
    Real val_pot_pivot=0.0;
    norm_input_U = 0.0;
    Integer i;
    Integer number_elements_larger_tau_L=0;
    Integer number_elements_larger_tau_U=0;
    Integer offset=0;
    Integer pos_larg_element=0;
    Integer pos_pot_pivot=-1;
    index_list complete_list_L;
    index_list complete_list_U;
    vector_dense<Real> input_abs_L;
    vector_dense<Real> input_abs_U;
    vector_dense<T> fabsdata;
    fabsdata.erase_resize_data_field(nnz);
    if (complete_list_L.dimension() != size) complete_list_L.resize_without_initialization(size);
    if (complete_list_U.dimension() != size) complete_list_U.resize_without_initialization(size);
    if (input_abs_L.dimension() != size) input_abs_L.erase_resize_data_field(size);
    if (input_abs_U.dimension() != size) input_abs_U.erase_resize_data_field(size);
#ifdef DEBUG
    if(size==0){
        std::cerr<<"vector_sparse_dynamic<T>::take_largest_elements_by_abs_value_with_threshold: size=0: returning empty list"<<std::endl;
        list_L.resize_without_initialization(0);
        list_U.resize_without_initialization(0);
        return;
    }
#endif
    for(i=0;i<nnz;i++){
        fabsdata[i]=std::abs(data[i]);
        if (invperm[pointer[i]]<mid) norm_input_L += absvalue_squared(data[i]);
        else if (invperm[pointer[i]]>mid){ 
            norm_input_U += absvalue_squared(data[i]);
            if(fabsdata[i] > val_larg_element) val_larg_element = fabsdata[i];
        } else {
            norm_input_U += absvalue_squared(data[i]);
            if(fabsdata[i] > val_larg_element) val_larg_element = fabsdata[i];
            val_pot_pivot = fabsdata[i];
            pos_pot_pivot = i;
        }
    }
    norm_input_L=sqrt(norm_input_L);
    norm_input_U=sqrt(norm_input_U);
    pivoting = ( (val_larg_element*piv_tol >= val_pot_pivot) || (pos_pot_pivot < 0) ); // if true, pivot needs to be last, else the //std::cout<<"this (i.e. w)"<<std::endl<<expand();
    if (!pivoting) fabsdata[pos_pot_pivot] =  norm_input_U; // this ensures that this element is selected and moved to the end when //std::cout<<"paramater: tau_L: "<<tau_L<<" tau_U "<<tau_U<<std::endl;
    for(i=0;i<nnz;i++){
        if(invperm[pointer[i]]<mid){
            if(fabsdata[i] > norm_input_L*tau_L){
                input_abs_L[number_elements_larger_tau_L]=fabsdata[i];
                //complete_list_L[number_elements_larger_tau_L]=invperm[pointer[i]];
                complete_list_L[number_elements_larger_tau_L]=pointer[i];
                number_elements_larger_tau_L++;
            }
        } else {
            if(fabsdata[i] > norm_input_U*tau_U){
                input_abs_U[number_elements_larger_tau_U]=fabsdata[i];
                //complete_list_U[number_elements_larger_tau_U]=invperm[pointer[i]]; // the true index in w
                complete_list_U[number_elements_larger_tau_U]=pointer[i]; // the true index in w
                number_elements_larger_tau_U++;
            }
        }
    }
    if(number_elements_larger_tau_L==0){
        list_L.resize(0);
    } else {
        if(number_elements_larger_tau_L > n_L){
            offset=number_elements_larger_tau_L-n_L;
            input_abs_L.sort(complete_list_L,0,number_elements_larger_tau_L-1,n_L);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            list_L.resize_without_initialization(n_L);
            for (i=0;i<list_L.dimension();i++) list_L[i]=complete_list_L[offset+i];
        } else {
            list_L.resize_without_initialization(number_elements_larger_tau_L);
            for(i=0;i<number_elements_larger_tau_L;i++) list_L[i]=complete_list_L[i];
            //list_L.switch_index(pos_larg_element,list_L.dimension()-1);
        }  // end if
    } // end if
    if(number_elements_larger_tau_U==0){
        list_U.resize(0);
    } else {
        if(number_elements_larger_tau_U > n_U){
            offset=number_elements_larger_tau_U-n_U;
            input_abs_U.sort(complete_list_U,0,number_elements_larger_tau_U-1,n_U);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            pos_larg_element=offset;
            for(i=offset+1;i<number_elements_larger_tau_U;i++)
                if(input_abs_U.read(i)>input_abs_U.read(pos_larg_element))
                    pos_larg_element=i;
            complete_list_U.switch_index(pos_larg_element,number_elements_larger_tau_U-1);
            list_U.resize_without_initialization(n_U);
            for (i=0;i<list_U.dimension();i++) list_U[i]=complete_list_U[offset+i];
        } else {
            pos_larg_element=0;
            for(i=1;i<number_elements_larger_tau_U;i++)
                if(input_abs_U.read(i)>input_abs_U.read(pos_larg_element))
                    pos_larg_element=i;
            complete_list_U.switch_index(pos_larg_element,number_elements_larger_tau_U-1);
            list_U.resize_without_initialization(number_elements_larger_tau_U);
            for(i=0;i<number_elements_larger_tau_U;i++) list_U[i]=complete_list_U[i];
        }  // end if
    } // end if
}

template<class T> void vector_sparse_dynamic<T>::take_weighted_largest_elements_by_abs_value_with_threshold(Real& norm_input_L, Real& norm_input_U, index_list& list_L, index_list& list_U, const index_list& perm, const index_list& invperm, const vector_dense<Real>& weights_L, Integer n_L, Integer n_U, Real tau_L,  Real tau_U, Integer mid) const {
    norm_input_L = 0.0;
    norm_input_U = 0.0;
    Integer i;
    Integer number_elements_larger_tau_L=0;
    Integer number_elements_larger_tau_U=0;
    Integer offset=0;
    Integer pos_larg_element=0;
    Real value_larg_element, value;
    Real product = 0.0;
    index_list complete_list_L;
    index_list complete_list_U;
    vector_dense<Real> input_abs_L;
    vector_dense<Real> input_abs_U;
    if (complete_list_L.dimension() != size) complete_list_L.resize_without_initialization(size);
    if (complete_list_U.dimension() != size) complete_list_U.resize_without_initialization(size);
    if (input_abs_L.dimension() != size) input_abs_L.erase_resize_data_field(size);
    if (input_abs_U.dimension() != size) input_abs_U.erase_resize_data_field(size);
#ifdef DEBUG
    if(size==0){
        std::cerr<<" vector_sparse_dynamic<T>::take_weighted_largest_elements_by_abs_value_with_threshold: size=0: returning empty list"<<std::endl;
        list_L.resize_without_initialization(0);
        list_U.resize_without_initialization(0);
        return;
    }
#endif
    for(i=0;i<nnz;i++)
        if (invperm[pointer[i]]<mid) norm_input_L += absvalue_squared(weights_L.read(invperm[pointer[i]])*data[i]);
        else norm_input_U += absvalue_squared(data[i]);
    norm_input_L=sqrt(norm_input_L);
    norm_input_U=sqrt(norm_input_U);
    for(i=0;i<nnz;i++){
        if(invperm[pointer[i]]<mid){
            product = std::abs(weights_L.read(invperm[pointer[i]])*data[i]);
            if (product>tau_L*norm_input_L){
                input_abs_L[number_elements_larger_tau_L]=product;
                //complete_list_L[number_elements_larger_tau_L]=invperm[pointer[i]];
                complete_list_L[number_elements_larger_tau_L]=pointer[i];
                number_elements_larger_tau_L++;
            }
        } else {
            if(std::abs(data[i])>tau_U*norm_input_U){
                input_abs_U[number_elements_larger_tau_U]=std::abs(data[i]);
                //complete_list_U[number_elements_larger_tau_U]=invperm[pointer[i]]; // the true index in w
                complete_list_U[number_elements_larger_tau_U]=pointer[i]; // the true index in w
                number_elements_larger_tau_U++;
            }

        }
    }
    if(number_elements_larger_tau_L==0){
        list_L.resize(0);
    } else {
        if(number_elements_larger_tau_L > n_L){
            offset=number_elements_larger_tau_L-n_L;
            //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            input_abs_L.sort(complete_list_L,0,number_elements_larger_tau_L-1,n_L);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            list_L.resize_without_initialization(n_L);
            for (i=0;i<list_L.dimension();i++) list_L[i]=complete_list_L[offset+i];
        } else {
            list_L.resize_without_initialization(number_elements_larger_tau_L);
            for(i=0;i<number_elements_larger_tau_L;i++) list_L[i]=complete_list_L[i];
        }  // end if
    } // end if
    if(number_elements_larger_tau_U==0){
        list_U.resize(0);
    } else {
        if(number_elements_larger_tau_U > n_U){
            offset=number_elements_larger_tau_U-n_U;
            //input_abs.quicksort(complete_list,0,number_elements_larger_tau-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            input_abs_U.sort(complete_list_U,0,number_elements_larger_tau_U-1,n_U);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            //input_abs_U.quicksort(complete_list_U,0,number_elements_larger_tau_U-1);   // sort abs. vector created above from small to large. Largest elements needed are at end.
            list_U.resize_without_initialization(n_U);
            for (i=0;i<list_U.dimension();i++) list_U[i]=complete_list_U[offset+i];
        } else {
            list_U.resize_without_initialization(number_elements_larger_tau_U);
            for(i=0;i<number_elements_larger_tau_U;i++) list_U[i]=complete_list_U[i];
        }  // end if
    } // end if
    // move largest element to the end of list to be the new pivot
    if(list_U.dimension()>0){
        pos_larg_element=0;
        value_larg_element=std::abs(read(list_U[0]));
        for(i=1;i<list_U.dimension();i++){
            value=std::abs(read(list_U[i]));
            if(value>value_larg_element){
                pos_larg_element=i;
                value_larg_element=value;
            } // end if
        } // end for
        list_U.switch_index(pos_larg_element,list_U.dimension()-1);
    }
}




template<class T>  Real vector_sparse_dynamic<T>::memory() const{
    return (Real) (sizeof(T)+ 2*sizeof(Integer))*size + 2*sizeof(Integer);
}

//************************************************************************************************************************
//                                                                                                                       *
//         The implementation of the class vector_sparse_dynamic_enhanced                                                *
//                                                                                                                       *
//************************************************************************************************************************

//*************************************************************************************************************************************
// Class vector_dense: Constructor, Destructor, etc.                                                                                  *
//*************************************************************************************************************************************

template<class T> vector_sparse_dynamic_enhanced<T>::vector_sparse_dynamic_enhanced(){
    current_position_iter=key.begin();
  }

template<class T> vector_sparse_dynamic_enhanced<T>::~vector_sparse_dynamic_enhanced(){}



template<class T> void vector_sparse_dynamic_enhanced<T>::resize(Integer m) {
    vector_sparse_dynamic<T>::resize(m);
    zero_set();
    current_position_iter=key.begin();
 }

template<class T> vector_sparse_dynamic_enhanced<T>::vector_sparse_dynamic_enhanced(Integer m) {
    this->size = 0;
    this->nnz = 0;
    this->data   = 0;
    this->occupancy = 0;
    this->pointer = 0;
    resize(m);
 }

/*
template<class T> vector_sparse_dynamic_enhanced<T>::vector_sparse_dynamic_enhanced(Integer m) {
    this->size = m;
    this->nnz = 0;
    this->data   = new (std::nothrow) T[m];
    this->occupancy = new (std::nothrow) Integer[m];
    this->pointer = new (std::nothrow) Integer[m];
    if (this->data == 0 || this->occupancy == 0 || this->pointer == 0){
        std::cerr<<"vector_sparse_dynamic_enhanced::vector_sparse_dynamic_enhanced: "<<ippe.error_message()<<std::endl;
        exit(1);
    }
    zero_set();
    current_position_iter=key.begin();
 }
*/

template<class T> vector_sparse_dynamic_enhanced<T>::vector_sparse_dynamic_enhanced(const vector_sparse_dynamic_enhanced& x) {
    Integer i;
    this->size = 0;
    this->nnz = 0;
    this->data   = 0;
    this->occupancy = 0;
    this->pointer = 0;
    resize(x.size);
    for(Integer i=0;i<this->nnz;i++) this->data[i]=x.data[i];
    for(Integer i=0;i<this->nnz;i++) this->pointer[i]=x.pointer[i];
    for(Integer i=0;i<this->nnz;i++) this->occupancy[this->pointer[i]]=x.occupancy[this->pointer[i]];
    key=x.key;
    current_position_iter=x.current_position_iter;
 }


template<class T> vector_sparse_dynamic_enhanced<T>& vector_sparse_dynamic_enhanced<T>::operator =(const vector_sparse_dynamic_enhanced<T>& x){   // Assignment-Operator
    Integer i;
    if(this == &x) return *this;
    resize(x.size);
    for(Integer i=0;i<this->nnz;i++) this->data[i]=x.data[i];
    for(Integer i=0;i<this->nnz;i++) this->pointer[i]=x.pointer[i];
    for(Integer i=0;i<this->nnz;i++) this->occupancy[this->pointer[i]]=x.occupancy[this->pointer[i]];
    key=x.key;
    current_position_iter=x.current_position_iter;
    return *this;
  }

template<class T> void vector_sparse_dynamic_enhanced<T>::zero_reset(){
     Integer i;
     for(i=0;i<this->nnz;i++) this->occupancy[this->pointer[i]]=-1;
     //for(i=0;i<nnz;i++) data[i]=0.0;
     this->nnz=0;
     key.clear();
  }

template<class T> void vector_sparse_dynamic_enhanced<T>::zero_set(){
     Integer i;
     for(i=0;i<this->size;i++) this->occupancy[i]=-1;
     //for(i=0;i<size;i++) data[i]=0.0;
     this->nnz=0;
     key.clear();
  }

template<class T> T& vector_sparse_dynamic_enhanced<T>::operator()(Integer j, Integer k){
     #ifdef DEBUG
        if(j<0 || j>= ((dynamic_cast<vector_sparse_dynamic<T> >(this))->size) || k<0 || k>= ((dynamic_cast<vector_sparse_dynamic<T> >(this))->size) ) {
            std::cout<<"vector_sparse_dynamic_enhanced<T>::operator(): out of range. Trying to access "<<j<<" with sorting index"<<k<<" in a vector having size "<<((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if (this->occupancy[j]<0) {
         this->occupancy[j]=this->nnz;
         key[k]=this->nnz;
         this->pointer[this->nnz]=j;
         this->nnz++;
         this->data[this->occupancy[j]]=0.0;
         return this->data[this->occupancy[j]];
     } else {
         #ifdef DEBUG
             if(this->pointer[key[k]] != j) std::cerr<<"vector_sparse_dynamic<T>::(): sorting index and index do not correspond. Ignoring discrepancy."<<std::endl;
         #endif
         return this->data[this->occupancy[j]];
     }
  }

template<class T> T vector_sparse_dynamic_enhanced<T>::read(Integer j) const {
     #ifdef DEBUG
        if(j<0 || j>= ((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)   ){
            std::cout<<"vector_sparse_dynamic_enhanced<T>::read: out of range. Trying to access "<<j<<" in a vector having size "<<((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if (this->occupancy[j]<0) {
         return 0.0;
     } else {
         return this->data[this->occupancy[j]];
     }
  }


template<class T> T& vector_sparse_dynamic_enhanced<T>::operator[](Integer j){
     #ifdef DEBUG
        if(j<0 || j>=((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)){
            std::cout<<"vector_sparse_dynamic_enhanced<T>::operator[]: out of range. Trying to access "<<j<<" in a vector having size "<<((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if (this->occupancy[j]<0) {
         std::cerr<<"using vector_sparse_dynamic_enhanced<T>::operator[] without initializing sorting value. This is an error. Vector will not properly sort. Setting sorting value equal to index value. Use () instead!!!!"<<std::endl;
         key[j]=this->nnz;
         this->occupancy[j]=this->nnz;
         this->pointer[this->nnz]=j;
         this->nnz++;
         this->data[this->occupancy[j]]=0.0;
         return this->data[this->occupancy[j]];
     } else {
         return this->data[this->occupancy[j]];
     }
  }

template<class T> Integer& vector_sparse_dynamic_enhanced<T>::current_index() {
    return this->pointer[(*current_position_iter).second];   // result: index in the vector itself of the current position
  }

template<class T> Integer& vector_sparse_dynamic_enhanced<T>::current_position(){
    return (*current_position_iter).second;            // result: position in pointer field
  }

template<class T> T& vector_sparse_dynamic_enhanced<T>::current_element()  {
    return this->data[(*current_position_iter).second];      // result: data stored at current position
 }

template<class T> Integer vector_sparse_dynamic_enhanced<T>::current_sorting_index() const {
    return (*current_position_iter).first;             // the sorting value of the current position
 }

template<class T> void vector_sparse_dynamic_enhanced<T>::move_to_beginning(){
    current_position_iter=key.begin();
  }

template<class T> void vector_sparse_dynamic_enhanced<T>::take_step_forward(){
    current_position_iter++;
  }

template<class T> bool vector_sparse_dynamic_enhanced<T>::at_beginning() const {
    return current_position_iter==key.begin();
  }

template<class T> bool vector_sparse_dynamic_enhanced<T>::at_end() const {
    return current_position_iter==key.end();
}

template<class T> void vector_sparse_dynamic_enhanced<T>::zero_set(Integer j){
     #ifdef DEBUG
        if(j<0 || j>=((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)){
            std::cout<<"vector_sparse_dynamic<T>::zero_set: out of range. Trying to access "<<j<<" in a vector having size "<<((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if(this->occupancy[j]>=0){
         this->data[this->occupancy[j]]=0.0;
         this->occupancy[j]=-1;
         // note: corresponding key entry is not erased!!
     }
  }


template<class T> void vector_sparse_dynamic_enhanced<T>::zero_set(Integer j, Integer k){
     #ifdef DEBUG
        if(j<0 || j>=((dynamic_cast<vector_sparse_dynamic<T> >(this))->size) || k<0 || k>=((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)){
            std::cout<<"vector_sparse_dynamic<T>::zero_set: out of range. Trying to access "<<j<<" with sorting index "<<k<<" in a vector having size "<<((dynamic_cast<vector_sparse_dynamic<T> >(this))->size)<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
     #endif
     if(this->occupancy[j]>=0){
         this->data[this->occupancy[j]]=0.0;
         this->occupancy[j]=-1;
         #ifdef DEBUG
             if(this->pointer[key[k]] != j){
                 std::cerr<<"vector_sparse_dynamic<T>::zero_set: sorting index and index do not correspond."<<std::endl;
                 return;
             }
         #endif
         if (current_sorting_index()==k) current_position_iter++;
         key.erase(k);
     }
  }

template<class T> void vector_sparse_dynamic_enhanced<T>::current_zero_set(){
     if(this->occupancy[current_index()]>=0){
         this->data[this->occupancy[current_index()]]=0.0;
         this->occupancy[current_index()]=-1;
     }
     Integer k=current_sorting_index();
     current_position_iter++;
     key.erase(k);
  }

template<class T>  Real vector_sparse_dynamic_enhanced<T>::memory() const{
    return (Real) (sizeof(T)+ 4*sizeof(Integer))*this->dim() + 2*sizeof(Integer);
}

//***********************************************************************************************************************
//                                                                                                                      *
//         The implementation of the class matrix_sparse                                                                *
//                                                                                                                      *
//***********************************************************************************************************************

//***********************************************************************************************************************
// Class matrix_sparse: private functions                                                                               *
//***********************************************************************************************************************

template<class T> void matrix_sparse<T>::insert_data(const vector_dense<T>& data_vector, const index_list& list, Integer begin_index){
     Integer j,k;
     for(Integer i=0; i<list.dimension(); i++){
         k=begin_index+i;
         j=list[i];
         indices[k]=j;
         data[k]=data_vector[j];
     }
  }

template<class T> void matrix_sparse<T>::insert_data(const vector_dense<T>& data_vector, const index_list& list, Integer begin_index_matrix, Integer begin_index_list, Integer n, Integer offset){
     Integer index_matrix, index_list, index_data;
     if(non_fatal_error( (n+begin_index_list>list.dimension()), "matrix_sparse<T>::insert_data: trying to insert too many elements.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     for(Integer i=0; i<n; i++){
         index_matrix          = begin_index_matrix+i;              // the index in (*this) that is being updated
         index_list            = begin_index_list+i;                // corresponding to this element in the list;
         index_data            = list[index_list];                  // the datum in the data_vector has the index/position "index_data"
         indices[index_matrix] = index_data+offset;                        // this position is stored now in (*this).indices
         data[index_matrix]    = data_vector[index_data]; // the datum corresponding to this position is stored in (*this).data.
     }
  }

template<class T> void matrix_sparse<T>::erase_resize_data_fields(Integer new_nnz){
    if (nnz != new_nnz){
        if (data    != 0){ delete [] data; data = 0;}
        if (indices != 0){ delete [] indices; indices = 0;}
        if(new_nnz>0){
            data = new T[new_nnz];
            indices = new Integer[new_nnz];
            nnz = new_nnz;
        } else {
            data = 0;
            indices = 0;
            nnz = 0;
        }
    }
}

template<class T> void matrix_sparse<T>::erase_resize_pointer_field(Integer new_pointer_size){
    if(new_pointer_size == 0){
        std::cerr<<"matrix_sparse::erase_resize_pointer_field: making pointer field of dimension 0 upon user request. This is likely to cause a segmentation fault."<<std::endl<<std::flush;
    }
    if (pointer_size != new_pointer_size){
        if (pointer != 0){ delete [] pointer; pointer = 0;}
        if (new_pointer_size > 0){
            pointer = new Integer[new_pointer_size];
            pointer_size = new_pointer_size;
        } else {
            pointer = 0;
            pointer_size = 0;
        }
    }
}

template<class T> void matrix_sparse<T>::erase_resize_all_fields(Integer new_pointer_size, Integer new_nnz){
    erase_resize_data_fields(new_nnz);
    erase_resize_pointer_field(new_pointer_size);
  }


template<class T> void matrix_sparse<T>::enlarge_fields_keep_data(Integer newnnz){
    Integer i;
    Integer* newindices = 0;
    T*       newdata    = 0;
    if (newnnz <= nnz) return;
    if(newnnz>0) newindices = new Integer[newnnz];
    for(i=0;i<nnz;i++) newindices[i] = indices[i];
    if (indices != 0) delete [] indices;
    indices = newindices;
    newindices = 0;
    if(newnnz>0) newdata = new T[newnnz];
    for(i=0;i<nnz;i++) newdata[i] = data[i];
    if(data != 0) delete [] data;
    data = newdata;
    newdata = 0;
    nnz = newnnz;        
}



template<class T>  T  matrix_sparse<T>::get_data(Integer k) const{
    #ifdef DEBUG
        if(non_fatal_error((k<0 || k>=nnz)," matrix_sparse::get_data: index out of range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    return data[k];
  }
template<class T>  Integer matrix_sparse<T>::get_index(Integer k) const {
    #ifdef DEBUG
        if(non_fatal_error((k<0 || k>=nnz)," matrix_sparse::get_index: index out of range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    return indices[k];
  }

template<class T>  Integer matrix_sparse<T>::get_pointer(Integer k) const {
    #ifdef DEBUG
        if(non_fatal_error((k<0 || k>=pointer_size)," matrix_sparse::get_pointer: index out of range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    return pointer[k];
  }

template<class T>  Integer matrix_sparse<T>::get_pointer_size() const {
    return pointer_size;
  }


template<class T>  T&  matrix_sparse<T>::set_data(Integer k){
    #ifdef DEBUG
        if(non_fatal_error((k<0 || k>=nnz)," matrix_sparse::set_data: index out of range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    return data[k];
  }
template<class T>  Integer& matrix_sparse<T>::set_index(Integer k){
    #ifdef DEBUG
        if(non_fatal_error((k<0 || k>=nnz)," matrix_sparse::set_index: index out of range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    return indices[k];
  }

template<class T>  Integer& matrix_sparse<T>::set_pointer(Integer k){
    #ifdef DEBUG
        if(non_fatal_error((k<0 || k>=pointer_size)," matrix_sparse::set_pointer: index out of range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    return pointer[k];
  }

template<class T> void matrix_sparse<T>::reformat(Integer new_number_rows, Integer new_number_columns, Integer new_nnz, orientation_type new_orientation){
    Integer new_pointer_size;
    if (new_orientation == ROW) new_pointer_size = new_number_rows+1;
    else new_pointer_size = new_number_columns+1;
    erase_resize_all_fields(new_pointer_size,new_nnz);
    number_rows    = new_number_rows;
    number_columns = new_number_columns;
    orientation    = new_orientation;
    for(Integer i=0; i<pointer_size; i++) pointer[i]=0;
}

template<class T> void matrix_sparse<T>::change_orientation_of_data(const matrix_sparse<T> &X)  {
    vector_dense<Integer> counter;
    reformat(X.number_rows,X.number_columns, X.pointer[X.pointer_size-1],other_orientation(X.orientation));
    counter.resize(pointer_size,0);
    Integer i,j,k,l;
    for (i=0;i<pointer_size;i++) pointer[i] = 0;
    for (i=0;i<X.pointer[X.pointer_size-1];i++) pointer[1+X.indices[i]]++;
    for (i=1;i<pointer_size;i++) pointer[i] += pointer[i-1];
    //for (i=0;i<pointer_size;i++) counter[i]=0;  // already initialized
    for (i=0;i<X.pointer_size-1;i++)
        for(j=X.pointer[i];j< X.pointer[i+1];j++){
            l = X.indices[j];
            k = pointer[l]+counter[l];
            data[k] = X.data[j];
            indices[k] = i;
            counter[l]++;
        }
}

/*
template<class T> void matrix_sparse<T>::change_orientation_of_data(const matrix_sparse<T> &X)  {
         reformat(X.number_rows,X.number_columns, X.pointer[X.pointer_size-1],other_orientation(X.orientation));
         Integer i,j,k,l;
         vector_dense<Integer> counter(pointer_size);
         for (i=0;i<pointer_size;i++) pointer[i] = 0;
         for (i=0;i<X.pointer[X.pointer_size-1];i++) pointer[1+X.indices[i]]++;
         for (i=1;i<pointer_size;i++) pointer[i] += pointer[i-1];
         for (i=0;i<pointer_size;i++) counter[i]=0;
         for (i=0;i<X.pointer_size-1;i++)
            for(j=X.pointer[i];j< X.pointer[i+1];j++){
                l = X.indices[j];
                k = pointer[l]+counter[l];
                data[k] = X.data[j];
                indices[k] = i;
                counter[l]++;
            }
  }

*/

template<class T> void matrix_sparse<T>::insert(const matrix_sparse<T> &A, const vector_dense<T>& row, const vector_dense<T>& column, T center, Integer pos_row, Integer pos_col, Real threshold){
    if(A.orient()==ROW) insert_orient(A,row,column,center,pos_row,pos_col,threshold);
    else insert_orient(A,column,row,center,pos_col,pos_row,threshold);
}


template<class T> void matrix_sparse<T>::insert_orient(const matrix_sparse<T> &A, const vector_dense<T>& along_orient, const vector_dense<T>& against_orient, T center, Integer pos_along_orient, Integer pos_against_orient, Real threshold){
    if(non_fatal_error(A.dim_along_orientation() != against_orient.dimension() || A.dim_against_orientation() != along_orient.dimension()   ,"matrix_sparse<T>::insert_orient: dimensions of vectors to be inserted are not compatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(pos_along_orient < 0 || pos_against_orient < 0 || pos_along_orient > A.dim_along_orientation() ||  pos_against_orient > A.dim_against_orientation(),"matrix_sparse<T>::insert_orient: insert positions are not available.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    array<bool> selected_along_orient(along_orient.dimension(),false);
    array<bool> selected_against_orient(against_orient.dimension(),false);
    Integer number_selected_along_orient = 0;
    Integer number_selected_against_orient = 0;
    bool selected_center = (std::abs(center)>threshold);
    bool not_inserted = true;
    Integer i,j;
    Integer counter = 0;
    for(i=0;i<along_orient.dimension();i++){
        if(std::abs(along_orient[i])>threshold){
            selected_along_orient[i] = true;
            number_selected_along_orient++;
        }
    }
    for(i=0;i<against_orient.dimension();i++){
        if(std::abs(against_orient[i])>threshold){
            selected_against_orient[i] = true;
            number_selected_against_orient++;
        }
    }
    if(selected_center) reformat(A.rows()+1, A.columns()+1, A.non_zeroes()+number_selected_along_orient+number_selected_against_orient+1,A.orient());
    else reformat(A.rows()+1, A.columns()+1, A.non_zeroes()+number_selected_along_orient+number_selected_against_orient,A.orient());
    pointer[0] = 0;
    for(i=0; i< A.dim_along_orientation(); i++){
        not_inserted = true;
        if(i == pos_along_orient){
            for(j=0;j<pos_against_orient;j++){
                if(selected_along_orient[j]){
                    data[counter] = along_orient[j];
                    indices[counter] = j;
                    counter++;
                }
            }
            if(selected_center){
                data[counter] = center;
                indices[counter] = j;
                counter++;
            }
            for(j=pos_against_orient;j<A.dim_against_orientation();j++){
                if(selected_along_orient[j]){
                    data[counter] = along_orient[j];
                    indices[counter] = j+1;
                    counter++;
                }
            }
            pointer[i+1] = counter;
        }
        for(j=A.pointer[i]; j<A.pointer[i+1]; j++){
            if(pos_against_orient<=A.indices[j] && not_inserted){
                if(selected_against_orient[i]){
                    data[counter] = against_orient[i];
                    indices[counter] = pos_against_orient;
                    counter++;
                }
                not_inserted = false;
            }
            data[counter] = A.data[j];
            if(pos_against_orient<=A.indices[j]) indices[counter] = A.indices[j]+1;
            else indices[counter] = A.indices[j];
            counter++;
        }
        if(not_inserted){
            if(selected_against_orient[i]){
                data[counter] = against_orient[i];
                indices[counter] = pos_against_orient;
                counter++;
            }
            not_inserted = false;
        }
        if(i<pos_along_orient){
            pointer[i+1] = counter;
        } else {
            pointer[i+2] = counter;
        }
    }
    if(A.dim_along_orientation() == pos_along_orient){
        for(j=0;j<pos_against_orient;j++){
            if(selected_along_orient[j]){
                data[counter] = along_orient[j];
                indices[counter] = j;
                counter++;
            }
        }
        if(selected_center){
            data[counter] = center;
            indices[counter] = j;
            counter++;
        }
        for(j=pos_against_orient;j<A.dim_against_orientation();j++){
            if(selected_along_orient[j]){
                data[counter] = along_orient[j];
                indices[counter] = j+1;
                counter++;
            }
        }
        pointer[A.dim_along_orientation()+1] = counter;
    }
}

template<class T> void matrix_sparse<T>::change_orientation(const matrix_sparse<T> &X){
    change_orientation_of_data(X);
  }

template<class T> Integer matrix_sparse<T>::largest_absolute_value_along_orientation(Integer k) const {
    Integer index = 0;
    Real absvalue=0.0;
    #ifdef DEBUG
        if(non_fatal_error(k<0 || k+1>=pointer_size, "matrix_sparse<T>::largest_absolute_values_along_orientation: no such row/column exists")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    for(Integer i=pointer[k]; i<pointer[k+1];i++)
        if(std::abs(data[i])>absvalue){
            absvalue=std::abs(data[i]);
            index=i;
        }
    return index;
  }


template<class T> void matrix_sparse<T>::sum_absolute_values_along_orientation(vector_dense<Real> &v) const {
    v.resize(pointer_size-1,0.0);
    for(Integer i=0;i<pointer_size-1;i++)
        for(Integer j=pointer[i];j<pointer[i+1];j++)
            v[i]+=std::abs(data[j]);
}

template<class T> void matrix_sparse<T>::sum_absolute_values_against_orientation(vector_dense<Real> &v) const {
    v.resize(dim_against_orientation(),0.0);
    for(Integer i=0;i<pointer_size-1;i++)
        for(Integer j=pointer[i];j<pointer[i+1];j++)
            v[indices[j]]+=std::abs(data[j]);
}



template<class T> void matrix_sparse<T>::generic_matrix_vector_multiplication_addition(matrix_usage_type use, const vector_dense<T>& x, vector_dense<T>& v) const {
    Integer i,j;
# ifdef VERYVERBOSE
    clock_t time_1,time_2;
    Real time=0.0;
    time_1 = clock();
#endif
    if ( ( (orientation == ROW)&&(use == ID) ) || ( (orientation == COLUMN)&&(use == TRANSPOSE) )  ){
# ifdef VERYVERBOSE
        std::cout<<"          generic_matrix_vector_multiplication_addition: ROW/ID or COLUMN/TRANSPOSE"<<std::endl;
#endif
        for(i=0;i<pointer_size-1;i++)
            for(j=pointer[i];j<pointer[i+1];j++)
                v[i]+=data[j]*x[indices[j]];
    } else {
# ifdef VERYVERBOSE
        std::cout<<"          generic_matrix_vector_multiplication_addition: ROW/TRANSPOSE or COLUMN/ID"<<std::endl;
#endif
        for(i=0;i<pointer_size-1;i++)
            for(j=pointer[i];j<pointer[i+1];j++)
                v[indices[j]]+=data[j]*x[i];
    }
#ifdef VERYVERBOSE
    time_2 = clock();
    time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
    std::cout<<std::endl<<"          generic_matrix_vector_multiplication_addition time "<<time<<std::endl<<std::flush;
#endif
}

template<class T> void matrix_sparse<T>::vector_of_matrix_matrix_multiplication(const matrix_sparse<T>& C, Integer i, vector_dense<T>& result, orientation_type o) const {
     if(o == ROW) row_of_matrix_matrix_multiplication(C,i,result);
     else column_of_matrix_matrix_multiplication(C,i,result);
   }

template<class T> void matrix_sparse<T>::row_of_matrix_matrix_multiplication(const matrix_sparse<T>& C, Integer i, vector_dense<T>& result) const{
    // check dimensions and range of argument i.
    if((number_columns == C.number_rows) && (i<number_rows)){
        // a few variables to store intermediate results:
        Integer j,k,ind1,ind2,ind1max, ind2max;
        T val;
        // begin actual calculations if dimensions are compatible:
        if (orientation == ROW){
            if (C.orientation == ROW){
                // orientation==ROW, C.orientation==ROW
                result.resize(C.number_columns,0.0);
                // calculate the i-th row of B*C store result in "result"
                for (j = pointer[i]; j < pointer[i+1]; j++){
                    ind1 = indices[j];
                    val = data[j];
                    for (k=C.pointer[ind1]; k<C.pointer[ind1+1]; k++){
                        ind2 = C.indices[k];
                        result[ind2] += val*C.data[k];
                    }    // end for k
                }    // end for j
            } else {
                // orientation==ROW, C.orientation==COLUMN
                result.resize(C.number_columns,0.0);
                // calculate the i-th row of B*C store result in row_res
                for (j=0; j<C.number_columns;j++){
                    ind1 = pointer[i];
                    ind2 = C.pointer[j];
                    ind1max = pointer[i+1];
                    ind2max = C.pointer[j+1];
                    while ((ind1<ind1max) && (ind2<ind2max)){
                        if (indices[ind1] < C.indices[ind2]) ind1++;
                        else if (indices[ind1] > C.indices[ind2]) ind2++;
                        else {
                            result[j] += data[ind1]*C.data[ind2];
                            ind1++;
                            ind2++;
                        }
                    } // end while
                } // end for j
            }    // end if
        } else {
            // orientation==COLUMN, C.orientation either
            std::cerr << "matrix_sparse::row_of_matrix_matrix_multiplication:"<<std::endl;
            std::cerr << "      if the result is to have row format, then the first factor must have row format as well."<<std::endl;
            std::cerr << "      the first factor is however a column matrix."<<std::endl;
            std::cerr << "      Try changing the first factor to a row matrix with change_orientation()."<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
    } else {
        std::cerr << "matrix_sparse::row_of_matrix_matrix_multiplication:"<<std::endl;
        std::cerr << "      matrix dimensions are incompatible or row number is too large."<<std::endl;
        std::cerr << "      row number is "<<i<<std::endl;
        std::cerr << "      dimensions of arguments are (should be equal and larger than row number) "<<number_rows<<" "<<C.number_rows<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> void matrix_sparse<T>::column_of_matrix_matrix_multiplication(const matrix_sparse<T>& C, Integer i, vector_dense<T>& result) const {
    // check compatibility of dimensions and argument i:
    if((number_columns == C.number_rows)&&(i<C.number_columns)){
        // a few variables to store intermediate results:
        Integer j,k,ind1,ind2,ind1max, ind2max;
        T val;
        // begin actual calculations if dimensions are compatible:
        // the values of the following variables, which will determine the size of the fields in the result, depends on the orientation of the result, i.e. of A.
        // if matrix_density is larger than 1, no dropping will occur. This is implemented using a negative density:
        if (orientation == ROW){
            if (C.orientation == ROW){
                //  orientation==ROW, C.orientation==ROW
                std::cerr << "matrix_sparse::column_of_matrix_matrix_multiplication:"<<std::endl;
                std::cerr << "      if the result is to have column format, then the second factor must have column format as well."<<std::endl;
                std::cerr << "      the second factor is however a row matrix."<<std::endl;
                std::cerr << "      Try changing the second factor to a column matrix with change_orientation()."<<std::endl;
                throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
            } else {
                //  orientation==ROW, C.orientation==COLUMN
                result.resize(number_rows,0.0);
                // calculate the i-th column of B*C store result in col_res
                for (j=0; j<number_rows;j++){
                    ind1 = C.pointer[i];
                    ind2 = pointer[j];
                    ind1max = C.pointer[i+1];
                    ind2max = pointer[j+1];
                    while ((ind1<ind1max) && (ind2<ind2max)){
                        if (C.indices[ind1] < indices[ind2]) ind1++;
                        else if (C.indices[ind1] > indices[ind2]) ind2++;
                        else {
                            result[j] += C.data[ind1]*data[ind2];
                            ind1++;
                            ind2++;
                        }
                    } // end while
                } // end for j
            }
        } else {
            if (C.orientation == ROW){
                // orientation==COLUMN, C.orientation==ROW
                std::cerr << "matrix_sparse::column_of_matrix_matrix_multiplication:"<<std::endl;
                std::cerr << "      if the result is to have column format, then the second factor must have column format as well."<<std::endl;
                std::cerr << "      the second factor is however a row matrix."<<std::endl;
                std::cerr << "      Try changing the second factor to a column matrix with change_orientation()."<<std::endl;
                throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
            } else {
                // orientation==COLUMN, C.orientation==COLUMN
                result.resize(number_rows,0.0);
                // calculate the i-th column of B*C store result in col_res
                for (j = C.pointer[i]; j < C.pointer[i+1]; j++){
                    ind1 = C.indices[j];
                    val = C.data[j];
                    for (k=pointer[ind1]; k<pointer[ind1+1]; k++){
                        ind2 = indices[k];
                        result[ind2] += val*data[k];
                    }
                }
            }
        }
    } else {
        std::cerr << "matrix_sparse::column_of_matrix_matrix_multiplication:"<<std::endl;
        std::cerr << "      matrix dimension are incompatible or column number is too large."<<std::endl;
        std::cerr << "      column number is "<<i<<std::endl;
        std::cerr << "      dimensions of arguments are (should be equal and larger than column number) "<<number_columns<<" "<<C.number_columns<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> void matrix_sparse<T>::matrix_matrix_multiplication(T beta, const matrix_sparse<T>& B, const matrix_sparse<T>& C, orientation_type result_orientation, Real matrix_density){
    // check compatibility of dimensions:
    if(B.number_columns != C.number_rows){
        std::cerr << "matrix_sparse::generic_matrix_matrix_multiplication:"<<std::endl;
        std::cerr << "      matrix formats are incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    // check if sparse matrices can be multiplied:
    if((result_orientation == ROW) && (B.orientation != ROW)){
        std::cerr << "matrix_sparse::generic_matrix_matrix_multiplication:"<<std::endl;
        std::cerr << "      if the result is to have row format, then the first factor must have row format as well."<<std::endl;
        std::cerr << "      the first factor is however a column matrix."<<std::endl;
        std::cerr << "      Try changing the first factor to a row matrix with change_orientation()."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    if((result_orientation == COLUMN) && (C.orientation != COLUMN)){
        std::cerr << "matrix_sparse::generic_matrix_matrix_multiplication:"<<std::endl;
        std::cerr << "      if the result is to have column format, then the second factor must have column format as well."<<std::endl;
        std::cerr << "      the second factor is however a row matrix."<<std::endl;
        std::cerr << "      Try changing the second factor to a column matrix with change_orientation()."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    // a few variables to store intermediate results:
    Integer i,offset;
    // the values of the following variables, which will determine the size of the fields in the result, depends on the orientation of the result, i.e. of A.
    // if matrix_density is larger than 1, no dropping will occur. This is implemented using a negative density:
    if (matrix_density >= 1.0) matrix_density = -1.0;
    Integer new_nnz_per_direction;
    Integer new_nnz;
    Integer iterative_dimension; // is number of columns, if multiplication is done column-wise, same for rows.
    Integer orientation_dimension; // the other dimension, i.e. the size of intermediate results, i.e the size of the columns if multiplication is done column-wise
    if (result_orientation == ROW) {
        if (matrix_density <= 0.0)
            new_nnz_per_direction = C.number_columns;
        else
            new_nnz_per_direction = (Integer) ceil(C.number_columns * matrix_density);
        new_nnz = new_nnz_per_direction * B.number_rows;
        iterative_dimension = B.number_rows;
        orientation_dimension = C.number_columns;
    } else {
        if (matrix_density <= 0.0)
            new_nnz_per_direction = B.number_rows;
        else
            new_nnz_per_direction = (Integer) ceil(B.number_rows * matrix_density);
        new_nnz = new_nnz_per_direction * C.number_columns;
        iterative_dimension = C.number_columns;
        orientation_dimension = B.number_rows;
    }
    index_list list(new_nnz_per_direction);
    vector_dense<T> res(orientation_dimension);
    // the format of the matrix D=beta*B*C is (B.number_columns x C.number_rows).
    reformat(B.number_rows, C.number_columns, new_nnz, result_orientation);
    // begin calculating the product allowing for the maximum density indicated above.
    // 1.) the entries are calculated by row/column,
    // 2.) dropping is performed by row/column to ensure the overall density requirements are met.
    for (i=0; i<iterative_dimension; i++){
        offset = i*new_nnz_per_direction;
        pointer[i] = offset;
        // calculate the i-th row of B*C store result in row_res
        B.vector_of_matrix_matrix_multiplication(C,i,res,orientation);
        // multiply the i-th row of B*C with beta
        res.scale(beta);
        // do dropping if needed
        if (new_nnz_per_direction != orientation_dimension){
            list.init();
            res.take_largest_elements_by_abs_value(list, new_nnz_per_direction);                                            // initialize list 0,1,2...,number_columns-1
        }
        // copy data from row_res to the i-th row of D (copy new_nnz_per_direction elements)
        insert_data(res,list,offset);
    }   // end for i
    pointer[iterative_dimension]=iterative_dimension*new_nnz_per_direction;
  }

template<class T> void matrix_sparse<T>::matrix_addition(T alpha, const matrix_sparse<T>& A, const matrix_sparse<T>& B){
    if((A.number_rows != B.number_rows)||(A.number_columns != B.number_columns)||(A.orientation != B.orientation)){
        std::cerr << "matrix_sparse::matrix_addition: The dimensions and/or orientations of the arguments are incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    reformat(A.number_rows, A.number_columns, min(A.number_rows*A.number_columns, A.nnz+B.nnz), A.orientation);
    Integer j, ind1, ind2, ind1max, ind2max;
    Integer counter = 0;
    pointer[0] = 0;
    for (j=0; j < A.dim_along_orientation(); j++){
        ind1    = A.pointer[j];
        ind2    = B.pointer[j];
        ind1max = A.pointer[j+1];
        ind2max = B.pointer[j+1];
        while ((ind1 < ind1max)&&(ind2 < ind2max)){
            if(A.indices[ind1] == B.indices[ind2]){
                indices[counter] = A.indices[ind1];
                data[counter] = (alpha*A.data[ind1]) + B.data[ind2];
                ind1++;
                ind2++;
                counter++;
            } else {
                if (A.indices[ind1] < B.indices[ind2]){
                    indices[counter] = A.indices[ind1];
                    data[counter] = alpha*A.data[ind1];
                    ind1++;
                    counter++;
                } else {
                    indices[counter]=B.indices[ind2];
                    data[counter]=B.data[ind2];
                    ind2++;
                    counter++;
                }
            }
        } // end while
        if(ind1 == ind1max)
            while(ind2 < ind2max){
                indices[counter]=B.indices[ind2];
                data[counter]=B.data[ind2];
                ind2++;
                counter++;
            }
        if(ind2 == ind2max)
            while(ind1 < ind1max){
                indices[counter]=A.indices[ind1];
                data[counter]=alpha*A.data[ind1];
                ind1++;
                counter++;
            }
        pointer[j+1]=counter;
    } // end for j
    compress();
  }


template<class T> void matrix_sparse<T>::keepFillin(const matrix_sparse<T> &m, Integer fillin,  matrix_sparse<T>& n) const {
    vector_dense<T> absvalue;
    index_list list;
    Integer i;
    if(fillin>m.pointer[1]){
        n=m;
        return;
    }
    absvalue.resize(m.pointer[1]);
    list.resize(m.pointer[1]);
    for(i=0;i<m.pointer[1];i++) absvalue[i]=std::abs(m.data[i]);
    absvalue.quicksort(list,0,m.pointer[1]-1);
    list.quicksort(m.pointer[1]-fillin,m.pointer[1]-1);
    if(n.nnz<fillin) n.reformat(m.number_rows,1,fillin,COLUMN);
    for(i=0;i<fillin;i++){
        n.data[i]=m.data[list[m.pointer[1]-fillin+i]];
        n.indices[i]=m.indices[list[m.pointer[1]-fillin+i]];
    }
    n.pointer[1]=fillin;
  }


//***********************************************************************************************************************
// Class matrix_sparse: constructors, destructors, etc.                                                                 *
//***********************************************************************************************************************



template<class T> matrix_sparse<T>::matrix_sparse(){
    nnz            = 0;
    pointer_size   = 0;
    data           = 0;
    indices        = 0;
    pointer        = 0;
    reformat(0,0,0,ROW);
}


template<class T> matrix_sparse<T>::matrix_sparse(orientation_type o, Integer m, Integer n){
    nnz            = 0;
    pointer_size   = 0;
    data           = 0;
    indices        = 0;
    pointer        = 0;
    reformat(m,n,0,o);
 }

template<class T> matrix_sparse<T>::matrix_sparse(orientation_type o, Integer m, Integer n, Integer nz){
    nnz            = 0;
    pointer_size   = 0;
    data           = 0;
    indices        = 0;
    pointer        = 0;
    reformat(m,n,nz,ROW);
 }

template<class T> matrix_sparse<T>::matrix_sparse(T* _data, Integer* _indices, Integer* _pointer,
        Integer _rows, Integer _columns, orientation_type _orientation, bool _non_owning)
{
    orientation = _orientation;
    number_rows = _rows;
    number_columns = _columns;
    data = _data;
    indices = _indices;
    pointer = _pointer;
    non_owning = _non_owning;
    if (orientation == ROW) pointer_size = number_rows+1;
    else pointer_size = number_columns+1;
    nnz = pointer[pointer_size-1];
}



template<class T> matrix_sparse<T>::matrix_sparse(const matrix_sparse& X){
    nnz            = 0;
    pointer_size   = 0;
    data           = 0;
    indices        = 0;
    pointer        = 0;
    reformat(X.number_rows,X.number_columns,X.nnz,X.orientation);
    Integer i;
    for (i=0;i<nnz;i++) data[i] = X.data[i];
    for (i=0;i<nnz;i++) indices[i] = X.indices[i];
    for (i=0;i<pointer_size;i++) pointer[i] = X.pointer[i];
  }


template<class T> matrix_sparse<T>::~matrix_sparse() { // std::cout<<"matrixdestruktor"<<std::endl;
    if (!non_owning) {
        if (data    != 0) delete [] data; data=0;
        if (indices != 0) delete [] indices; indices=0;
        if (pointer != 0) delete [] pointer; pointer=0;
    }
}

//***********************************************************************************************************************
// Class matrix_sparse: Basic functions                                                                                 *
//***********************************************************************************************************************

template<class T> void matrix_sparse<T>::matrix_vector_multiplication_add(matrix_usage_type use, const vector_dense<T>& x, vector_dense<T>& v) const {
    if(non_fatal_error(((use==ID)&&((columns() != x.size)||(rows() != v.size)) ), "matrix_sparse::matrix_vector_multiplication_add: incompatible dimensions."  )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(((use==TRANSPOSE)&&((rows() != x.size)||(columns() != v.size)) ), "matrix_sparse::matrix_vector_multiplication_add: incompatible dimensions."  )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    generic_matrix_vector_multiplication_addition(use,x,v);
  }

template<class T> void matrix_sparse<T>::matrix_vector_multiplication(matrix_usage_type use, const vector_dense<T>& x, vector_dense<T>& v) const {
    if(non_fatal_error(((use==ID)&&(columns() != x.dimension()) ), "matrix_sparse::matrix_vector_multiplication: incompatible dimensions."  )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(((use==TRANSPOSE)&&(rows() != x.dimension()) ), "matrix_sparse::matrix_vector_multiplication: incompatible dimensions."  )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(use == ID) v.resize(rows(),0.0);
    else v.resize(columns(),0.0);
    generic_matrix_vector_multiplication_addition(use,x,v);
  }


template<class T> void matrix_sparse<T>::matrix_vector_multiplication(matrix_usage_type use, vector_dense<T>& v) const {
    vector_dense<T> w;
    w.interchange(v);
    if(non_fatal_error(((use==ID)&&(columns() != w.dimension()) ), "matrix_sparse::matrix_vector_multiplication: incompatible dimensions."  )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(((use==TRANSPOSE)&&(rows() != w.dimension()) ), "matrix_sparse::matrix_vector_multiplication: incompatible dimensions."  )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(use == ID) v.resize(rows(),0.0);
    else v.resize(columns(),0.0);
    generic_matrix_vector_multiplication_addition(use,w,v);
  }




//***********************************************************************************************************************
// Class matrix_sparse: operations on the matrix itself                                                                 *
//***********************************************************************************************************************

template<class T> matrix_sparse<T> matrix_sparse<T>::transpose_in_place() {
     orientation = other_orientation(orientation);
     std::swap(number_rows, number_columns);
     return *this;
  }

template<class T> void matrix_sparse<T>::scalar_multiply(T d) {
     for(Integer i=0;i<nnz;i++) data[i]*=d;
  }


template<class T> void matrix_sparse<T>::exponential_scale_orientation_based(const vector_dense<T>& D1, const vector_dense<T>& D2){
    if(non_fatal_error( ((D1.dimension() != this->dim_along_orientation() )||(D2.dimension() != this->dim_against_orientation())), "matrix_sparse::scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer i;
    Integer j;
    for(i=0;i<pointer_size-1;i++)
        for(j=pointer[i];j<pointer[i+1];j++)
            data[j] *= std::exp(D1[i]*D2[read_index(j)]);
  }

template<class T> void matrix_sparse<T>::scale_orientation_based(const vector_dense<T>& D1, const vector_dense<T>& D2){
    if(non_fatal_error( ((D1.dimension() != this->dim_along_orientation())||(D2.dimension() != this->dim_against_orientation())), "matrix_sparse::scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer i;
    Integer j;
    for(i=0;i<pointer_size-1;i++)
        for(j=pointer[i];j<pointer[i+1];j++)
            data[j] *= D1[i]*D2[read_index(j)];
  }


template<class T> void matrix_sparse<T>::exponential_scale(const vector_dense<T>& D1, const vector_dense<T>& D2){
      if(orientation == ROW) this->exponential_scale_orientation_based(D1,D2);
      else  this->exponential_scale_orientation_based(D2,D1);
  }

template<class T> void matrix_sparse<T>::scale(const vector_dense<T>& D1, const vector_dense<T>& D2){
      if(orientation == ROW) this->scale_orientation_based(D1,D2);
      else  this->scale_orientation_based(D2,D1);
  }

template<class T> void matrix_sparse<T>::scale(const vector_dense<T>& v, orientation_type o){
    if(non_fatal_error( (((o==ROW)&&(v.dimension() != number_rows))||((o==COLUMN)&&(v.dimension() != number_columns))), "matrix_sparse::scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer i;
    Integer j;
    if(o==orientation)
        for(i=0;i<pointer_size-1;i++)
            for(j=pointer[i];j<pointer[i+1];j++)
                data[j] *= v[i];
    else
        for(j=0;j<nnz;j++)
          data[j] *= v[indices[j]];
  }


template<class T> void matrix_sparse<T>::scale(const matrix_sparse<T>& A, const vector_dense<T>& v, orientation_type o){
    if(non_fatal_error( (((o==ROW)&&(v.dimension() != A.number_rows))||((o==COLUMN)&&(v.dimension() != A.number_columns))), "matrix_sparse::scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    reformat(A.number_rows,A.number_columns,A.actual_non_zeroes(), A.orientation);
    Integer i;
    Integer j;
    if(o==orientation)
        for(i=0;i<pointer_size-1;i++)
            for(j=pointer[i];j<pointer[i+1];j++)
                data[j] = A.data[j]*v[i];
    else
        for(j=0;j<nnz;j++)
            data[j]  = A.data[j]* v[A.indices[j]];
    for(i=0;i<pointer_size;i++) pointer[i]=A.pointer[i];
    for(i=0;i<A.actual_non_zeroes();i++) indices[i]=A.indices[i];
  }



template<class T> void matrix_sparse<T>::inverse_scale(const vector_dense<T>& v, orientation_type o){
    if(non_fatal_error( (((o==ROW)&&(v.dimension() != number_rows))||((o==COLUMN)&&(v.dimension() != number_columns))), "matrix_sparse::inverse_scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer i;
    Integer j;
    if(o==orientation)
        for(i=0;i<pointer_size-1;i++)
            for(j=pointer[i];j<pointer[i+1];j++)
                data[j] /= v[i];
    else
        for(j=0;j<nnz;j++)
          data[j] /= v[indices[j]];
  }

template<class T> void matrix_sparse<T>::inverse_scale(const matrix_sparse<T>& A, const vector_dense<T>& v, orientation_type o){
    if(non_fatal_error( (((o==ROW)&&(v.dimension() != A.number_rows))||((o==COLUMN)&&(v.dimension() != A.number_columns))), "matrix_sparse::inverse_scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    reformat(A.number_rows,A.number_columns,A.actual_non_zeroes(), A.orientation);
    Integer i;
    Integer j;
    if(o==orientation)
        for(i=0;i<pointer_size-1;i++)
            for(j=pointer[i];j<pointer[i+1];j++)
                data[j] = A.data[j]/v[i];
    else
        for(j=0;j<A.actual_non_zeroes();j++)
            data[j] = A.data[j]/v[A.indices[j]];
    for(i=0;i<pointer_size;i++) pointer[i]=A.pointer[i];
    for(i=0;i<A.actual_non_zeroes();i++) indices[i]=A.indices[i];
  }

template<class T> void matrix_sparse<T>::inverse_scale_orientation_based(const vector_dense<T>& D1, const vector_dense<T>& D2){
    if(non_fatal_error( ((D1.dimension() != this->dim_along_orientation())||(D2.dimension() != this->dim_against_orientation())), "matrix_sparse::inverse_scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer i;
    Integer j;
    for(i=0;i<pointer_size-1;i++)
        for(j=pointer[i];j<pointer[i+1];j++)
            data[j] /= (D1[i]*D2[read_index(j)]);
  }

template<class T> void matrix_sparse<T>::inverse_scale(const vector_dense<T>& D1, const vector_dense<T>& D2){
      if(orientation == ROW) this->inverse_scale_orientation_based(D1,D2);
      else  this->inverse_scale_orientation_based(D2,D1);
  }


template<class T> void matrix_sparse<T>::normalize_columns(vector_dense<T>& D_r){
    D_r.norm2_of_dim1(*this,COLUMN);
    inverse_scale(D_r,COLUMN);
  }

template<class T> void matrix_sparse<T>::normalize_rows(vector_dense<T>& D_l){
    D_l.norm2_of_dim1(*this,ROW);
    inverse_scale(D_l,ROW);
  }


template<class T> void matrix_sparse<T>::normalize(){
    vector_dense<T> D_l, D_r;
    D_r.norm2_of_dim1(*this,COLUMN);
    inverse_scale(D_r,COLUMN);
    D_l.norm2_of_dim1(*this,ROW);
    inverse_scale(D_l,ROW);
  }


template<class T> void matrix_sparse<T>::normalize(vector_dense<T>& D_l, vector_dense<T>& D_r){
    D_r.norm2_of_dim1(*this,COLUMN);
    inverse_scale(D_r,COLUMN);
    D_l.norm2_of_dim1(*this,ROW);
    inverse_scale(D_l,ROW);
  }



template<class T> void matrix_sparse<T>::normalize_columns(const matrix_sparse<T>& A, vector_dense<T>& D_r){
    D_r.norm2_of_dim1(A,COLUMN);
    inverse_scale(A,D_r,COLUMN);
  }

template<class T> void matrix_sparse<T>::normalize_rows(const matrix_sparse<T>& A, vector_dense<T>& D_l){
    D_l.norm2_of_dim1(A,ROW);
    inverse_scale(A,D_l,ROW);
  }



template<class T> void matrix_sparse<T>::normalize(const matrix_sparse<T>& A, vector_dense<T>& D_l, vector_dense<T>& D_r){
    D_r.norm2_of_dim1(A,COLUMN);
    inverse_scale(A,D_r,COLUMN);
    D_l.norm2_of_dim1(*this,ROW);
    inverse_scale(D_l,ROW);
  }

template<class T> bool matrix_sparse<T>::numerical_zero_check(Real threshold) const {
    if(threshold < 0.0) return false;
    Real sum = 0.0;
    threshold *= threshold;
    Integer i,j;
    for(i=0;i<pointer_size-1;i++){
        for(j=pointer[i];j<pointer[i+1];j++)
            sum += absvalue_squared(data[j]);
        if(sum > threshold) return false;
    }
    return true;
}

//***********************************************************************************************************************
// Class matrix_sparse: matrix-valued operations                                                                        *
//***********************************************************************************************************************

template<class T> matrix_sparse<T> matrix_sparse<T>::operator *(T d) const {
    matrix_sparse<T> A(*this);
    A.scalar_multiply(d);
    return A;
  }

template<class T> matrix_sparse<T> matrix_sparse<T>::reorder(const index_list& invperm){
    Integer k,j;
    Integer needed_storage = max_one_dim_size();
    index_list used_indices;
    vector_dense<T> stored_data;
    index_list list;
    used_indices.resize_without_initialization(needed_storage);
    stored_data.resize_without_initialization(needed_storage);
    list.resize_without_initialization(needed_storage);
    for(k=0;k<pointer_size-1;k++){
        list.init();
        for(j=pointer[k];j<pointer[k+1];j++)
            used_indices[j-pointer[k]]=indices[j];
        for(j=pointer[k];j<pointer[k+1];j++)
            stored_data[j-pointer[k]]=data[j];
        quicksort(used_indices,list,invperm,0,pointer[k+1]-pointer[k]-1);
        for(j=pointer[k];j<pointer[k+1];j++){
            indices[j]=used_indices[j-pointer[k]];
        }
        for(j=pointer[k];j<pointer[k+1];j++)
            data[j]=stored_data[list[j-pointer[k]]];
    }  // end for k
    return *this;
  }

template<class T> matrix_sparse<T> matrix_sparse<T>::normal_order(){
    Integer k,j;
    Integer needed_storage = max_one_dim_size();
    index_list used_indices;
    vector_dense<T> stored_data;
    index_list list;
    used_indices.resize_without_initialization(needed_storage);
    stored_data.resize_without_initialization(needed_storage);
    list.resize_without_initialization(needed_storage);
    for(k=0;k<pointer_size-1;k++){
        list.init();
        for(j=pointer[k];j<pointer[k+1];j++)
            used_indices[j-pointer[k]]=indices[j];
        for(j=pointer[k];j<pointer[k+1];j++)
            stored_data[j-pointer[k]]=data[j];
        used_indices.quicksort(list,0,pointer[k+1]-pointer[k]-1);
        for(j=pointer[k];j<pointer[k+1];j++){
            indices[j]=used_indices[j-pointer[k]];
        }
        for(j=pointer[k];j<pointer[k+1];j++)
            data[j]=stored_data[list[j-pointer[k]]];
    }  // end for k
    return *this;
  }



template<class T> void matrix_sparse<T>::transp(const matrix_sparse<T>& X) {
    change_orientation_of_data(X);
    transpose_in_place();
  }


template<class T> void matrix_sparse<T>::transpose(const matrix_sparse<T>& X) {
    change_orientation_of_data(X);
    transpose_in_place();
  }

template<class T> matrix_sparse<T> matrix_sparse<T>::operator = (const matrix_sparse<T>& X){
    if(this==&X) return *this;
    reformat(X.number_rows,X.number_columns,X.nnz,X.orientation);
    Integer i;
    for (i=0;i<nnz;i++) data[i] = X.data[i];
    for (i=0;i<nnz;i++) indices[i] = X.indices[i];
    for (i=0;i<pointer_size;i++) pointer[i] = X.pointer[i];
    return *this;
}

template<class T> void matrix_sparse<T>::setup(Integer m, Integer n, Integer nonzeroes, T* data_array, Integer* indices_array, Integer* pointer_array, orientation_type O){
    #ifdef DEBUG
         std::cerr<<"matrix_sparse::setup: WARNING: making a matrix using pointers. This is not recommended. You are responsible for making sure that this does not result in a segmentation fault!"<<std::endl<<std::flush;
    #endif
    number_rows = m;
    number_columns = n;
    nnz = nonzeroes;
    data = data_array;
    pointer = pointer_array;
    indices = indices_array;
    orientation = O;
    if(orientation == ROW) pointer_size = number_rows + 1;
    else pointer_size = number_columns + 1;
}

template<class T> void matrix_sparse<T>::free(Integer& m, Integer& n, Integer& nonzeroes, Integer& ps, T*& data_array, Integer*& indices_array, Integer*& pointer_array, orientation_type& O){
    #ifdef DEBUG
         std::cerr<<"matrix_sparse::free: WARNING: not freeing memory and destructor will not free memory. This is not recommended. You are responsible for freeing the memory using the pointer that this function is returning."<<std::endl<<std::flush;
    #endif
    m = number_rows;
    number_rows = 0;
    n = number_columns;
    number_columns = 0;
    nonzeroes = nnz;
    nnz = 0;
    data_array = data;
    data = 0;
    pointer_array = pointer;
    pointer = 0;
    indices_array = indices;
    indices = 0;
    O = orientation;
    ps = pointer_size;
    pointer_size = 0;
}

template<class T> void matrix_sparse<T>::null_matrix_keep_data(){
    #ifdef DEBUG
         std::cerr<<"matrix_sparse::null_matrix_keep_data: WARNING: not freeing memory and destructor will not free memory. This is not recommended. You are responsible for freeing memory somehow...."<<std::endl<<std::flush;
    #endif
    number_rows = 0;
    number_columns = 0;
    nnz = 0;
    pointer_size = 0;
    data = 0;
    pointer = 0;
    indices = 0;
}

template<class T> void matrix_sparse<T>::copy_and_destroy(matrix_sparse<T>& A){
    number_columns = A.number_columns;
    A.number_columns = 0;
    number_rows = A.number_rows;
    A.number_rows = 0;
    nnz = A.nnz;
    A.nnz = 0;
    pointer_size = A.pointer_size;
    A.pointer_size = 0;
    orientation = A.orientation;
    if(data != 0) delete [] data;
    data = A.data;
    A.data = 0;
    if(indices != 0) delete [] indices;
    indices = A.indices;
    A.indices = 0;
    if(pointer != 0) delete [] pointer;
    pointer = A.pointer;
    A.pointer = 0;
}

template<class T> void matrix_sparse<T>::interchange(matrix_sparse<T>& A){
    std::swap(number_columns,A.number_columns);
    std::swap(number_rows,A.number_rows);
    std::swap(nnz,A.nnz);
    std::swap(pointer_size,A.pointer_size);
    std::swap(orientation,A.orientation);
    std::swap(data,A.data);
    std::swap(indices,A.indices);
    std::swap(pointer,A.pointer);
}


template<class T> void matrix_sparse<T>::interchange(T*& Adata, Integer*& Aindices, Integer*& Apointer, Integer& Anumber_rows, Integer& Anumber_columns, orientation_type& Aorientation){
#ifdef DEBUG
    std::cerr<<"matrix_sparse::interchange: WARNING: making a matrix using pointers. This is not recommended. You are responsible for making sure that this does not result in a segmentation fault!"<<std::endl<<std::flush;
#endif
    std::swap(number_columns,Anumber_columns);
    std::swap(number_rows,Anumber_rows);
    std::swap(orientation,Aorientation);
    std::swap(data,Adata);
    std::swap(indices,Aindices);
    std::swap(pointer,Apointer);
    if (orientation == ROW) pointer_size = number_rows+1;
    else pointer_size = number_columns+1;
    nnz = pointer[pointer_size-1];
}

template<class T> void matrix_sparse<T>::interchange(T*& Adata, Integer*& Aindices, Integer*& Apointer, Integer& Anumber_rows, Integer& Anumber_columns, Integer& Annz, orientation_type& Aorientation){
#ifdef DEBUG
    std::cerr<<"matrix_sparse::interchange: WARNING: making a matrix using pointers. This is not recommended. You are responsible for making sure that this does not result in a segmentation fault!"<<std::endl<<std::flush;
#endif
    std::swap(number_columns,Anumber_columns);
    std::swap(number_rows,Anumber_rows);
    std::swap(nnz,Annz);
    std::swap(orientation,Aorientation);
    std::swap(data,Adata);
    std::swap(indices,Aindices);
    std::swap(pointer,Apointer);
    if (orientation == ROW) pointer_size = number_rows+1;
    else pointer_size = number_columns+1;
}

//***********************************************************************************************************************
// Class matrix_sparse: vector-valued operations                                                                        *
//***********************************************************************************************************************

template<class T> vector_dense<T> matrix_sparse<T>::operator*(const vector_dense<T>& x) const {
    vector_dense<T> v;
    v.resize_without_initialization(number_rows);
    if (number_columns != x.dimension()){
        std::cerr<<"matrix_sparse::operator *(vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else
        matrix_vector_multiplication(ID,x,v);
    return v;
  }

//***********************************************************************************************************************
// Class matrix_sparse: functions, information                                                                          *
//***********************************************************************************************************************

template<class T> Integer matrix_sparse<T>::rows() const {
     return number_rows;
  }

template<class T> Integer matrix_sparse<T>::columns() const{
     return number_columns;
  }

template<class T> Integer matrix_sparse<T>::dimension() const{
     #ifdef DEBUG
         if(!square_check()) std::cerr<<"matrix_sparse::dimension: matrix is not square. Returning number of rows."<<std::endl;
     #endif
     return number_rows;
  }

template<class T> Real matrix_sparse<T>::row_density() const {
    return ((Real) actual_non_zeroes())/((Real)rows());
  }

template<class T> Real matrix_sparse<T>::column_density() const {
    return  ((Real) actual_non_zeroes())/((Real)columns());
  }

template<class T> bool matrix_sparse<T>::square_check() const {
    return (number_rows == number_columns);
  }

template<class T> Integer matrix_sparse<T>::read_pointer_size() const {
    return pointer_size;
  }

template<class T> orientation_type matrix_sparse<T>::orient() const {
    return orientation;
  }

template<class T> Integer matrix_sparse<T>::non_zeroes() const {
    return nnz;
  }

template<class T> Integer matrix_sparse<T>::actual_non_zeroes() const {
    return pointer[pointer_size-1];
  }


template<class T> Integer matrix_sparse<T>::read_pointer(Integer i) const {
    #ifdef DEBUG
        if(i<0||i>=pointer_size){
            std::cerr<<"matrix_sparse::read_pointer: index out of range. Attempting to access pointer at index "<<i<<" out of "<<pointer_size<<" possible indices of a ("<<number_rows<<","<<number_columns<<")-matrix."<<std::endl;
            if(check_consistency()) std::cerr<<"matrix_sparse::read_pointer: matrix is consistent."<<std::endl;
            else std::cerr<<"matrix_sparse::read_pointer: matrix is NOT consistent."<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
    #endif
    return pointer[i];
  }


template<class T> Integer matrix_sparse<T>::read_index(Integer i) const {
    #ifdef DEBUG
        if(i<0||i>=nnz){
            std::cerr<<"matrix_sparse::read_index: index out of range. Attempting to access indices at index "<<i<<" out of "<<nnz<<" possible indices of a ("<<number_rows<<","<<number_columns<<")-matrix."<<std::endl;
            if(check_consistency()) std::cerr<<"matrix_sparse::read_index: matrix is consistent."<<std::endl;
            else std::cerr<<"matrix_sparse::read_index: matrix is NOT consistent."<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
    #endif
    return indices[i];
  }


template<class T> T matrix_sparse<T>::read_data(Integer i) const {
    #ifdef DEBUG
        if(i<0||i>=nnz){
            std::cerr<<"matrix_sparse::read_data: index out of range. Attempting to access data at index "<<i<<" out of "<<nnz<<" possible indices of a ("<<number_rows<<","<<number_columns<<")-matrix."<<std::endl;
            if(check_consistency()) std::cerr<<"matrix_sparse::read_data: matrix is consistent."<<std::endl;
            else std::cerr<<"matrix_sparse::read_data: matrix is NOT consistent."<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
    #endif
    return data[i];
  }


template<class T> Integer matrix_sparse<T>::dim_along_orientation() const {
     if (orientation == ROW) return number_rows;
     else return number_columns;
  }

template<class T> Integer matrix_sparse<T>::dim_against_orientation() const {
     if (orientation == ROW) return number_columns;
     else return number_rows;
  }

template<class T> Integer matrix_sparse<T>::max_one_dim_size() const {
    Integer m=0;
    for(Integer i=0;i<pointer_size-1;i++) m=max(m,pointer[i+1]-pointer[i]);
    return m;
  }

template<class T> Real matrix_sparse<T>::norm1() const {
    vector_dense<T> v;
    if (orientation == ROW) sum_absolute_values_against_orientation(v);
    else sum_absolute_values_along_orientation(v);
    return v.norm_max();
  }

template<class T> Real matrix_sparse<T>::norm_prod() const {
    vector_dense<T> col_res;
    vector_dense<T> col_res_abs;
    vector_dense<T> abs_values;
    Real norm;
    col_res.resize_without_initialization(number_rows);
    col_res_abs.resize_without_initialization(number_rows);
    abs_values.resize_without_initialization(number_rows);
    Integer i, j, ind1, ind2, ind1max, ind2max;
    // calculate the columns of the product. col_res contains the i-th column of the result.
    for (i=0; i<number_columns; i++){
        col_res.set_all(0.0);
        // calculate the (j,i)-th element of the product
        for (j=0; j<number_rows;j++){
            ind1 = pointer[i];
            ind2 = pointer[j];
            ind1max = pointer[i+1];
            ind2max = pointer[j+1];
            while ((ind1<ind1max) && (ind2<ind2max)){
                if (indices[ind1] < indices[ind2]) ind1++;
                else if (indices[ind1] > indices[ind2]) ind2++;
                else {
                    col_res[j] += data[ind1]*data[ind2];
                    ind1++;
                    ind2++;
                }
            } // end while
        } // end for j
        // now col_res contains the i-th column
        col_res_abs.absvalue(col_res);  // make vector containing abs. value of the i-th column
        abs_values[i]=col_res_abs.sum_over_elements();  // take the sum of the absolute values and store in abs_values
    } // end for i
    norm = abs_values.max_over_elements();  // return the largest of the column sums
    return norm;
}

template<class T> Real norm1_prod (const matrix_sparse<T>& B, const matrix_sparse<T>& C) {
    if ((B.orientation == ROW) && (C.orientation == COLUMN)){
        if (B.number_columns == C.number_rows){
            vector_dense<T> col_res(B.number_rows);
            vector_dense<T> col_res_abs(B.number_rows);
            vector_dense<T> abs_values(C.number_columns);
            Integer i, j, ind1, ind2, ind1max, ind2max;
            T norm;
            // calculate the columns of the product. col_res contains the i-th column of the result.
            for (i=0; i<C.number_columns; i++){
                col_res.set_all(0.0);
                // calculate the (j,i)-th element of the product
                for (j=0; j<B.number_rows;j++){
                    ind1 = C.pointer[i];
                    ind2 = B.pointer[j];
                    ind1max = C.pointer[i+1];
                    ind2max = B.pointer[j+1];
                    while ((ind1<ind1max) && (ind2<ind2max)){
                        if (C.indices[ind1] < B.indices[ind2]) ind1++;
                        else if (C.indices[ind1] > B.indices[ind2]) ind2++;
                        else {
                            col_res[j] += C.data[ind1]*B.data[ind2];
                            ind1++;
                            ind2++;
                        }
                    } // end while
                } // end for j
                // now col_res contains the i-th column
                col_res_abs.absvalue(col_res);  // make vector containing abs. value of the i-th column
                abs_values[i]=col_res_abs.sum_over_elements();  // take the sum of the absolute values and store in abs_values
            } // end for i
            norm = abs_values.max_over_elements();  // return the largest of the column sums
            return norm;
        } else {
            std::cerr << "norm1(const matrix_sparse&, const matrix_sparse&):"<<std::endl;
            std::cerr << "     the matrix dimensions are incompatible to carry out matrix multiplication "<<std::endl;
            throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
        }
    } else {
        std::cerr << "norm1(const matrix_sparse&, const matrix_sparse&):"<<std::endl;
        std::cerr << "     the first argument must be in ROW, the second in COLUMN format! "<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> Real matrix_sparse<T>::normF() const {
     Real normsq = 0.0;
     for(Integer i=0;i<pointer[pointer_size-1];i++) normsq += absvalue_squared(data[i]);
     return sqrt(normsq);
  }

template<class T> Integer matrix_sparse<T>::bandwidth() const {
     Integer i,j,bw = 0;
     for(i=0;i<pointer_size-1;i++)
         for(j=pointer[i];j<pointer[i+1];j++)
             bw = max(bw, abs(indices[j]-i));
     return bw+1;
  }


template<class T> Real matrix_sparse<T>::norm_max() const {
    vector_dense<T> v;
    if (orientation == ROW) sum_absolute_values_along_orientation(v);
    else sum_absolute_values_against_orientation(v);
    return v.norm_max();
  }

template<class T> T matrix_sparse<T>::scalar_product_along_orientation(Integer m, Integer n) const{
    if((m>=dim_against_orientation())||(n>=dim_against_orientation())){
        std::cerr<<"matrix_sparse::scalar_product_along_orientation: the arguments are too large. Such rows/columns do not exist."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    T scalar_product=0.0;
    Integer ind_m = pointer[m];
    Integer ind_n = pointer[n];
    Integer ind_mmax = pointer[m+1];
    Integer ind_nmax = pointer[n+1];
    while ((ind_m < ind_mmax)&&(ind_n < ind_nmax)){
        if(indices[ind_m] == indices[ind_n]){
            scalar_product += data[ind_m] * data[ind_n];
            ind_m++;
            ind_n++;
        } else {
            if (indices[ind_m] < indices[ind_n]){
                ind_m++;
            } else {
                ind_n++;
            }
        }
    } // end while
    return scalar_product;
  }



//***********************************************************************************************************************
// Class matrix_sparse: matrix conversion                                                                               *
//***********************************************************************************************************************

template<class T> matrix_dense<T> matrix_sparse<T>::expand() const {
#ifdef DEBUG
    if(non_fatal_error(!check_consistency(),"matrix_sparse::expand(): matrix is inconsistent.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    Integer i;
    Integer j;
    matrix_dense<T> A;
    A.resize(number_rows,number_columns);
    A.set_all(0.0);
    if (orientation == ROW){
        for(i=0;i<pointer_size-1;i++){ // std::cout<<"i schleife i="<<i<<"pointer[i]= "<<pointer[i]<< "pointer[i+1]= "<<pointer[i+1]<<std::endl;
            for(j=pointer[i];j<pointer[i+1];j++){
                //std::cout<<"expand:j scheife "<<i<<" "<<indices[j]<<"j "<<j;
                A.data[i][indices[j]]+=data[j];
                //std::cout<<" erledigt"<<std::endl;
            }
        }
    } else {
        for(i=0;i<pointer_size-1;i++){
            for(j=pointer[i];j<pointer[i+1];j++){
                //std::cout<<"expand:j scheife "<<i<<" "<<indices[j]<<"j "<<j<<std::flush;
                A.data[indices[j]][i]+=data[j];
                //std::cout<<" erledigt"<<std::endl;
            }
        }
    }
    return A;
  }

template<class T> void matrix_sparse<T>::compress(const matrix_dense<T>& A, orientation_type o, double threshold){
    Integer counter=0;
    Integer i,j;
    number_rows=A.rows();
    number_columns=A.columns();
    for(i=0;i<A.rows();i++)
        for(j=0;j<A.columns();j++)
            if (std::abs(A.read(i,j)) > threshold) counter++;
    reformat(number_rows, number_columns,counter,o);
    counter = 0;
    if(o == ROW){
        for(i=0;i<A.rows();i++){
            pointer[i]=counter;
            for(j=0;j<A.columns();j++)
                if(std::abs(A.read(i,j)) > threshold) {
                    indices[counter] = j;
                    data[counter] = A.read(i,j);
                    counter++;
                }
        }
        pointer[A.rows()]=counter;
    } else {
        for(j=0;j<A.columns();j++){
            pointer[j]=counter;
            for(i=0;i<A.rows();i++)
                if (std::abs(A.read(i,j)) > threshold) {
                    indices[counter] = i;
                    data[counter] = A.read(i,j);
                    counter++;
                }
        }
        pointer[A.columns()]=counter;
    }
  }


template<class T> void matrix_sparse<T>::compress(double threshold){
    // need a few variables:
    Integer i,j;
    Integer k;
    Integer counter=0;
    // make new fields to temporarily store the new data.
    array<Integer> new_pointer;
    array<Integer> new_indices;
    array<T>       new_data;
    new_pointer.erase_resize_data_field(pointer_size);
    new_indices.erase_resize_data_field(nnz);
    new_data.erase_resize_data_field(nnz);
    // check by one row/column if the absolute value of the data is larger than the treshold and then copy.
    new_pointer[0]=0;
    for (i=0; i<pointer_size-1; i++){
        for (j=pointer[i]; j<pointer[i+1]; j++){
            if (std::abs(data[j]) >threshold){
                new_data[counter]=data[j];
                new_indices[counter]=indices[j];
                counter++;
            }
        }
        new_pointer[i+1]=counter;
    }
    reformat(number_rows,number_columns,counter,orientation);
    // copy data into fields having the appropriate size
    for (k=0; k<nnz;k++) data[k]=new_data[k];
    for (k=0; k<nnz;k++) indices[k]=new_indices[k];
    for (k=0; k<pointer_size;k++) pointer[k]=new_pointer[k];
}


template<class T> void matrix_sparse<T>::positional_compress(const iluplusplus_precond_parameter& IP, double threshold){
    // need a few variables:
    Integer i,j;
    Integer k;
    Integer counter=0;
    Integer size = max(rows(),columns());
    // make new fields to temporarily store the new data.
    array<Integer> new_pointer;
    array<Integer> new_indices;
    array<T>       new_data;
    new_pointer.erase_resize_data_field(pointer_size);
    new_indices.erase_resize_data_field(nnz);
    new_data.erase_resize_data_field(nnz);
    // check by one row/column if the absolute value of the data is sufficiently large, then copy
    new_pointer[0]=0;
    for (i=0; i<pointer_size-1; i++){
        for (j=pointer[i]; j<pointer[i+1]; j++){
            if (std::abs(data[j])*IP.get_TABLE_POSITIONAL_WEIGHTS((IP.get_SIZE_TABLE_POS_WEIGHTS()*abs(indices[j]-i))/size) >threshold){
                new_data[counter]=data[j];
                new_indices[counter]=indices[j];
                counter++;
            }
        }
        new_pointer[i+1]=counter;
    }
    reformat(number_rows,number_columns,counter,orientation);
    // copy data into fields having the appropriate size
    for (k=0; k<nnz;k++) data[k]=new_data[k];
    for (k=0; k<nnz;k++) indices[k]=new_indices[k];
    for (k=0; k<pointer_size;k++) pointer[k]=new_pointer[k];
}



//***********************************************************************************************************************
// Class matrix_sparse: Input/Output                                                                                    *
//***********************************************************************************************************************

template<class T> std::ostream& operator << (std::ostream& os, const matrix_sparse<T> & x){
     Integer i_data;
     os<<"The matrix has "<<x.rows()<<" rows and "<<x.columns()<<" columns."<<std::endl;
     if(x.orient() == ROW){
         for(Integer i_row=0;i_row<x.rows();i_row++){
             os<<"*** row: "<<i_row<<" ***"<<std::endl;
         if(x.get_pointer(i_row)<x.non_zeroes())
              for(i_data=x.get_pointer(i_row);i_data<x.get_pointer(i_row+1);i_data++)
                  os<<"("<<i_row<<"x"<<x.get_index(i_data)<<") "<<x.get_data(i_data)<<std::endl;}
     }
     else
         for(Integer i_column=0;i_column<x.columns();i_column++){
             os<<"*** column: "<<i_column<<" ***"<<std::endl;
             if(x.get_pointer(i_column)<x.non_zeroes())
             for(i_data=x.get_pointer(i_column);i_data<x.get_pointer(i_column+1);i_data++)
                 os<<"("<<x.get_index(i_data)<<"x"<<i_column<<") "<<x.get_data(i_data)<<std::endl;
         }
     return os;
  }

//***********************************************************************************************************************
// Class matrix_sparse: generating special matrices                                                                     *
//***********************************************************************************************************************

template<class T> void matrix_sparse<T>::diag(T d){
    Integer smallerdim = min(number_rows,number_columns);
    erase_resize_data_fields(smallerdim);
    Integer i;
    for(i=0;i<smallerdim;i++) data[i]    = d;
    for(i=0;i<smallerdim;i++) indices[i] = i;
    for(i=0;i<smallerdim;i++) pointer[i] = i;
    for(i=smallerdim;i<pointer_size;i++) pointer[i]= smallerdim;
}

template<class T> void matrix_sparse<T>::diag(Integer m, Integer n, T d, orientation_type o){
    Integer smallerdim = min(m,n);
    reformat(m,n,smallerdim,o);
    Integer i;
    for(i=0;i<smallerdim;i++) data[i]    = d;
    for(i=0;i<smallerdim;i++) indices[i] = i;
    for(i=0;i<smallerdim;i++) pointer[i] = i;
    for(i=smallerdim;i<pointer_size;i++) pointer[i]= smallerdim;
}


template<class T> void matrix_sparse<T>::square_diag(Integer n, T d, orientation_type o){
    reformat(n,n,n,o);
    Integer i;
    for(i=0;i<n;i++) data[i]    = d;
    for(i=0;i<n;i++) indices[i] = i;
    for(i=0;i<=n;i++) pointer[i] = i;
}


template<class T> void matrix_sparse<T>::read_binary(std::string filename){
    std::ifstream the_file(filename.c_str(), std::ios::binary);
    Integer new_pointer_size;
    Integer new_nnz;
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) &number_rows,    sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) &number_columns, sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) &new_pointer_size,   sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) &new_nnz,            sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) &orientation,    sizeof(orientation_type));
    erase_resize_all_fields(new_pointer_size,new_nnz);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) pointer,         sizeof(Integer)*pointer_size);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) indices,         sizeof(Integer)*nnz);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.read((char*) data,            sizeof(T)*nnz);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::read_binary: error reading file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.close();
}

template<class T> void matrix_sparse<T>::write_mtx(std::string filename) const {
    Integer i,j;
    std::ofstream the_file(filename.c_str());
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_mtx: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    //the_file<<rows()<<" "<<columns()<<" "<<non_zeroes()<<std::endl;
    if(orient() == ROW){
        for(i=0; i<pointer_size-1; i++) 
            for(j=pointer[i]; j<pointer[i+1]; j++)
                the_file<<i+1<<" "<<indices[j]+1<<" "<<data[j]<<std::endl;
    } else {
        for(i=0; i<pointer_size-1; i++) 
            for(j=pointer[i]; j<pointer[i+1]; j++)
                the_file<<indices[j]+1<<" "<<i+1<<" "<<data[j]<<std::endl;
    }
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_mtx: error writing file.")) throw iluplusplus_error(FILE_ERROR);
}

template<class T> void matrix_sparse<T>::write_binary(std::string filename) const {
    std::ofstream the_file(filename.c_str(), std::ios::binary);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) &number_rows,    sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) &number_columns, sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) &pointer_size,   sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) &nnz,            sizeof(Integer));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) &orientation,    sizeof(orientation_type));
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) pointer,         sizeof(Integer)*pointer_size);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) indices,         sizeof(Integer)*nnz);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.write((char*) data,            sizeof(T)*nnz);
    if(non_fatal_error(!the_file.good(),"matrix_sparse::write_binary: error writing file.")) throw iluplusplus_error(FILE_ERROR);
    the_file.close();
}


template<class T> void matrix_sparse<T>::random(Integer m, Integer n, orientation_type O, Integer min_nnz, Integer max_nnz){
    Integer j,k,r,index,number_elements;
    bool index_is_not_new;
    Integer iter_dim = ((O==ROW) ? m : n);
    Integer length = ((O==ROW) ? n : m);
    if(max_nnz<min_nnz) max_nnz = min_nnz;
    //srand(time(0)); // if you initialize the seed in every call, calls in succession (less than 1 second apart) use the same seed an produce the same matrix.
    reformat(m,n,length*max_nnz,O);
    for(k=0; k<iter_dim; k++){
        number_elements = rand()%(max_nnz - min_nnz + 1) + min_nnz;
        for(j=0; j<number_elements; j++){
            index_is_not_new = true;
            while(index_is_not_new){
                index = rand()%length;
                index_is_not_new = false;
                for(r=pointer[k];r<pointer[k]+j;r++) index_is_not_new = index_is_not_new || (index == indices[r]);
            } // end while
            indices[pointer[k]+j] = index;
            data[pointer[k]+j] =  2.0* ((T) rand()) / ((T)RAND_MAX) - 1.0;
        } // end for j
        pointer[k+1] = pointer[k] + number_elements;
    }  // end for k
    //compress(COMPARE_EPS); 
    normal_order();
}


template<class T> void matrix_sparse<T>::random_perturbed_projection_matrix(Integer n, Integer EV1, Integer min_nnz, Integer max_nnz, orientation_type O, Real eps){
    matrix_sparse<T> A,B;
    Integer i;
    if(EV1<0) EV1 = 0;
    if(EV1>n) EV1 = n;
    A.random(n,n,O,min_nnz,max_nnz);
    B.reformat(n,n,EV1,O);
    for(i=0;i<EV1;i++) B.pointer[i] = i;
    for(i=EV1;i<=n;i++) B.pointer[i] = EV1;
    for(i=0;i<EV1;i++) B.indices[i] = i;
    for(i=0;i<EV1;i++) B.data[i] = 1.0;
    matrix_addition(eps/A.normF(),A,B);
}


template<class T> void matrix_sparse<T>::random_multiplicatively_perturbed_projection_matrix(Integer n, Integer rank, Integer min_nnz, Integer max_nnz, orientation_type O, Real eps_EV, Real eps_similarity) {
    matrix_dense<T> A;
    A.random_multiplicatively_perturbed_projection_matrix(n,rank,min_nnz,max_nnz,O,eps_EV,eps_similarity);
    compress(A,O,-1.0);
}


template<class T> void matrix_sparse<T>::tridiag(T a, T b, T c){
    if (number_rows != number_columns) {
        std::cerr << "Error in matrix_sparse::tridiag: This function requires square matrices."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    Integer dimension = number_rows;
    Integer i;
    reformat(dimension,dimension,3*dimension-2,orientation);
    pointer[0]=0;
    for (i=1;i<dimension;i++) pointer[i] = 3*i-1;
    pointer[dimension]=nnz;
    for (i=0;i<dimension;i++)indices[3*i]=i;
    for (i=1;i<dimension;i++)indices[3*i-2]=i;
    for (i=0;i<dimension-1;i++)indices[3*i+2]=i;
    for (i=0;i<dimension;i++)data[3*i]=b;
    if (orientation == ROW){
        for (i=1;i<dimension;i++)data[3*i-2]=c;
        for (i=0;i<dimension-1;i++)data[3*i+2]=a;
    } else {
        for (i=1;i<dimension;i++)data[3*i-2]=a;
        for (i=0;i<dimension-1;i++)data[3*i+2]=c;
    }
}

template<class T> void matrix_sparse<T>::extract(const matrix_sparse<T> &A, Integer m, Integer n){
    Integer i;
    if(non_fatal_error( (m+n>A.pointer_size || n<0 || m<0),"matrix_sparse<T>::extract: arguments out of range.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(A.orientation==ROW){
        reformat(n,A.number_columns,A.pointer[m+n]-A.pointer[m],ROW);
    } else {
        reformat(A.number_rows,n,A.pointer[m+n]-A.pointer[m],COLUMN);
    }
    for(i=0;i<=n;i++) pointer[i]=A.pointer[m+i]-A.pointer[m];
    for(i=0;i<pointer[n];i++){
        indices[i]=A.indices[A.pointer[m]+i];
        data[i]=A.data[A.pointer[m]+i];
    }
    nnz=pointer[n];
  }


//***********************************************************************************************************************
// Class matrix_sparse: Testing                                                                                         *
//***********************************************************************************************************************


template<class T> bool matrix_sparse<T>::check_consistency() const {
      Integer i,j;
      if (pointer[pointer_size-1]>nnz){
          std::cerr<<"matrix_sparse<T>::check_consistency(): matrix is inconstistent: pointer[pointer_size-1]="<<pointer[pointer_size-1]<<">"<<nnz<<"=nnz"<<std::endl;
          return false;
      }
      for(i=0;i<pointer_size-1;i++){ 
          if (pointer[i]>pointer[i+1]){
              std::cerr<<"matrix_sparse<T>::check_consistency(): matrix is inconstistent: pointer["<<i<<"]="<<pointer[i]<<" and pointer["<<i+1<<"]="<<pointer[i+1]<<std::endl;
              return false;
          } 
      }
      for(i=pointer[0];i<pointer[pointer_size-1];i++){
         if (indices[i]>=dim_against_orientation()){
             std::cerr<<"matrix_sparse<T>::check_consistency(): matrix is inconstistent: indices["<<i<<"]="<<indices[i]<<" but permissible dimension is only "<<dim_against_orientation()<<std::endl;
             return false;
         }
         if (indices[i]<0){
             std::cerr<<"matrix_sparse<T>::check_consistency(): matrix is inconstistent: indices["<<i<<"]="<<indices[i]<<" but index must be positive "<<std::endl;
             return false;
         }
      }
          for(i=0;i<pointer_size-1;i++){
              for(j=pointer[i];j<pointer[i+1]-1;j++){
                  if(indices[j]>=indices[j+1]){
                      std::cout<<j<<std::endl; 
                      std::cerr<<"matrix_sparse<T>::check_consistency(): matrix is inconstistent: in row/column i = "<<i<<" natural ordering of indices is violated: indices["<<j<<"]="<<indices[j]<<" is larger than indices["<<j+1<<"]="<<indices[j+1]<<std::endl;   
                      return false;
                  }
              }
          }
      return true;
  }

template<class T> void matrix_sparse<T>::print_pointer() const {
     std::cout << "Pointer: Size of pointer: "<<pointer_size<<std::endl;
     std::cout << "Contents of the pointer field:"<<std::endl;
     for(Integer i=0;i<pointer_size;i++) std::cout<<pointer[i]<<std::endl;
  }

template<class T> void matrix_sparse<T>::print_indices() const {
     std::cout << "Indices: Number of indices in field: "<<nnz<<std::endl;
     std::cout << "Contents of the index field:"<<std::endl;
     for(Integer i=0;i<nnz;i++) std::cout<<indices[i]<<std::endl;
  }

template<class T> void matrix_sparse<T>::print_orientation() const {
  if (orientation == ROW) std::cout<<"ROW";
  else std::cout<<"COLUMN";
  }


template<class T> void matrix_sparse<T>::print_data() const {
     std::cout << "Data: reserved space: "<<nnz<<std::endl;
     std::cout << "Number of data elements stored: "<<pointer[pointer_size-1]<<std::endl;
     std::cout << "Contents of the data field:"<<std::endl;
     for(Integer i=0;i<nnz;i++) std::cout<<data[i]<<std::endl;
  }

template<class T> void matrix_sparse<T>::print_info() const {
     std::cout << "A ("<<number_rows<<"x"<<number_columns<<") sparse matrix having ";
     print_orientation();
     std::cout<<" orientation and "<<nnz<< " non-zero elements."<<std::endl;
  }


template<class T> void matrix_sparse<T>::print_detailed_info() const {
    print_info();
    std::cout<<"pointer_size "<<pointer_size<<std::endl;
    std::cout<<"pointer[pointer_size-1] (actual non-zeroes used) "<<pointer[pointer_size-1]<<std::endl;
    if(check_consistency()) std::cout<<"Matrix passes consistency test."<<std::endl;
    else  std::cout<<"Matrix FAILS consistency test."<<std::endl;
  }

template<class T> void matrix_sparse<T>::print_all() const{
     std::cout << "A ("<<number_rows<<"x"<<number_columns<<") sparse matrix having ";
     print_orientation();
     std::cout<< " orientation."<<std::endl;
     std::cout<<std::endl;
     print_pointer();
     print_indices();
     print_data();
  }


template<class T> special_matrix_type matrix_sparse<T>::shape() const{
    Integer i,j;
    // notation is for a ROW matrix
    bool lower_triangular=true;
    bool upper_triangular=true;
    for(i=0;i<pointer_size-1;i++)
       for(j=pointer[i];j<pointer[i+1];j++)
           lower_triangular = lower_triangular && (indices[j]<=i);
    for(i=0;i<pointer_size-1;i++)
       for(j=pointer[i];j<pointer[i+1];j++)
           upper_triangular = upper_triangular && (indices[j]>=i);
    if (upper_triangular && lower_triangular) return DIAGONAL;
    if ((orientation == ROW) && upper_triangular) return UPPER_TRIANGULAR;
    else return LOWER_TRIANGULAR;
    if ((orientation == ROW) && lower_triangular) return LOWER_TRIANGULAR;
    else return UPPER_TRIANGULAR;
    return UNSTRUCTURED;
  }


/*
template<class T> Real matrix_sparse<T>::degree_of_symmetry() const {
     try {
         matrix_sparse<T> symmetric_part;
         matrix_sparse<T> transposed_matrix;
         if(square_check()){
             transposed_matrix.transp(*this);
             symmetric_part.matrix_addition_complete(1.0, (*this), transposed_matrix);
             return (symmetric_part.normF()/(2.0*normF()));
         } else {
             std::cerr<<"matrix_sparse::degree_of_symmetry(): matrix needs to be square. Returning -1.0."<<std::endl;
             return -1.0;
         }
    }
    catch(iluplusplus_error ippe){
       std::cerr << "matrix_sparse::degree_of_symmetry: "<<ippe.error_message() << std::endl;
       throw;
    }
  }
*/


//***********************************************************************************************************************
// Class matrix_sparse: Interaction with systems of linear equations                                                    *
//***********************************************************************************************************************


template<class T> void matrix_sparse<T>::triangular_solve(special_matrix_type form, matrix_usage_type use, vector_dense<T>& x) const{
    if(non_fatal_error(!(square_check()),"matrix_sparse::triangular_solve: matrix needs to be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(x.dimension() != number_rows, "matrix_sparse::triangular_solve: size of rhs is incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer j,k;
# ifdef VERYVERBOSE
    clock_t time_1,time_2;
    Real time=0.0;
    time_1 = clock();
#endif
    if ( ((form==LOWER_TRIANGULAR) && (orientation==ROW) && (use==ID)) ||
            ((form==UPPER_TRIANGULAR )&& (orientation==COLUMN) && (use==TRANSPOSE)) )
    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve: using: LOWER_TRIANGULAR,ROW,ID"<<std::endl;
        else         std::cout<<"      triangular_solve: using: UPPER_TRIANGULAR,COLUMN,TRANSPOSE"<<std::endl;
#endif
        for(k=0;k<number_rows;k++){
            for(j=pointer[k];j<pointer[k+1]-1;j++) x[k]-= data[j]*x[indices[j]];
            //non_fatal_error(data[pointer[k+1]-1]==0,"matrix_sparse::triangular_solve: pivot must be non-zero.");
            x[k] /= data[pointer[k+1]-1];
        }   // end for k
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve time: "<<time<<std::endl<<std::flush;
#endif
        return;
    } // end if
    if ( ((form==LOWER_TRIANGULAR) && (orientation==COLUMN) && (use==ID)) ||
            ((form==UPPER_TRIANGULAR) && (orientation==ROW) && (use==TRANSPOSE)) )

    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve: using: LOWER_TRIANGULAR,COLUMN,ID"<<std::endl;
        else         std::cout<<"      triangular_solve: using: UPPER_TRIANGULAR,ROW,TRANSPOSE"<<std::endl;
#endif
        Integer k,j;
        for(k=0;k<number_columns;k++){
            x[k] /= data[pointer[k]];
            for(j=pointer[k]+1;j<pointer[k+1];j++) x[indices[j]] -= data[j]*x[k];
        }
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve time: "<<time<<std::endl<<std::flush;
#endif
        return;
    }  // end if
    if ( ((form==UPPER_TRIANGULAR) && (orientation==ROW) && (use==ID)) ||
            ((form==LOWER_TRIANGULAR) && (orientation==COLUMN) && (use==TRANSPOSE)) )
    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve: using: UPPER_TRIANGULAR,ROW,ID"<<std::endl;
        else         std::cout<<"      triangular_solve: using: LOWER_TRIANGULAR,COLUMN,TRANSPOSE"<<std::endl;
#endif
        for(k=number_rows-1;k>=0;k--){
            for(j=pointer[k]+1;j<pointer[k+1];j++) x[k]-= data[j]*x[indices[j]];
            x[k] /= data[pointer[k]];
        }   // end for k
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve time: "<<time<<std::endl<<std::flush;
#endif
        return;
    }  // end if
    if ( ((form==UPPER_TRIANGULAR )&& (orientation==COLUMN) && (use==ID)) ||
            ((form==LOWER_TRIANGULAR) && (orientation==ROW) && (use==TRANSPOSE)))
    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve: using: UPPER_TRIANGULAR,COLUMN,ID"<<std::endl;
        else         std::cout<<"      triangular_solve: using: LOWER_TRIANGULAR,ROW,TRANSPOSE"<<std::endl;
#endif
        Integer k,j;
        for(k=number_columns-1;k>=0;k--){
            x[k] /= data[pointer[k+1]-1];
            for(j=pointer[k];j<pointer[k+1]-1;j++) x[indices[j]] -= data[j]*x[k];
        }
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve time: "<<time<<std::endl<<std::flush;
#endif
        return;
    }  // end if
    std::cerr<<"matrix_sparse::triangular_solve: unknown matrix usage"<<std::endl;
    throw(OTHER_ERROR);
}


template<class T> void matrix_sparse<T>::triangular_solve(special_matrix_type form, matrix_usage_type use, const vector_dense<T>& b, vector_dense<T>& x) const {
    x=b;
    triangular_solve(form,use,x);
  }


template<class T> void matrix_sparse<T>::triangular_solve_with_smaller_matrix(special_matrix_type form, matrix_usage_type use, vector_dense<T>& x) const{
    if(non_fatal_error(!(square_check()),"matrix_sparse::triangular_solve: matrix needs to be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer j,k;
    Integer offset = x.dimension()-number_rows;
# ifdef VERYVERBOSE
    clock_t time_1,time_2;
    Real time=0.0;
    time_1 = clock();
#endif
    if ( ((form==LOWER_TRIANGULAR) && (orientation==ROW) && (use==ID)) ||
            ((form==UPPER_TRIANGULAR )&& (orientation==COLUMN) && (use==TRANSPOSE)) )
    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve_with_smaller_matrix: using: LOWER_TRIANGULAR,ROW,ID"<<std::endl;
        else         std::cout<<"      triangular_solve_with_smaller_matrix: using: UPPER_TRIANGULAR,COLUMN,TRANSPOSE"<<std::endl;
#endif
        for(k=0;k<number_rows;k++){
            for(j=pointer[k];j<pointer[k+1]-1;j++) x[k+offset]-= data[j]*x[indices[j]+offset];
            //non_fatal_error(data[pointer[k+1]-1]==0,"matrix_sparse::triangular_solve: pivot must be non-zero.");
            x[k+offset] /= data[pointer[k+1]-1];
        }   // end for k
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve_with_smaller_matrix time: "<<time<<std::endl<<std::flush;
#endif
        return;
    } // end if
    if ( ((form==LOWER_TRIANGULAR) && (orientation==COLUMN) && (use==ID)) ||
            ((form==UPPER_TRIANGULAR) && (orientation==ROW) && (use==TRANSPOSE)) )

    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve_with_smaller_matrix: using: LOWER_TRIANGULAR,COLUMN,ID"<<std::endl;
        else         std::cout<<"      triangular_solve_with_smaller_matrix: using: UPPER_TRIANGULAR,ROW,TRANSPOSE"<<std::endl;
#endif
        for(k=0;k<number_columns;k++){
            x[k+offset] /= data[pointer[k]];
            for(j=pointer[k]+1;j<pointer[k+1];j++) x[indices[j]+offset] -= data[j]*x[k+offset];
        }
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve_with_smaller_matrix time: "<<time<<std::endl<<std::flush;
#endif
        return;
    }  // end if
    if ( ((form==UPPER_TRIANGULAR) && (orientation==ROW) && (use==ID)) ||
            ((form==LOWER_TRIANGULAR) && (orientation==COLUMN) && (use==TRANSPOSE)) )
    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve_with_smaller_matrix: using: UPPER_TRIANGULAR,ROW,ID"<<std::endl;
        else         std::cout<<"      triangular_solve_with_smaller_matrix: using: LOWER_TRIANGULAR,COLUMN,TRANSPOSE"<<std::endl;
#endif
        for(k=number_rows-1;k>=0;k--){
            for(j=pointer[k]+1;j<pointer[k+1];j++) x[k+offset]-= data[j]*x[indices[j]+offset];
            x[k+offset] /= data[pointer[k]];
        }   // end for k
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve_with_smaller_matrix time: "<<time<<std::endl<<std::flush;
#endif
        return;
    }  // end if
    if ( ((form==UPPER_TRIANGULAR )&& (orientation==COLUMN) && (use==ID)) ||
            ((form==LOWER_TRIANGULAR) && (orientation==ROW) && (use==TRANSPOSE)))
    {
# ifdef VERYVERBOSE
        if (use==ID )std::cout<<"      triangular_solve_with_smaller_matrix: using: UPPER_TRIANGULAR,COLUMN,ID"<<std::endl;
        else         std::cout<<"      triangular_solve_with_smaller_matrix: using: LOWER_TRIANGULAR,ROW,TRANSPOSE"<<std::endl;
#endif
        for(k=number_columns-1;k>=0;k--){
            x[k+offset] /= data[pointer[k+1]-1];
            for(j=pointer[k];j<pointer[k+1]-1;j++) x[indices[j]+offset] -= data[j]*x[k+offset];
        }
#ifdef VERYVERBOSE
        time_2 = clock();
        time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
        std::cout<<std::endl<<"          triangular_solve_with_smaller_matrix time: "<<time<<std::endl<<std::flush;
#endif
        return;
    }  // end if
    std::cerr<<"matrix_sparse::triangular_solve_with_smaller_matrix: unknown matrix usage"<<std::endl;
    throw iluplusplus_error(OTHER_ERROR);
}

template<class T> void matrix_sparse<T>::triangular_solve_with_smaller_matrix_permute_first(special_matrix_type form, matrix_usage_type use, const index_list& perm, vector_dense<T>& x) const{
#ifdef DEBUG
    if(non_fatal_error(perm.dimension() != rows(),"matrix_sparse::triangular_solve_with_smaller_matrix_permute_first: permutation must have same dimension as number of rows of matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    vector_dense<T> w;
    Integer i;
    Integer offset = x.dimension()-number_rows;
    w.resize_without_initialization(number_rows);
    for(i=0;i<number_rows;i++) w[i]=x[offset+i];
    for(i=0;i<number_rows;i++) x[offset+i]=w[perm[i]];
    triangular_solve_with_smaller_matrix(form,use,x);
}


template<class T> void matrix_sparse<T>::triangular_solve_with_smaller_matrix_permute_last(special_matrix_type form, matrix_usage_type use, const index_list& perm, vector_dense<T>& x) const{
    vector_dense<T> w;
    Integer i;
    Integer offset = x.dimension()-number_rows;
    w.resize_without_initialization(number_rows);
    triangular_solve_with_smaller_matrix(form,use,x);
    for(i=0;i<number_rows;i++) w[i]=x[offset+i];
    for(i=0;i<number_rows;i++) x[offset+i]=w[perm[i]];
}

template<class T> void matrix_sparse<T>::triangular_solve(special_matrix_type form, matrix_usage_type use, const index_list& perm, const vector_dense<T>& b, vector_dense<T>& x) const{
    if(non_fatal_error(!(square_check()),"matrix_sparse::triangular_solve: matrix needs to be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(b.dimension() != number_rows, "matrix_sparse::triangular_solve: size of rhs is incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);

    if (x.dimension() != b.dimension())
        x.resize(b.dimension(),0);

    Integer k;
    Integer j;
     # ifdef VERYVERBOSE
          clock_t time_1,time_2;
          Real time=0.0;
          time_1 = clock();
     #endif
    if ( ((form==PERMUTED_LOWER_TRIANGULAR) && (orientation==ROW) && (use==ID)) ||
         ((form==PERMUTED_UPPER_TRIANGULAR )&& (orientation==COLUMN) && (use==TRANSPOSE)) )
    {
        std::cerr<<"matrix_sparse<T>::triangular_solve with permutation: this particular form of solving should not be needed!!"<<std::endl;
        throw iluplusplus_error(OTHER_ERROR);
        /*  not adapted, this is still the non-permuted form
        for(k=0;k<number_rows;k++) diagonal_is_nonzero = (diagonal_is_nonzero && (k==indices[pointer[k+1]-1]));
        fatal_error(!diagonal_is_nonzero, "matrix_sparse::triangular_solve: pivot must be non-zero.");
        for(k=0;k<number_rows;k++){
            for(j=pointer[k];j<pointer[k+1]-1;j++) x[k]-= data[j]*x[indices[j]];
            x[k] /= data[pointer[k+1]-1];
        }   // end for k
        return;
        */
    } // end if
    if ( ((form==PERMUTED_LOWER_TRIANGULAR) && (orientation==COLUMN) && (use==ID)) ||
         ((form==PERMUTED_UPPER_TRIANGULAR) && (orientation==ROW) && (use==TRANSPOSE)) )
    {   // untested, requires inverse perm (called perm as well)
        # ifdef VERYVERBOSE
            if (use==ID )std::cout<<"      triangular_solve: using: PERMUTED_LOWER_TRIANGULAR,COLUMN,ID"<<std::endl;
            else         std::cout<<"      triangular_solve: using: PERMUTED_UPPER_TRIANGULAR,ROW,TRANSPOSE"<<std::endl;
        #endif
        Integer k,j;
        vector_dense<T> y;
        y=b;
        for(k=0;k<number_columns;k++){
            y[k] /= data[pointer[k]];
            for(j=pointer[k]+1;j<pointer[k+1];j++) y[indices[j]] -= data[j]*y[perm[k]];
         }
        for(k=0;k<number_columns;k++) x[k]=y[perm[k]];
        #ifdef VERYVERBOSE
             time_2 = clock();
             time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
             std::cout<<std::endl<<"          triangular_solve time: "<<time<<std::endl<<std::flush;
        #endif
        return;
    }  // end if

    if ( ((form==PERMUTED_UPPER_TRIANGULAR) && (orientation==ROW) && (use==ID)) ||
         ((form==PERMUTED_LOWER_TRIANGULAR) && (orientation==COLUMN) && (use==TRANSPOSE)) )
    {
        # ifdef VERYVERBOSE
            if (use==ID )std::cout<<"      triangular_solve: using: PERMUTED_UPPER_TRIANGULAR,ROW,ID"<<std::endl;
            else         std::cout<<"      triangular_solve: using: PERMUTED_LOWER_TRIANGULAR,COLUMN,TRANSPOSE"<<std::endl;
        #endif
        vector_dense<T> y;
        y=b;
        for(k=number_rows-1;k>=0;k--){
            for(j=pointer[k]+1;j<pointer[k+1];j++) y[k]-= data[j]*x[indices[j]];
            non_fatal_error(data[pointer[k]]==0,"matrix_sparse::triangular_solve with permutation: pivot must be non-zero.");
            x[perm[k]] = y[k]/data[pointer[k]];
        }   // end for k
        #ifdef VERYVERBOSE
             time_2 = clock();
             time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
             std::cout<<std::endl<<"          triangular_solve time: "<<time<<std::endl<<std::flush;
        #endif
        return;
    }  // end if
    if ( ((form==PERMUTED_UPPER_TRIANGULAR )&& (orientation==COLUMN) && (use==ID)) ||
         ((form==PERMUTED_LOWER_TRIANGULAR) && (orientation==ROW) && (use==TRANSPOSE)))
    {
        std::cerr<<"matrix_sparse<T>::triangular_solve with permutation: this particular form of solving should not be needed!!"<<std::endl;
        throw iluplusplus_error(OTHER_ERROR);
        /* not adapted, this is still the non-permuted form
        for(k=0;k<number_rows;k++) diagonal_is_nonzero = (diagonal_is_nonzero && (k==indices[pointer[k+1]-1]));
        fatal_error(!diagonal_is_nonzero, "matrix_sparse::triangular_solve: pivot must be non-zero.");
        vector_dense<Integer> position_elements_row(number_columns);
        vector_dense<Integer> column_of_elements_row(number_columns);
        vector_dense<Integer> current_position_in_column(number_columns);
        Integer nnz_row=0;
        for(k=number_rows-1;k>=0;k--){
            nnz_row=0;
            current_position_in_column[k]=pointer[k+1]-1;
            for(j=k+1;j<number_columns;j++){
                if((current_position_in_column[j]>pointer[j])&&(indices[current_position_in_column[j]-1]==k)){
                    current_position_in_column[j]--;
                    position_elements_row[nnz_row]=current_position_in_column[j];
                    column_of_elements_row[nnz_row]=j;
                    nnz_row++;
                }
            }
            for(j=0;j<nnz_row;j++) x[k]-= data[position_elements_row[j]]*x[column_of_elements_row[j]];
            x[k] /= data[pointer[k+1]-1];
        }  // end for k
        return;
        */
    }  // end if
    std::cerr<<"matrix_sparse::triangular_solve with permutation: unknown matrix usage, trying without permutation"<<std::endl;
    triangular_solve(form,use,b,x);
     #ifdef VERYVERBOSE
          time_2 = clock();
          time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
          std::cout<<std::endl<<"          triangular_solve "<<time<<std::endl<<std::flush;
     #endif
}

template<class T> void matrix_sparse<T>::triangular_solve(special_matrix_type form, matrix_usage_type use, const index_list& perm, vector_dense<T>& x) const {
    vector_dense<T> b = x;
    triangular_solve(form, use, perm, b, x);
}


template<class T> void matrix_sparse<T>::expand_kernel(const matrix_sparse<T>& A, const vector_dense<T>& b, const vector_dense<T>& c, T beta, T gamma, Integer row_pos, Integer col_pos){
    if(non_fatal_error(A.columns() != b.dimension() || A.rows() != c.dimension(),"matrix_sparse<T>::expand_kernel: arguments have incompatible dimension.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(beta == 0.0 || gamma == 0.0,"matrix_sparse<T>::expand_kernel: beta and gamma must be non-zero.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    vector_dense<T> x,y;
    T delta,z;
    A.matrix_vector_multiplication(ID,b,x);
    delta = c*x; // scalar multiplication ( delta = (c^T)*A*b
    z = delta/(beta*gamma);
    x.scale(-1.0/beta);
    A.matrix_vector_multiplication(TRANSPOSE,c,y);
    y.scale(-1.0/gamma);
    insert(A,y,x,z,row_pos,col_pos,-1.0);
}

template<class T> void matrix_sparse<T>::expand_kernel(const matrix_sparse<T>& A, const vector_dense<T>& b, const vector_dense<T>& c, T beta, T gamma, Integer row_pos, Integer col_pos, vector_dense<T>& bnew, vector_dense<T>& cnew){
    expand_kernel(A,b,c,beta,gamma,row_pos,col_pos);
    bnew.insert(b,col_pos,beta);
    cnew.insert(c,row_pos,gamma);
}

template<class T> void matrix_sparse<T>::regularize(const matrix_sparse<T>& A,const vector_dense<T>& b, const vector_dense<T>& c, T d,Integer row_pos, Integer col_pos){
    insert(A,c,b,d,row_pos,col_pos,-1.0);
}

template<class T> void matrix_sparse<T>::regularize_with_rhs(const matrix_sparse<T>& A,const vector_dense<T>& b, const vector_dense<T>& c, T d,Integer row_pos, Integer col_pos, const vector_dense<T>& old_rhs, vector_dense<T>& new_rhs){
    vector_dense<T> h;
    regularize(A,b,c,d,row_pos,col_pos);
    h.vector_addition(old_rhs,b);
    new_rhs.insert(h,row_pos,d);
}




template<class T> void matrix_sparse<T>::triangular_drop(special_matrix_type form, const matrix_sparse<T>& M, Integer max_fill_in, Real tau){
    //fatal_error(!(square_check()),"matrix_sparse::triangular_drop: matrix needs to be square.");
    Integer k,i;
    vector_dense<T> w(M.dim_along_orientation());
    index_list list;
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>M.dim_along_orientation()) max_fill_in = M.dim_along_orientation();
    Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (M.columns()-max_fill_in)*max_fill_in;
    if (tau > 500) tau = 0.0;
    else tau=-log10(tau);
    reformat(M.rows(),M.columns(),reserved_memory,M.orient());
    if (
            (((form==LOWER_TRIANGULAR)||(form==PERMUTED_LOWER_TRIANGULAR))&& M.orient()==ROW) ||
            (((form==UPPER_TRIANGULAR)||(form==PERMUTED_UPPER_TRIANGULAR))&& M.orient()==COLUMN)
       )
    { // begin if
        for(k=0;k<M.dim_against_orientation();k++){
            // copy data to w
            for(i=0;i<M.pointer[k+1]-M.pointer[k]-1;i++)
                w[i]=M.data[M.pointer[k]+i];
            // take largest elements of w, store result in list
            w.take_largest_elements_by_abs_value_with_threshold(list,max_fill_in-1,tau,0, M.pointer[k+1]-M.pointer[k]-1);
            // copy elements to *this
            for(i=0;i<list.dimension();i++){
                data[pointer[k]+i]=M.data[M.pointer[k]+list[i]];
                indices[pointer[k]+i]=M.indices[M.pointer[k]+list[i]];
            }
            // copy pivot
            data[pointer[k]+list.dimension()]=M.data[M.pointer[k+1]-1];
            indices[pointer[k]+list.dimension()]=M.indices[M.pointer[k+1]-1];
            // do pointer
            pointer[k+1]=pointer[k]+list.dimension()+1;
        } // end for k
        compress();
        return;
    } // end if
    if (
            (((form==LOWER_TRIANGULAR)||(form==PERMUTED_LOWER_TRIANGULAR))&& M.orient()==COLUMN) ||
            (((form==UPPER_TRIANGULAR)||(form==PERMUTED_UPPER_TRIANGULAR))&& M.orient()==ROW)
       )
    { // begin if
        for(k=0;k<M.dim_against_orientation();k++){
            // copy data to w
            for(i=0;i<M.pointer[k+1]-M.pointer[k]-1;i++)
                w[i]=M.data[M.pointer[k]+1+i];
            // take largest elements of w, store result in list
            w.take_largest_elements_by_abs_value_with_threshold(list,max_fill_in-1,tau,0, M.pointer[k+1]-M.pointer[k]-1);
            // copy elements to *this
            for(i=0;i<list.dimension();i++){
                data[pointer[k]+1+i]=M.data[M.pointer[k]+1+list[i]];
                indices[pointer[k]+1+i]=M.indices[M.pointer[k]+1+list[i]];
            }
            // copy pivot
            data[pointer[k]]=M.data[M.pointer[k]];
            indices[pointer[k]]=M.indices[M.pointer[k]];
            // do pointer
            pointer[k+1]=pointer[k]+list.dimension()+1;
        } // end for k
        compress();
        return;
    } // end if
    // default case, i.e. no triangular matrix
    std::cerr<<"matrix_sparse<T>::triangular_drop: matrix is not triangular. *this is the same as M."<<std::endl;
    *this=M;
    return;
}


template<class T> void matrix_sparse<T>::weighted_triangular_drop_along_orientation(special_matrix_type form, const matrix_sparse<T>& M, const vector_dense<T> weights, Integer max_fill_in, Real tau){
    Integer size=M.rows();
#ifdef DEBUG
    if(non_fatal_error(!(square_check()),"matrix_sparse::weighted_triangular_drop_along_orientation: matrix needs to be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error((weights.dimension() != size),"matrix_sparse::weighted_triangular_drop_along_orientation: weights must have same size as matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    Integer k,i;
    vector_dense<T> w(size);
    index_list list;
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>size) max_fill_in = size;
    Integer reserved_memory = max_fill_in*(max_fill_in+1)/2 + (size-max_fill_in)*max_fill_in;
    if (tau > 500) tau = 0.0;
    else tau=-log10(tau);
    reformat(size,size,reserved_memory,M.orient());
    if (
            (((form==LOWER_TRIANGULAR)||(form==PERMUTED_LOWER_TRIANGULAR))&& M.orient()==ROW) ||
            (((form==UPPER_TRIANGULAR)||(form==PERMUTED_UPPER_TRIANGULAR))&& M.orient()==COLUMN)
       )
    { // begin if
        for(k=0;k<M.dim_against_orientation();k++){
            // copy data to w
            for(i=0;i<M.pointer[k+1]-M.pointer[k]-1;i++)
                w[i]=M.data[M.pointer[k]+i]*weights.read(M.indices[M.pointer[k]+i]);
            // take largest elements of w, store result in list
            w.take_largest_elements_by_abs_value_with_threshold(list,max_fill_in-1,tau,0, M.pointer[k+1]-M.pointer[k]-1);
            // copy elements to *this
            for(i=0;i<list.dimension();i++){
                data[pointer[k]+i]=M.data[M.pointer[k]+list[i]];
                indices[pointer[k]+i]=M.indices[M.pointer[k]+list[i]];
            }
            // copy pivot
            data[pointer[k]+list.dimension()]=M.data[M.pointer[k+1]-1];
            indices[pointer[k]+list.dimension()]=M.indices[M.pointer[k+1]-1];
            // do pointer
            pointer[k+1]=pointer[k]+list.dimension()+1;
        } // end for k
        compress();
        return;
    } // end if
    if (
            (((form==LOWER_TRIANGULAR)||(form==PERMUTED_LOWER_TRIANGULAR))&& M.orient()==COLUMN) ||
            (((form==UPPER_TRIANGULAR)||(form==PERMUTED_UPPER_TRIANGULAR))&& M.orient()==ROW)
       )
    { // begin if
        for(k=0;k<M.dim_against_orientation();k++){
            // copy data to w
            for(i=0;i<M.pointer[k+1]-M.pointer[k]-1;i++)
                w[i]=M.data[M.pointer[k]+1+i]*weights.read(M.indices[pointer[k]+1+i]);
            // take largest elements of w, store result in list
            w.take_largest_elements_by_abs_value_with_threshold(list,max_fill_in-1,tau,0, M.pointer[k+1]-M.pointer[k]-1);
            // copy elements to *this
            for(i=0;i<list.dimension();i++){
                data[pointer[k]+1+i]=M.data[M.pointer[k]+1+list[i]];
                indices[pointer[k]+1+i]=M.indices[M.pointer[k]+1+list[i]];
            }
            // copy pivot
            data[pointer[k]]=M.data[M.pointer[k]];
            indices[pointer[k]]=M.indices[M.pointer[k]];
            // do pointer
            pointer[k+1]=pointer[k]+list.dimension()+1;
        } // end for k
        compress();
        return;
    } // end if
    // default case, i.e. no triangular matrix
    std::cerr<<"matrix_sparse<T>::weighted_triangular_drop_along_orientation: matrix is not triangular. *this is the same as M."<<std::endl;
    *this=M;
    return;
  }


template<class T> void matrix_sparse<T>::weighted_triangular_drop_against_orientation(special_matrix_type form, const matrix_sparse<T>& M, const vector_dense<T> weights, Integer max_fill_in, Real tau){
    Integer size=M.rows();
#ifdef DEBUG
    if(non_fatal_error(!(square_check()),"matrix_sparse::weighted_triangular_drop_against_orientation: matrix needs to be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error((weights.dimension() != size),"matrix_sparse::weighted_triangular_drop_against_orientation: weights must have same size as matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    Integer k,i,number_elements_in_column,index_of_examined_element;
    Integer current_column=0; // notation for M being a ROW matrix and M needing to be read by columns
    vector_dense<T> w(size);
    index_list list;
    index_list position_in_matrix_list;
    index_list marker;
    position_in_matrix_list.resize_without_initialization(size);
    marker.resize_without_initialization(size);
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>size) max_fill_in = size;
    if (tau > 500) tau = 0.0;
    else tau=-log10(tau);
    reformat(M.rows(),M.columns(),M.non_zeroes(),M.orient());
    if (
            (((form==LOWER_TRIANGULAR)||(form==PERMUTED_LOWER_TRIANGULAR))&& M.orient()==ROW) ||
            (((form==UPPER_TRIANGULAR)||(form==PERMUTED_UPPER_TRIANGULAR))&& M.orient()==COLUMN)
       )
    { // begin if
        // make *this a zero matrix.
        for (i=0;i<M.nnz;i++) data[i] = 0.0;
        for (i=0;i<M.nnz;i++) indices[i] = M.indices[i];
        for (i=0;i<M.pointer_size;i++) pointer[i] = M.pointer[i];
        for (i=0;i<M.pointer_size-1;i++) data[M.pointer[i+1]-1]=M.data[M.pointer[i+1]-1]; // copy pivots
        // copy current column to w (weighted);
        // copy the corresponding positions in the matrix to "position_in_matrix_list"
        // update marker and number_elements_in_column if an element is found
        for(k=0;k<size;k++) marker[k]=pointer[k];
        for(k=0;k<size;k++){
            current_column=M.indices[M.pointer[k+1]-1]; // pivot
            number_elements_in_column=0;
            // copy data to w
            for(i=k+1;i<size;i++){ // starting at k+1, i.e. skipping pivot
                index_of_examined_element=M.indices[marker[i]];
                if(index_of_examined_element==current_column && marker[i]<M.pointer[i+1]){ // i.e. marker indicates an element found in the column
                    w[number_elements_in_column]=M.data[marker[i]]*weights[i];
                    position_in_matrix_list[number_elements_in_column]=marker[i];
                    number_elements_in_column++;
                    marker[i]++;
                }  // end if
            }  // end for i
            // take largest elements of w, store result in list
            w.take_largest_elements_by_abs_value_with_threshold(list,max_fill_in-1,tau,0,number_elements_in_column);
            // list contains elements to be kept, so zero the rest
            for(i=0;i<list.dimension();i++) data[position_in_matrix_list[list[i]]]=M.data[position_in_matrix_list[list[i]]];
        } // end for k
        compress();
        return;
    } // end if
    if (
            (((form==LOWER_TRIANGULAR)||(form==PERMUTED_LOWER_TRIANGULAR))&& M.orient()==COLUMN) ||
            (((form==UPPER_TRIANGULAR)||(form==PERMUTED_UPPER_TRIANGULAR))&& M.orient()==ROW)
       )
    { // begin if
        // make *this a zero matrix.
        for (i=0;i<M.nnz;i++) data[i] = 0.0;
        for (i=0;i<M.nnz;i++) indices[i] = M.indices[i];
        for (i=0;i<M.pointer_size;i++) pointer[i] = M.pointer[i];
        for (i=0;i<M.pointer_size-1;i++) data[M.pointer[i]]=M.data[M.pointer[i]]; // copy pivots
        // copy current column to w (weighted);
        // copy the corresponding positions in the matrix to "position_in_matrix_list"
        // update marker and number_elements_in_column if an element is found
        for(k=0;k<size;k++){
            current_column=M.indices[M.pointer[k]];
            marker[k]=M.pointer[k]+1; // skip the pivot, which is located in M.pointer[k]
            number_elements_in_column=0;
            // copy data to w
            for(i=0;i<k;i++){
                index_of_examined_element=M.indices[marker[i]];
                if(index_of_examined_element==current_column && marker[i]<M.pointer[i+1]){ // i.e. marker indicates an element found in the column
                    w[number_elements_in_column]=M.data[marker[i]]*weights[i];
                    position_in_matrix_list[number_elements_in_column]=marker[i];
                    number_elements_in_column++;
                    marker[i]++;  
                }  // end if
            }  // end for i   
            // take largest elements of w, store result in list
            w.take_largest_elements_by_abs_value_with_threshold(list,max_fill_in-1,tau,0,number_elements_in_column);
            // list contains elements to be kept, so zero the rest
            for(i=0;i<list.dimension();i++) data[position_in_matrix_list[list[i]]]=M.data[position_in_matrix_list[list[i]]]; 
        } // end for k
        compress();
        return;
    } // end if
    // default case, i.e. no triangular matrix
    std::cerr<<"matrix_sparse<T>::weighted_triangular_drop_against_orientation: matrix is not triangular. *this is the same as M."<<std::endl;
    *this=M;
    return;
  }

template<class T> void matrix_sparse<T>::weighted_triangular_drop(special_matrix_type form, const matrix_sparse<T>& M, const vector_dense<T> weights, orientation_type o, Integer max_fill_in, Real tau){
    if (o == M.orient())
        weighted_triangular_drop_along_orientation(form,M,weights,max_fill_in,tau);
    else
        weighted_triangular_drop_against_orientation(form,M,weights,max_fill_in,tau);
  }

//***********************************************************************************************************************
// Class matrix_sparse: Preconditioners                                                                                 *
//***********************************************************************************************************************

template<class T> bool matrix_sparse<T>::ILUCDP(const matrix_sparse<T>& Arow, const matrix_sparse<T>& Acol, matrix_sparse<T>& U, index_list& perm, index_list& permrows, Integer max_fill_in, Real threshold, Real perm_tol,  Integer bpr, Integer& zero_pivots, Real& time_self, Real mem_factor){
    clock_t time_begin, time_end;
    time_begin=clock();
    if (threshold > 500.0) threshold=0.0;
    else threshold=std::exp(-threshold*std::log(10.0));
    if (perm_tol > 500.0) perm_tol=0.0;
    else perm_tol=std::exp(-perm_tol*std::log(10.0));
#ifdef VERBOSE
    clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_8,time_9;
    Real time_init=0.0;
    Real time_read=0.0;
    Real time_calc_L=0.0;
    Real time_scu_L=0.0;  // sorting, copying, updating access information
    Real time_calc_U=0.0;
    Real time_scu_U=0.0;
    Real time_zeroset=0.0;
    Real time_compress=0.0;
    Real time_resort=0.0;
    time_0 = clock();
#endif
    if(non_fatal_error(!Arow.square_check(),"matrix_sparse::ILUCDP: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(!Acol.square_check(),"matrix_sparse::ILUCDP: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(Acol.rows()!=Arow.rows(),"matrix_sparse::ILUCDP: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = Acol.columns();
    Integer a,b,k,i,j,p,current_row_col_U,current_col_row_L;
    Integer h,pos, selected_row;
    T current_data_row_L,current_data_col_U;
    zero_pivots=0;
    Real norm_L,norm_U; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    vector_sparse_dynamic<T> w, z;
    vector_dense<bool> non_pivot, unused_rows;
    vector_dense<Integer> numb_el_row_L, pointer_num_el_row_L;
    index_list list_L, list_U;
    index_list inverse_perm, inverse_permrows;
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*Acol.non_zeroes());
    array<Integer> linkU, rowU, startU, linkL, colL, startL;
    linkU.erase_resize_data_field(reserved_memory);
    rowU.erase_resize_data_field(reserved_memory);
    startU.erase_resize_data_field(n);
    linkL.erase_resize_data_field(reserved_memory);
    colL.erase_resize_data_field(reserved_memory);
    startL.erase_resize_data_field(n);
    U.reformat(n,n,reserved_memory,ROW);
    reformat(n,n,reserved_memory,COLUMN);
    perm.resize(n);
    permrows.resize(n);
    inverse_perm.resize(n);
    inverse_permrows.resize(n);
    non_pivot.resize(n,true); 
    unused_rows.resize(n,true);
    numb_el_row_L.resize(n,0);
    pointer_num_el_row_L.resize(n+2,n);
    w.resize(n);
    z.resize(n);
    pointer_num_el_row_L[0]=0;
    for(k=0;k<n;k++) startU[k]=-1;
    for(k=0;k<n;k++) startL[k]=-1;
    // (1.) begin for k
#ifdef VERBOSE
    time_1 = clock();
    time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
#endif
    for(k=0;k<n;k++){
        if (k == bpr) perm_tol = 1.0;  // permute always
        //if (k == bpr) threshold = 0.0;
        //if (k == bpr) threshold *= (0.1>threshold) ? 0.1 : threshold;
#ifdef VERBOSE
        time_2=clock();
#endif
        // (2.) initialize z
        selected_row = permrows[k];
        unused_rows[selected_row]=false;
        z.zero_reset();
#ifdef VERBOSE
        time_3=clock();
        time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
        // read row of A
        for(i=Arow.pointer[selected_row];i<Arow.pointer[selected_row+1];i++){
            if(non_pivot[Arow.indices[i]]) z[Arow.indices[i]] = Arow.data[i];
        }     // end for i
#ifdef VERBOSE
        time_4=clock();
        time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
#endif
        // (3.) begin while
        h=startL[selected_row]; // h=startL[permrows[k]];
        while(h!=-1){
            current_col_row_L=colL[h];
            current_data_row_L=data[h];
            h=linkL[h];
            for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
            } // end for
        }   // (5.) end while
#ifdef VERBOSE
        time_5=clock();
        time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif
        // (6.) sort and copy data to U; update information for accessing columns of U
        z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,threshold,perm[k],perm_tol);
        // dropping too stringent?
        if(list_U.dimension()==0){
            if(threshold>0.0)
#ifdef VERBOSE
                std::cout<<"Dropping too stringent, selecting elements without threshold."<<std::endl;
#endif
            z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,0.0,perm[k],perm_tol);
        }
        // still no non-zero elements?
        if(list_U.dimension()==0){
#ifdef VERBOSE
            std::cout<<"Obtained a zero row, setting an arbitrary element to 1."<<std::endl;
#endif
            zero_pivots++;
            z[perm[k]]=1.0;
            list_U.resize(1);
            list_U[0]=perm[k];
        } // end if
        if(U.pointer[k]+list_U.dimension()>reserved_memory){
            std::cerr<<"matrix_sparse::ILUCDP: memory reserved was insufficient. Returning 0x0 matrices and permutations of dimension 0."<<std::endl;
            perm.resize(0);
            permrows.resize(0);
            reformat(0,0,0,COLUMN);
            U.reformat(0,0,0,ROW);
            return false;
        }
        // copy data, update access information.
        // copy pivot
        U.data[U.pointer[k]]=z.read(list_U[list_U.dimension()-1]);
        U.indices[U.pointer[k]]=list_U[list_U.dimension()-1];
        for(j=1;j<list_U.dimension();j++){
            pos=U.pointer[k]+j;
            U.data[pos]=z.read(list_U[list_U.dimension()-1-j]);
            U.indices[pos]=list_U[list_U.dimension()-1-j];
            h=startU[U.indices[pos]];
            startU[U.indices[pos]]=pos;
            linkU[pos]=h;
            rowU[pos]=k;
        }
        U.pointer[k+1]=U.pointer[k]+list_U.dimension();
        if(U.data[U.pointer[k]]==0){
            std::cerr<<"matrix_sparse::ILUCDP: Pivot is zero, because pivoting was not permitted. Preconditioner does not exist.Returning 0x0 matrices and permutations of dimension 0. "<<std::endl;
            std::cout<<"dim list_U "<<list_U.dimension()<<std::endl;
            std::cout<<"last element corresponding to pivot: "<<z[perm[k]]<<std::endl;
            perm.resize(0);
            permrows.resize(0);
            reformat(0,0,0,COLUMN);
            U.reformat(0,0,0,ROW);
            return false;
        }
        // store positions of columns of U, but without pivot
        // update non-pivots.
        // (7.) update permutations
        p=inverse_perm[U.indices[U.pointer[k]]];
        inverse_perm.switch_index(perm[k],U.indices[U.pointer[k]]);
        perm.switch_index(k,p);
        non_pivot[U.indices[U.pointer[k]]]=false;
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
#endif
        // (8.) read w
        w.zero_reset();
#ifdef VERBOSE
        time_7=clock();
        time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
#endif
        // read column of A
        for(i=Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
            if(unused_rows[Acol.indices[i]])
                w[Acol.indices[i]] = Acol.data[i];
        }     // end for i
#ifdef VERBOSE
        time_8=clock();
        time_read += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
#endif
        // (9.) begin while
        h=startU[perm[k]];
        while(h!=-1){
            current_row_col_U=rowU[h];
            current_data_col_U=U.data[h];
            h=linkU[h];
            // (10.) w = w - U(i,perm(k))*l_i
            for(j=pointer[current_row_col_U];j<pointer[current_row_col_U+1];j++){
                if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U*data[j];
            } // end for
        }   // (11.) end while
#ifdef VERBOSE
        time_9=clock();
        time_calc_L += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
#endif
        // (12.) sort and copy data to L
        // sort
        w.take_largest_elements_by_abs_value_with_threshold(norm_L,list_L,max_fill_in-1,threshold,0,n);
        if(pointer[k]+list_L.dimension()+1>reserved_memory){
            std::cerr<<"matrix_sparse::ILUCDP: memory reserved was insufficient. Returning 0x0 matrices and permutations of dimension 0."<<std::endl;
            perm.resize(0);
            permrows.resize(0);
            reformat(0,0,0,COLUMN);
            U.reformat(0,0,0,ROW);
            return false;
        }
        // copy data
        data[pointer[k]]=1.0;
        indices[pointer[k]]=selected_row;
        for(j=0;j<list_L.dimension();j++){
            pos = pointer[k]+j+1;
            data[pos] = w.read(list_L[j])/U.data[U.pointer[k]];
            b = indices[pos] = list_L[j];
            h=startL[b];
            startL[b]=pos;
            linkL[pos]=h;
            colL[pos]=k;
            // begin updating fields for number elements of row of L
            if (b > bpr) {
                b = inverse_permrows[b];
                a = --pointer_num_el_row_L[++numb_el_row_L[b]];
                inverse_permrows.switch_index(permrows[a],permrows[b]);
                permrows.switch_index(a,b);
                numb_el_row_L.switch_entry(a,b);
            }
            // end updating fields
        } // end for j
        // sort permrows if necessary, i.e. if num_el_row_L increases at next iteration.
        if(pointer_num_el_row_L[numb_el_row_L[k]+1] == k+1){ 
            permrows.quicksort_with_inverse(inverse_permrows,pointer_num_el_row_L[numb_el_row_L[k]+1],pointer_num_el_row_L[numb_el_row_L[k]+2]-1);}
        // end sorting
        pointer[k+1]=pointer[k]+list_L.dimension()+1;
#ifdef VERBOSE
        time_0=clock();
        time_scu_L += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
#endif
    }  // (13.) end for k
#ifdef VERBOSE
    time_2 = clock();
#endif
    compress();
    U.compress();
#ifdef VERBOSE
    time_3=clock();
    time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef VERBOSE
    time_4=clock();
    time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
    std::cout<<"    ILUCDP-Times: "<<std::endl;
    std::cout<<"        initialization:                           "<<time_init<<std::endl;
    std::cout<<"        reading matrix:                           "<<time_read<<std::endl;
    std::cout<<"        sparse zero set:                          "<<time_zeroset<<std::endl;
    std::cout<<"        calculating L:                            "<<time_calc_L<<std::endl;
    std::cout<<"        calculating U:                            "<<time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info L: "<<time_scu_L<<std::endl;
    std::cout<<"        sorting, copying, updating access info U: "<<time_scu_U<<std::endl;
    std::cout<<"        compressing:                              "<<time_compress<<std::endl;
    std::cout<<"        resorting:                                "<<time_resort<<std::endl;
    std::cout<<"      Total times:"<<std::endl;
    std::cout<<"        calculations:                             "<<time_calc_L+time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info:   "<<time_scu_L+time_scu_U<<std::endl;
    std::cout<<"        other administration:                     "<<time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Grand total                                 "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
#endif
    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    return true;
}


template<class T> bool matrix_sparse<T>::partialILUCDP(const matrix_sparse<T>& Arow, const matrix_sparse<T>& Acol, matrix_sparse<T>& Anew, const iluplusplus_precond_parameter& IP, bool force_finish, matrix_sparse<T>& U, vector_dense<T>& Dinv, index_list& perm, index_list& permrows, index_list& inverse_perm, index_list& inverse_permrows,Integer last_row_to_eliminate, Real threshold, Integer bp, Integer bpr, Integer epr, Integer& zero_pivots, Real& time_self, Real mem_factor, Real& total_memory_allocated, Real& total_memory_used){
    time_self = 0.0;
    total_memory_allocated = 0.0;
    Integer n = Acol.columns();
    if(!Arow.square_check()){
        std::cerr<<"matrix_sparse::partialILUCDP: argument matrix must be square. Returning 0x0 matrices."<<std::endl<<std::flush;
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        perm.resize(0);
        permrows.resize(0);
        inverse_perm.resize(0);
        inverse_permrows.resize(0);
        return false;
    }
    if(n==0){
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        perm.resize(0);
        permrows.resize(0);
        inverse_perm.resize(0);
        inverse_permrows.resize(0);
        return true;
    }
    clock_t time_begin, time_end;
    time_begin=clock();
    Integer max_fill_in;
    Integer bandwidth, bandwidth_L, bandwidth_U;
    Integer a,b,k,i,j,p,current_row_col_U,current_col_row_L;//help;
    Integer h,pos, selected_row;
    Integer pos_pivot=-1; // is set later
    T current_data_row_L,current_data_col_U;
    T val_larg_el = 0.0;
    zero_pivots=0;
    Real norm_U,norm; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    Real max_inv_piv=0.0;
    Real threshold_Schur_factor = std::exp(-IP.get_THRESHOLD_SHIFT_SCHUR()*std::log(10.0));
    Real post_fact_threshold;
    Real perm_tol = IP.get_perm_tol();
    bool end_level_now = false;  // indicates if next iteration in k-loop starts a new level, i.e. calculations of Schur complement begin.
    bool eliminate = true;       // indicates if standard elimination is being performed or Schur complement being calculated
    //bool pivoting = true;        // indicates if columns are pivoted in a particular step
    T pivot = 0.0;
    Integer k_Anew,n_Anew=0; // set later
    Integer reserved_memory_Anew=0; // will be set later
    T  xplus, xminus, yplus, yminus,vi;
    Real nuplus,numinus;
    Integer nplus, nminus,pk;
    bool use_improved_SCHUR = (IP.get_SCHUR_COMPLEMENT()>0);
    bool use_weightsLU = IP.get_USE_WEIGHTED_DROPPING() || IP.get_USE_WEIGHTED_DROPPING2();
    bool use_norm_row_U=false;
    Real weightL, weightU;
    Real move_level_parameter=0;
    Integer reserved_memory_L;
    Integer reserved_memory_U;
    Integer reserved_memory_droppedU;
#ifdef VERBOSE
    clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_8,time_9;
    Real time_init=0.0;
    Real time_read=0.0;
    Real time_calc_L=0.0;
    Real time_scu_L=0.0;  // sorting, copying, updating access information
    Real time_calc_U=0.0;
    Real time_scu_U=0.0;
    Real time_calc_Anew=0.0;
    Real time_scu_Anew=0.0;
    Real time_zeroset=0.0;
    Real time_compress=0.0;
    Real time_resort=0.0;
    Real time_dropping=0.0;
    time_0 = clock();
#endif
    if(IP.get_MAX_FILLIN_IS_INF())  max_fill_in = n;
    else max_fill_in = IP.get_fill_in();
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    if(IP.get_DROP_TYPE_L()==4||IP.get_DROP_TYPE_U()==4) bandwidth=Arow.bandwidth(); else bandwidth=0;
    switch (IP.get_DROP_TYPE_L()){
        case 3: bandwidth_L = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_L = bandwidth; break;
        default: bandwidth_L = 0;
    }
    switch (IP.get_DROP_TYPE_U()){
        case 3: bandwidth_U = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_U = bandwidth; break;
        default: bandwidth_U = 0;
    }
    if (threshold > 500.0) threshold=0.0;
    else threshold=std::exp(-threshold*std::log(10.0));
    if (perm_tol > 500.0) perm_tol=0.0;
    else perm_tol=std::exp(-perm_tol*std::log(10.0));
    if  (IP.get_POST_FACT_THRESHOLD() > 500.0) post_fact_threshold = 0.0; 
    else post_fact_threshold = threshold*std::exp(-IP.get_POST_FACT_THRESHOLD()*std::log(10.0));
    if(last_row_to_eliminate+1>n) last_row_to_eliminate = n-1;
    if(last_row_to_eliminate<0) last_row_to_eliminate = 0;
    if(epr<0)  epr=0;
    if(epr>=n) epr=n-1;
    if(bpr<0)  bpr=0;
    if(bpr>=n) bpr=n-1;
    reserved_memory_L = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Acol.non_zeroes()));
    reserved_memory_U = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Acol.non_zeroes()));
    reserved_memory_droppedU = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Acol.non_zeroes()));
#ifdef STATISTICS
    vector_dense<Integer> L_total,L_kept,U_total,U_kept;
    Real average_total,average_kept,average_prop, min_prop, max_prop, stand_dev_kept, stand_dev_prop, stand_dev_total;
    Integer min_total, max_total, min_kept, max_kept,help;
    Real sum1, sum2, sum3, prop;
#endif
    vector_sparse_dynamic<T> w,z;
    vector_dense<bool> non_pivot, unused_rows;
    vector_dense<Integer> numb_el_row_L, pointer_num_el_row_L;
    vector_dense<Real> norm_row_U;
    sorted_vector row_reorder_weight;
    index_list list_L, list_U;
    index_list rejected_L, rejected_U;
    matrix_sparse<T> droppedU;  // matrix containing dropped elements of U. Used to calculate an improved Schur complement.
    array< std::queue<T> > droppedL_data;
    array< std::queue<Integer> > droppedL_colindex;
    Real droppedL_data_memory = 0.0;
    Real droppedL_colindex_memory = 0.0;
    vector_dense<T> vxL,vyL,vxU,vyU,xL,yL,xU,yU;
    vector_dense<Real> weightsL,weightsU;
    array<Integer> linkU, rowU, startU, linkL, colL, startL;
    Dinv.resize(n,1.0);
    perm.resize(n);
    permrows.resize(n);
    inverse_perm.resize(n);
    inverse_permrows.resize(n);
    w.resize(n); z.resize(n);
    non_pivot.resize(n,true);
    unused_rows.resize(n,true);
    numb_el_row_L.resize(n,0);
    pointer_num_el_row_L.resize(n+2,epr+1);
    linkU.erase_resize_data_field(reserved_memory_U); //h=link[startU[i]]] points to second 2nd element, link[h] to next, etc.
    rowU.erase_resize_data_field(reserved_memory_U);  // row indices of elements of U.data.
    startU.erase_resize_data_field(n); // startU[i] points to start of points to an index of data belonging to column i 
    linkL.erase_resize_data_field(reserved_memory_L); //h=link[startL[i]]] points to second 2nd element, link[h] to next, etc.
    colL.erase_resize_data_field(reserved_memory_L);  // column indices of elements of data.
    startL.erase_resize_data_field(n); // startL[i] points to start of points to an index of data belonging to row i 
    U.reformat(n,n,reserved_memory_U,ROW);
    reformat(n,n,reserved_memory_L,COLUMN);
    if(IP.get_FINAL_ROW_CRIT() <= -1){ row_reorder_weight.resize(n); if(n>0) row_reorder_weight.remove(0);}
    if(IP.get_FINAL_ROW_CRIT() == -3 || IP.get_FINAL_ROW_CRIT() == -4) {use_norm_row_U=true; norm_row_U.resize(n,0.0);}
    if(IP.get_USE_INVERSE_DROPPING()){
        xL.resize(n,0); yL.resize(n,0); vxL.resize(n,0);vyL.resize(n,0);xU.resize(n,0);yU.resize(n,0);vxU.resize(n,0);vyU.resize(n,0);
    }
    if(use_weightsLU){
        weightsL.resize(n,IP.get_INIT_WEIGHTS_LU()); weightsU.resize(n,IP.get_INIT_WEIGHTS_LU());  // set equal to 1 for diagonal element
    }
    if(use_improved_SCHUR){
        droppedU.reformat(n,n,reserved_memory_droppedU,ROW);
        droppedL_data.resize(n);
        droppedL_colindex.resize(n);
    }
#ifdef STATISTICS
    L_total.resize(n,0); L_kept.resize(n,0); U_total.resize(n,0); U_kept.resize(n,0);
#endif
    pointer_num_el_row_L[0]=0;
    for(k=0;k<n;k++) startU[k]=-1;
    for(k=0;k<n;k++) startL[k]=-1;
    // stores dropped elements of L by rows
    // (1.) begin for k
#ifdef VERBOSE
    time_1 = clock();
    time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
#endif
    for(k=0;k<n;k++){
        if (IP.get_BEGIN_TOTAL_PIV() && k == bp){ perm_tol = 1.0;}// permute always
#ifdef VERBOSE
        time_2=clock();
#endif
        // (2.) initialize z
        selected_row = permrows[k];
        unused_rows[selected_row]=false;
        z.zero_reset();
#ifdef VERBOSE
        time_3=clock();
        time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
        // read row of A
        for(i=Arow.pointer[selected_row];i<Arow.pointer[selected_row+1];i++){
            if(non_pivot[Arow.indices[i]]) z[Arow.indices[i]] = Arow.data[i];
        }     // end for i
#ifdef VERBOSE
        time_4=clock();
        time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
#endif
        // (3.) begin while
        h=startL[selected_row]; // h=startL[permrows[k]];
        // do standard elimination
        while(h!=-1){
            current_col_row_L=colL[h];
            current_data_row_L=data[h];
            h=linkL[h];
            for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L/Dinv[current_col_row_L]*U.data[j];
                // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
            } // end for
            // use improved Schur?
            if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of L, small of U):
                for(j=droppedU.pointer[current_col_row_L];j<droppedU.pointer[current_col_row_L+1];j++){
                    if(non_pivot[droppedU.indices[j]]) z[droppedU.indices[j]] -= current_data_row_L/Dinv[current_col_row_L]*droppedU.data[j];
                    // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
                } // end for
            } //end if
        }   // (5.) end while
        if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of U, small of L):
            while(droppedL_data[k].size()>0){
                current_col_row_L = droppedL_colindex[k].front();  // read
                current_data_row_L = droppedL_data[k].front();
                for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                    if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L/Dinv[current_col_row_L]*U.data[j];
                    // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
                } // end for
                droppedL_colindex[k].pop();  // remove
                droppedL_data[k].pop();
            }  // end while
        } // end if
#ifdef VERBOSE
        time_5=clock();
        if(eliminate) time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
        else time_calc_Anew += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif

        /*
           if(eliminate){  // select potential pivot
           val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
           if (std::abs(val_larg_el*perm_tol)>std::abs(z.read(perm[k])) && pos_pivot>=0){ 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           pivoting = false;
           pivot = z.read(perm[k]);
           }
           }
           */

        /*
           if(eliminate){  // select potential pivot
           val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
           if(non_pivot[selected_row]){  // not pivoting is with respect to the diagonal element of the selected row (if possible)
           if (std::abs(val_larg_el*perm_tol)>std::abs(z.read(selected_row)) && pos_pivot>=0){
           std::cout<<"*1 "; 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           std::cout<<"*2 "; 
           pivoting = false;
           pos_pivot = selected_row;
           pivot = z.read(pos_pivot);
           }
           } else {   // not pivoting is with respect to the perm(k)-th element if diagonal element has already been eliminated
           if (std::abs(val_larg_el*perm_tol)>std::abs(z.read(perm[k])) && pos_pivot>=0){ 
           std::cout<<"*3 "; 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           std::cout<<"*4 "; 
           pivoting = false;
           pos_pivot = perm[k];
           pivot = z.read(pos_pivot);
           }
           }
           }
           */
        if(eliminate){  // select potential pivot
            val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
            if(non_pivot[selected_row]){  // not pivoting is with respect to the diagonal element of the selected row (if possible)
                if (std::abs(val_larg_el*perm_tol)>std::abs(z.read(selected_row)) && pos_pivot>=0 && IP.get_perm_tol() <= 500.0){
                    //pivoting = true;
                    pivot = val_larg_el;
                } else {
                    //pivoting = false;
                    pos_pivot = selected_row;
                    pivot = z.read(pos_pivot);
                }
            } else {   // pivot if possible... only if nothing else works, use corresponding column
                if ( (std::abs(val_larg_el)>0.0) && pos_pivot>=0){ 
                    //pivoting = true;
                    pivot = val_larg_el;
                } else {
                    //pivoting = false;
                    pos_pivot = perm[k];
                    pivot = z.read(pos_pivot);
                }
            }
        }

        /*

           if(eliminate){  // select potential pivot
           val_larg_el=z.abs_max(pos_pivot); // finds largest element by absolute value. Returns value and position in z.
           if(non_pivot[selected_row]){  // not pivoting is with respect to the diagonal element of the selected row (if possible)
           if (std::abs(val_larg_el*perm_tol)>std::abs(z.read(selected_row)) && pos_pivot>=0 && IP.get_perm_tol() <= 500.0){
           pivoting = true;
           pivot = val_larg_el;
           } else {
           pivoting = false;
           pos_pivot = selected_row;
           pivot = z.read(pos_pivot);
           }
           } else {   // not pivoting is with respect to the perm(k)-th element if diagonal element has already been eliminated
           if (std::abs(val_larg_el*perm_tol)>std::abs(z.read(perm[k])) && pos_pivot>=0 && IP.get_perm_tol() <= 500.0){ 
           pivoting = true;
           pivot = val_larg_el;
           } else {
           pivoting = false;
           pos_pivot = perm[k];
           pivot = z.read(pos_pivot);
           }
           }
           }

*/

        if(eliminate && !force_finish && !IP.get_EXTERNAL_FINAL_ROW() && k > IP.get_MIN_ELIM_FACTOR()*n && IP.get_SMALL_PIVOT_TERMINATES() && std::abs(pivot) < IP.get_MIN_PIVOT()){  // terminate level because pivot is too small.
            eliminate = false;
            end_level_now = true;
            threshold *= threshold_Schur_factor;
            last_row_to_eliminate = k-1;  // the current row will already be the first row of Anew
            n_Anew = n-k;
            reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Acol.non_zeroes()));
            Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
            if(use_improved_SCHUR){ 
                for(Integer p = 0; p < droppedL_data.dim(); p++) droppedL_data_memory += droppedL_data.read(p).size();
                for(Integer p = 0; p < droppedL_colindex.dim() ; p++) droppedL_colindex_memory += droppedL_colindex.read(p).size();
                droppedL_data_memory *= sizeof(T);
                droppedL_colindex_memory *= sizeof(Integer);
            }
        }
        if(eliminate){  // select pivot scale z/U
            Dinv[k]=1.0/pivot;
            /*
               if (pivoting) {
               Dinv[k]=1.0/val_larg_el;
               } else {
               Dinv[k]=1.0/z.read(perm[k]); pos_pivot=perm[k];
               }
               */
            z.scale(Dinv[k]);
            z[pos_pivot]=0.0; // eliminate pivot for sorting
            // update permutations
            p=inverse_perm[pos_pivot];
            inverse_perm.switch_index(perm[k],pos_pivot);
            perm.switch_index(k,p);
            non_pivot[pos_pivot]=false;
        }
        if(use_weightsLU){
            for(j=0;j<z.non_zeroes();j++) weightsU[z.get_pointer(j)] += std::abs(z.get_data(j));
        }
        // (8.) read w
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
#endif
        w.zero_reset();
#ifdef VERBOSE
        time_7 = clock();
        time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
#endif
        // read column of A
        if(eliminate){
            for(i=Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
                if(unused_rows[Acol.indices[i]])
                    w[Acol.indices[i]] = Acol.data[i];
            }     // end for i
#ifdef VERBOSE
            time_8=clock();
            time_read += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
#endif
            // (9.) begin while
            h=startU[perm[k]];
            while(h!=-1){
                current_row_col_U=rowU[h];
                current_data_col_U=U.data[h];
                h=linkU[h];
                // (10.) w = w - U(i,perm(k))*l_i
                for(j=pointer[current_row_col_U];j<pointer[current_row_col_U+1];j++){
                    //if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U*data[j];
                    if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U/Dinv[current_row_col_U]*data[j];
                } // end for
            }   // (11.) end while
#ifdef VERBOSE
            time_7 = time_9=clock();
            time_calc_L += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
#endif
        } // end if 
        w.scale(Dinv[k]);
        if(use_weightsLU){
            for(j=0;j<w.non_zeroes();j++){ 
                weightsL[w.get_pointer(j)] += std::abs(w.get_data(j));
            }
        }
#ifdef VERBOSE
        time_8 = clock();
        time_scu_L += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
#endif
        pk = perm[k];
        if(IP.get_USE_INVERSE_DROPPING() && eliminate){
            if(k==0){
                xU[pk]=1.0; yU[pk]=1.0;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)]=vxU[z.get_pointer(j)]=z.get_data(j);
            } else {
                // initialise
                xplus  =  1.0 - vxU[pk];
                xminus = -1.0 - vxU[pk];
                nplus  = 0;
                nminus = 0;
                yplus  =  1.0 - vyU[pk];
                yminus = -1.0 - vyU[pk];
                nuplus  = 0.0;
                numinus = 0.0;
                // do x_k
                for(j=0;j<z.non_zeroes();j++) nuplus  += std::abs(vxU[z.get_pointer(j)]+z.get_data(j)*xplus);
                for(j=0;j<z.non_zeroes();j++) numinus += std::abs(vxU[z.get_pointer(j)]+z.get_data(j)*xminus);
                if(nuplus > numinus) xU[pk] = xplus;
                else xU[pk] = xminus;
                for(j=0;j<z.non_zeroes();j++) vxU[z.get_pointer(j)] +=  z.get_data(j)*xU[pk];
                xU[pk]=max(std::abs(xplus),std::abs(xminus));
                // do y_k
                for(j=0;j<z.non_zeroes();j++){
                    vi=vyU[z.get_pointer(j)];
                    if(std::abs(vi+z.get_data(j)*yplus) > max(2.0*std::abs(vi),(Real)0.5)) nplus++;
                    if(max(2.0*std::abs(vi+z.get_data(j)*yplus),(Real) 0.5)<std::abs(vi)) nplus--;
                    if(std::abs(vi+z.get_data(j)*yminus) > max(2.0*std::abs(vi),(Real) 0.5)) nminus++;
                    if(max(2.0*std::abs(vi+z.get_data(j)*yminus),(Real) 0.5)<std::abs(vi)) nminus--;
                }
                if(nplus > nminus) yU[pk]=yplus;
                else yU[pk]= yminus;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)] += z.get_data(j)*yU[pk];
                yU[pk]=max(std::abs(yplus),std::abs(yminus));
            }
        }   // values for dropping are now in xU[pk],yU[pk]
#ifdef STATISTICS
        L_total[k]= w.non_zeroes(); 
        U_total[k]= z.non_zeroes();
#endif
        if(!eliminate){  // drop in Schur complement
            z.take_largest_elements_by_abs_value_with_threshold(norm_U,list_U,max_fill_in,threshold,0,n);
        } else { // drop in U
            weightU=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()){norm = z.norm2(); if(norm==0.0) norm=1e-16; weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP()/norm);} 
            if(IP.get_USE_STANDARD_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP2());  // drop if |w_i|<tau
            if(IP.get_USE_INVERSE_DROPPING())  weightU = IP.combine(weightU,IP.get_WEIGHT_INVERSE_DROP()*max(std::abs(xU[pk]),std::abs(yU[pk])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_WEIGHTED_DROP()*weightsU[pk]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP()*w.norm1());
            //if(USE_ERR_PROP_DROPPING) weightU = combine(weightU,WEIGHT_ERR_PROP_DROP*(w.norm_max()));
            //if(USE_ERR_PROP_DROPPING) weightU = combine(weightU,WEIGHT_ERR_PROP_DROP*(1.0+w.norm_max()));
            if(IP.get_USE_ERR_PROP_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP2()*w.norm1()/std::abs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_PIVOT_DROP()*std::abs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG()) weightU *= std::abs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG()){max_inv_piv = max(max_inv_piv,std::abs(Dinv[k])); weightU *= max_inv_piv;}
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,0,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightsU,weightU,max_fill_in-1,threshold,0,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUCDP: DROP_TYPE_U does not have permissible value."<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            perm.resize(0);
                            permrows.resize(0);
                            inverse_perm.resize(0);
                            inverse_permrows.resize(0);
                            return false;
                }
            } else {
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,0,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,weightsU,weightU,max_fill_in-1,threshold,0,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in,threshold-1,0,n,k,bandwidth_U,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUCDP: DROP_TYPE_U does not have permissible value."<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            perm.resize(0);
                            permrows.resize(0);
                            inverse_perm.resize(0);
                            inverse_permrows.resize(0);
                            return false;
                }
            }
        }

#ifdef VERBOSE
        time_9 = clock();
        time_dropping += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef STATISTICS
        U_kept[k]= list_U.dimension();
        //if (U_total[k] != U_kept[k]){ std::cout<<"k = "<<k<<" U_kept"<<U_kept[k]<<" U_total "<<U_total[k]<<std::endl<<"Vector"<<std::endl; z.print_non_zeroes();}
#endif
        // update U or Anew
        if(eliminate){
            if(U.pointer[k]+list_U.dimension()+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+list_U.dimension()+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                linkU.enlarge_dim_keep_data(reserved_memory_U);
                rowU.enlarge_dim_keep_data(reserved_memory_U);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for U at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<U.pointer[k]+list_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            U.indices[U.pointer[k]]=pos_pivot;
            for(j=0;j<list_U.dimension();j++){
                pos=U.pointer[k]+j+1;
                U.data[pos]=z.read(list_U[list_U.dimension()-1-j]);
                U.indices[pos]=list_U[list_U.dimension()-1-j];
                if(use_norm_row_U) norm_row_U[k] += std::abs(U.data[pos]);
                h=startU[U.indices[pos]];
                startU[U.indices[pos]]=pos;
                linkU[pos]=h;
                rowU[pos]=k;
            }
            U.pointer[k+1]=U.pointer[k]+list_U.dimension()+1;
            if(pivot==0.0){
                zero_pivots++;
                Dinv[k]=1.0;
#ifdef VERBOSE
                std::cerr<<"matrix_sparse::partialILUCDP: Preconditioner does not exist (zero pivot). Setting diagonal to 1."<<std::endl;
#endif
            }
            if(use_improved_SCHUR){ // update droppedU
                if(droppedU.pointer[k]+rejected_U.dimension()>reserved_memory_droppedU){
                    reserved_memory_droppedU = 2*(droppedU.pointer[k]+rejected_U.dimension());
                    droppedU.enlarge_fields_keep_data(reserved_memory_droppedU);
                    // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for droppedU at position 1"<<std::endl;
                    // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<droppedU.pointer[k]+rejected_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                    // reformat(0,0,0,COLUMN);
                    // U.reformat(0,0,0,ROW);
                    // Dinv.resize_without_initialization(0);
                    // Anew.reformat(0,0,0,ROW);
                    // perm.resize(0);
                    // permrows.resize(0);
                    // inverse_perm.resize(0);
                    // inverse_permrows.resize(0);
                    // return false;
                }
                for(j=0;j<rejected_U.dimension();j++){
                    pos=droppedU.pointer[k]+j;
                    droppedU.data[pos]=z.read(rejected_U[j]);
                    droppedU.indices[pos]=rejected_U[j];
                }
                droppedU.pointer[k+1]=droppedU.pointer[k]+rejected_U.dimension();
            }  // end updating droppedU
        } else {
            k_Anew = k -last_row_to_eliminate-1;
            if(U.pointer[k]+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                linkU.enlarge_dim_keep_data(reserved_memory_U);
                rowU.enlarge_dim_keep_data(reserved_memory_U);                 
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For U:    Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<U.pointer[k]+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            if(Anew.pointer[k_Anew]+list_U.dimension()>reserved_memory_Anew){
                reserved_memory_Anew = 2*(Anew.pointer[k_Anew]+list_U.dimension());
                Anew.enlarge_fields_keep_data(reserved_memory_Anew);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For Anew: Reserved memory for non-zero elements: "<<reserved_memory_Anew<<" Memory needed: "<<Anew.pointer[k_Anew]+list_U.dimension()<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            Dinv[k]=1.0;
            U.indices[U.pointer[k]]=perm[k];
            U.pointer[k+1]=U.pointer[k]+1;
            for(j=0;j<list_U.dimension();j++){
                pos=Anew.pointer[k_Anew]+j;
                Anew.data[pos]=z.read(list_U[list_U.dimension()-1-j]);
                Anew.indices[pos]=list_U[list_U.dimension()-1-j];
            }
            Anew.pointer[k_Anew+1]=Anew.pointer[k_Anew]+list_U.dimension();
        }
        // store positions of columns of U, but without pivot
        // update non-pivots.
#ifdef VERBOSE
        time_0=clock();
        if(eliminate) time_scu_U += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
        else time_scu_Anew += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
#endif
        // (12.) sort and copy data to L
        // sort
        pk = permrows[k];
        if(eliminate){

            if(IP.get_USE_INVERSE_DROPPING()){
                if(k==0){
                    xL[pk]=1.0; yL[pk]=1.0;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)]=vxL[w.get_pointer(j)]=w.get_data(j);
                } else {
                    // initialise
                    xplus  =  1.0 - vxL[pk];
                    xminus = -1.0 - vxL[pk];
                    nplus  = 0;
                    nminus = 0;
                    yplus  =  1.0 - vyL[pk];
                    yminus = -1.0 - vyL[pk];
                    nuplus  = 0.0;
                    numinus = 0.0;
                    // do x_k
                    for(j=0;j<w.non_zeroes();j++) nuplus  += std::abs(vxL[w.get_pointer(j)]+w.get_data(j)*xplus);
                    for(j=0;j<w.non_zeroes();j++) numinus += std::abs(vxL[w.get_pointer(j)]+w.get_data(j)*xminus);
                    if(nuplus > numinus) xL[pk] = xplus;
                    else xL[pk] = xminus;
                    for(j=0;j<w.non_zeroes();j++) vxL[w.get_pointer(j)] +=  w.get_data(j)*xL[pk];
                    xL[pk]=max(std::abs(xplus),std::abs(xminus));
                    // do y_k
                    for(j=0;j<w.non_zeroes();j++){
                        vi=vyL[w.get_pointer(j)];
                        if(std::abs(vi+w.get_data(j)*yplus) > max(2.0*std::abs(vi),(Real) 0.5)) nplus++;
                        if(max(2.0*std::abs(vi+w.get_data(j)*yplus),(Real) 0.5)<std::abs(vi)) nplus--;
                        if(std::abs(vi+w.get_data(j)*yminus) > max(2.0*std::abs(vi),(Real) 0.5)) nminus++;
                        if(max(2.0*std::abs(vi+w.get_data(j)*yminus),(Real) 0.5)<std::abs(vi)) nminus--;
                    }
                    if(nplus > nminus) yL[pk]=yplus;
                    else yL[pk]= yminus;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)] += w.get_data(j)*yL[pk];
                    yL[pk]=max(std::abs(yplus),std::abs(yminus));
                }  // values for dropping are now in xL[pk],yL[pk]
            }
            weightL=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()) {norm = w.norm2(); if(norm==0.0) norm=1e-16; weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP()/norm);}
            if(IP.get_USE_STANDARD_DROPPING2()) weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP2());
            if(IP.get_USE_INVERSE_DROPPING())  weightL = IP.combine(weightL,IP.get_WEIGHT_INVERSE_DROP()*max(std::abs(xL[pk]),std::abs(yL[pk])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_WEIGHTED_DROP()*weightsL[pk]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP()*z.norm1());
            if(IP.get_USE_ERR_PROP_DROPPING2())weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP2()*z.norm1()/std::abs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING())weightL = IP.combine(weightL,IP.get_WEIGHT_PIVOT_DROP()*std::abs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG())  weightL *= std::abs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG())  weightL *= max_inv_piv;
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in,threshold,0,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightsL,weightL,max_fill_in-1,threshold,0,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUCDP: DROP_TYPE_L does not have permissible value."<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            perm.resize(0);
                            permrows.resize(0);
                            inverse_perm.resize(0);
                            inverse_permrows.resize(0);
                            return false;
                }
            } else {
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in,threshold,0,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,0,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,weightsL,weightL,max_fill_in-1,threshold,0,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,0,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUCDP: DROP_TYPE_L does not have permissible value."<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            perm.resize(0);
                            permrows.resize(0);
                            inverse_perm.resize(0);
                            inverse_permrows.resize(0);
                            return false;
                }
            }
#ifdef VERBOSE
            time_1 = clock();
            time_dropping += (Real)(time_1-time_0)/(Real)CLOCKS_PER_SEC;
            time_0 = time_1;
#endif
#ifdef STATISTICS
            L_kept[k]= list_L.dimension();
            //if (L_total[k] != L_kept[k]){ std::cout<<"k = "<<k<<" L_kept"<<L_kept[k]<<" L_total "<<L_total[k]<<std::endl<<"Vector"<<std::endl; w.print_non_zeroes();}
#endif
            if(pointer[k]+list_L.dimension()+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+list_L.dimension()+1);
                enlarge_fields_keep_data(reserved_memory_L);
                linkL.enlarge_dim_keep_data(reserved_memory_L);
                colL.enlarge_dim_keep_data(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for L at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=selected_row;
            for(j=0;j<list_L.dimension();j++){
                pos = pointer[k]+j+1;
                //data[pos] = w.read(list_L[j])/U.data[U.pointer[k]];
                //data[pos] = w.read(list_L[j])*Dinv[k];
                data[pos] = w.read(list_L[j]); // scaling has already been performed previously
                b = indices[pos] = list_L[j];
                h=startL[b];
                startL[b]=pos;
                linkL[pos]=h;
                colL[pos]=k;
                // begin updating fields for number elements of row of L
                if (b >= bpr && b <=  epr) {
                    if(IP.get_FINAL_ROW_CRIT() >= -1 && IP.get_FINAL_ROW_CRIT() < 11){  // resorting by the number of elements in row of L. Eliminating in increasing order.
                        b = inverse_permrows[b];
                        a = --pointer_num_el_row_L[++numb_el_row_L[b]];
                        inverse_permrows.switch_index(permrows[a],permrows[b]);
                        permrows.switch_index(a,b);
                        numb_el_row_L.switch_entry(a,b);
                    } else {   // resorting by 1-norm of number of elements in row of L. Eliminating in increasing order.
                        switch(IP.get_FINAL_ROW_CRIT()){
                            case -2: row_reorder_weight.add(b,std::abs(data[pos])); break;
                            case -3: row_reorder_weight.add(b,std::abs(data[pos])*norm_row_U[b]); break;
                            case -4: row_reorder_weight.add(b,std::abs(data[pos])/std::abs(Dinv[b])*norm_row_U[b]); break;
                            default: std::cerr<<"matrix_sparse::partialILUCDP: FINAL_ROW_CRIT has undefined value. Please set to correct value."<<std::endl;
                                     reformat(0,0,0,COLUMN);
                                     U.reformat(0,0,0,ROW);
                                     Dinv.resize_without_initialization(0);
                                     Anew.reformat(0,0,0,ROW);
                                     perm.resize(0);
                                     permrows.resize(0);
                                     inverse_perm.resize(0);
                                     inverse_permrows.resize(0);
                                     return false;
                        }
                    }
                }
                // end updating fields
            } // end for j
            // sort permrows if necessary, i.e. if num_el_row_L increases at next iteration.
            if(IP.get_FINAL_ROW_CRIT() >= -1 && IP.get_FINAL_ROW_CRIT() < 11 && pointer_num_el_row_L[numb_el_row_L[k]+1] == k+1) 
                permrows.quicksort_with_inverse(inverse_permrows,pointer_num_el_row_L[numb_el_row_L[k]+1],pointer_num_el_row_L[numb_el_row_L[k]+2]-1);
            // end sorting
            if(IP.get_FINAL_ROW_CRIT() < -1 && k<n-1){  // still need to update permutations and inverse permutations for rows in this case
                b = row_reorder_weight.index_min();
                if(IP.get_USE_MAX_AS_MOVE()) move_level_parameter=row_reorder_weight.read_max(); else move_level_parameter=row_reorder_weight.read_min();
                row_reorder_weight.remove_min();
                p=inverse_permrows[b];
                inverse_permrows.switch_index(permrows[k+1],b); // k+1 the next loop
                permrows.switch_index(k+1,p);
            }
            pointer[k+1]=pointer[k]+list_L.dimension()+1;
            if(use_improved_SCHUR){ // update droppedU
                for(j=0;j<rejected_L.dimension();j++){
                    pos = rejected_L[j]; // row index of current element
                    droppedL_colindex[pos].push(k);  // store corresponding column index = k
                    droppedL_data[pos].push(w.read(pos));  // store corresponding data element.

                }
            }  // end updating droppedU
        } else {  //  else branch of if(eliminate)
            if(pointer[k]+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+1);
                enlarge_fields_keep_data(reserved_memory_L);
                linkL.enlarge_dim_keep_data(reserved_memory_L);
                colL.enlarge_dim_keep_data(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUCDP: memory reserved was insufficient. Overflow for L at position 2"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=selected_row;
            pointer[k+1]=pointer[k]+1;
        }  //  end:  if(eliminate)
#ifdef VERBOSE
        time_1=clock();
        time_scu_L += (Real)(time_1-time_0)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate && IP.get_FINAL_ROW_CRIT() < 11 && !force_finish){
            if(IP.get_EXTERNAL_FINAL_ROW()){
                if (k >= last_row_to_eliminate && k >= IP.get_EXT_MIN_ELIM_FACTOR()*n){
                    end_level_now = true;
                }  // end if (last_row_to_eliminate == k)
            } else {
                if(k > IP.get_MIN_ELIM_FACTOR()*n)
                    switch(IP.get_FINAL_ROW_CRIT()){
                        case -3:  if(move_level_parameter > IP.get_MOVE_LEVEL_THRESHOLD()) end_level_now = true; break;
                        case -2:  if(move_level_parameter > IP.get_MOVE_LEVEL_THRESHOLD()) end_level_now = true; break;
                        case -1:  if( numb_el_row_L[k] > ((Real) IP.get_MOVE_LEVEL_FACTOR()*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  0:  if( numb_el_row_L[k] > ((Real) 0.5*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  1:  if( numb_el_row_L[k] > ((Real)     Acol.non_zeroes())/n) end_level_now = true; break;
                        case  2:  if( numb_el_row_L[k] > ((Real) 2.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  3:  if( numb_el_row_L[k] > ((Real) 4.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  4:  if( numb_el_row_L[k] > ((Real) 6.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  5:  if( numb_el_row_L[k] > 10) end_level_now = true; break;
                        case  6:  if( numb_el_row_L[k] > ((Real) 1.5*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  7:  if( z.norm2() > IP.get_ROW_U_MAX()) end_level_now = true; break;
                        case  8:  if( numb_el_row_L[k] > ((Real) 3.0*Acol.non_zeroes())/n) end_level_now = true; break;
                        case  9:  if( numb_el_row_L[k] > ((Real) 1.2*Acol.non_zeroes())/n) end_level_now = true; break;
                                      //case 10:  end_level_now = true; break;
                        default:
                                      std::cerr<<"Please set FINAL_ROW_CRIT to a permissible value. Returning empty preconditioner."<<std::endl;
                                      reformat(0,0,0,COLUMN);
                                      U.reformat(0,0,0,ROW);
                                      Dinv.resize_without_initialization(0);
                                      Anew.reformat(0,0,0,ROW);
                                      perm.resize(0);
                                      permrows.resize(0);
                                      inverse_perm.resize(0);
                                      inverse_permrows.resize(0);
                                      return false;
                                      break;
                    }  // end switch
            }   // end if(EXTERNAL_ROW)
            if(end_level_now){
                eliminate = false;
                threshold *= threshold_Schur_factor;
                last_row_to_eliminate = k;
                n_Anew = n-k-1;
                //reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(mem_factor*Acol.non_zeroes()));
                reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Acol.non_zeroes()));
                Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
                Anew.pointer[0]=0;
                if(use_improved_SCHUR){ 
                    for(Integer p = 0; p < droppedL_data.dim(); p++) droppedL_data_memory += droppedL_data.read(p).size();
                    for(Integer p = 0; p < droppedL_colindex.dim() ; p++) droppedL_colindex_memory += droppedL_colindex.read(p).size();
                    droppedL_data_memory *= sizeof(T);
                    droppedL_colindex_memory *= sizeof(Integer);
                }
            } // end if(end_level_now)
        } // end if (eliminate)
        if (eliminate && IP.get_REQUIRE_ZERO_SCHUR() && IP.get_REQ_ZERO_SCHUR_SIZE()>= n-k-1){
            //std::cout<<"setting schur complement: matrix has dimension "<<n<<" IP.get_REQ_ZERO_SCHUR_SIZE() = "<<IP.get_REQ_ZERO_SCHUR_SIZE()<<" k = "<<k<<std::endl; 
            eliminate = false;
            if(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                enlarge_fields_keep_data(reserved_memory_L);
                linkL.enlarge_dim_keep_data(reserved_memory_L);
                colL.enlarge_dim_keep_data(reserved_memory_L);
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            } 
            if(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                U.enlarge_fields_keep_data(reserved_memory_U);
                linkU.enlarge_dim_keep_data(reserved_memory_U);
                rowU.enlarge_dim_keep_data(reserved_memory_U);                 
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // perm.resize(0);
                // permrows.resize(0);
                // inverse_perm.resize(0);
                // inverse_permrows.resize(0);
                // return false;
            } 
            n_Anew = IP.get_REQ_ZERO_SCHUR_SIZE();
            Anew.reformat(n_Anew,n_Anew,0,ROW);
            //std::cout<<"reformated Anew to dimension = "<<Anew.rows()<<std::endl; 
            for(j=k+1;j<n;j++){
                //std::cout<<"doing row/column of L/U j=  "<<j<<std::endl; 
                data[pointer[j]]=1.0;
                indices[pointer[j]]=permrows[j];
                pointer[j+1]=pointer[j]+1;
                U.data[U.pointer[j]]=1.0;
                Dinv[j]=1.0;
                U.indices[U.pointer[j]]=perm[j];
                U.pointer[j+1]=U.pointer[j]+1;
            }
            break;
        }  // end if
    }  // (13.) end for k
#ifdef VERBOSE
    time_2 = clock();
#endif
    Real memory_L_allocated = memory();
    Real memory_U_allocated = U.memory();
    Real memory_Anew_allocated = Anew.memory();
    compress();
    U.compress();
    if(eliminate) Anew.reformat(0,0,0,ROW); // if eliminated till end, then Anew is a 0x0 matrix.
    else {
        if(Anew.nnz>0){
            Anew.compress();
            // abuse linkU to store data
            linkU.destroy_resize_data_field(Anew.nnz);
            // resort and shift indices to standard
            for(j=0;j<Anew.nnz;j++) linkU[j]=Anew.indices[j];
            for (i=0; i<Anew.rows(); i++)
                for(j=Anew.pointer[i]; j<Anew.pointer[i+1]; j++)
                    Anew.indices[j] = inverse_perm[linkU[j]]-last_row_to_eliminate-1;
            Anew.normal_order();
            Anew.number_columns=n_Anew; // originally, Anew has n columns
        } else {
            Anew.reformat(n_Anew,n_Anew,0,ROW);
        }
    }
    permute(permrows,ROW);
    U.permute(perm,COLUMN);
    if(IP.get_USE_POS_COMPRESS()){
        positional_compress(IP,post_fact_threshold);
        U.positional_compress(IP,post_fact_threshold);
    }
#ifdef VERBOSE
    time_3=clock();
    time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef VERBOSE
    time_4=clock();
    time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
    std::cout<<"    partialILUCDP-Times: "<<std::endl;
    std::cout<<"        initialization:                              "<<time_init<<std::endl;
    std::cout<<"        reading matrix:                              "<<time_read<<std::endl;
    std::cout<<"        sparse zero set:                             "<<time_zeroset<<std::endl;
    std::cout<<"        calculating L:                               "<<time_calc_L<<std::endl;
    std::cout<<"        calculating U:                               "<<time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info L:    "<<time_scu_L<<std::endl;
    std::cout<<"        sorting, copying, updating access info U:    "<<time_scu_U<<std::endl;
    std::cout<<"        calculating Anew:                            "<<time_calc_Anew<<std::endl;
    std::cout<<"        sorting, copying, updating access info Anew: "<<time_scu_Anew<<std::endl;
    std::cout<<"        dropping:                                    "<<time_dropping<<std::endl;
    std::cout<<"        compressing:                                 "<<time_compress<<std::endl;
    std::cout<<"        resorting:                                   "<<time_resort<<std::endl;
    std::cout<<"      Total times:"<<std::endl;
    std::cout<<"        calculations:                                "<<time_calc_L+time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info:      "<<time_scu_L+time_scu_U<<std::endl;
    std::cout<<"        other administration:                        "<<time_init+time_read+time_zeroset+time_compress+time_resort+time_dropping<<std::endl;
    std::cout<<"      Grand total                                    "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
#endif
    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    Real allocated_mem_lists_L = ((Real) (reserved_memory_L+1)) * sizeof(Integer);  // for linkL, colL
    Real allocated_mem_lists_U = ((Real) (reserved_memory_U+1)) * sizeof(Integer);  // for linkU, rowU
    Real used_mem_lists_L = ((Real) (nnz+1)) * sizeof(Integer);  // for linkL, colL
    Real used_mem_lists_U = ((Real) (U.nnz+1)) * sizeof(Integer);  // for linkU, rowU
    total_memory_allocated = w.memory() + z.memory() + non_pivot.memory() + unused_rows.memory() + numb_el_row_L.memory()+
        pointer_num_el_row_L.memory() + norm_row_U.memory() + row_reorder_weight.memory() + list_L.memory()+
        list_U.memory() + rejected_L.memory() + rejected_U.memory() + droppedU.memory() + vxL.memory() +
        vyL.memory() + vxU.memory() + vyU.memory() + xL.memory() + yL.memory() + xU.memory() + yU.memory() +
        startU.memory() +  startL.memory() + Dinv.memory()+ perm.memory() + permrows.memory() + inverse_perm.memory()
        + inverse_permrows.memory() + droppedL_data_memory + droppedL_colindex_memory;
    total_memory_used = total_memory_allocated;
    total_memory_allocated += 2.0*allocated_mem_lists_U + 2.0* allocated_mem_lists_L + memory_U_allocated + memory_L_allocated + memory_Anew_allocated;
    total_memory_used += 2.0*used_mem_lists_U + 2.0* used_mem_lists_L + U.memory() + memory() + Anew.memory();
    //       Real mem_mat = Arow.memory();
    //       std::cout<<std::endl;
    //       std::cout<<"Relative Memory for a matrix of dimension "<<n<<":"<<std::endl;
    //       std::cout<<"w                    "<< w.memory()/mem_mat<<std::endl;
    //       std::cout<<"z                    "<< z.memory()/mem_mat<<std::endl;
    //       std::cout<<"non_pivot            "<< non_pivot.memory()/mem_mat<<std::endl;
    //       std::cout<<"unused_rows          "<< unused_rows.memory()/mem_mat<<std::endl;
    //       std::cout<<"numb_el_row_L        "<< numb_el_row_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"pointer_num_el_row_L "<< pointer_num_el_row_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"norm_row_U           "<< norm_row_U.memory()/mem_mat<<std::endl;
    //       std::cout<<"row_reorder_weight   "<< row_reorder_weight.memory()/mem_mat<<std::endl;
    //       std::cout<<"list                 "<< list_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"list_U               "<< list_U.memory()/mem_mat<<std::endl;
    //       std::cout<<"rejected_L           "<< rejected_L.memory()/mem_mat<<std::endl;
    //       std::cout<<"rejected_U           "<< rejected_U.memory()/mem_mat<<std::endl;
    //       std::cout<<"droppedU             "<< droppedU.memory()/mem_mat<<std::endl;
    //       std::cout<<"vxL                  "<< vxL.memory()/mem_mat<<std::endl;
    //       std::cout<<"vyL                  "<< vyL.memory()/mem_mat<<std::endl;
    //       std::cout<<"vxU                  "<< vxU.memory()/mem_mat<<std::endl;
    //       std::cout<<"vyU                  "<< vyU.memory()/mem_mat<<std::endl;
    //       std::cout<<"xL                   "<< xL.memory()/mem_mat<<std::endl;
    //       std::cout<<"yL                   "<< yL.memory()/mem_mat<<std::endl;
    //       std::cout<<"xU                   "<< xU.memory()/mem_mat<<std::endl;
    //       std::cout<<"yU                   "<< yU.memory()/mem_mat<<std::endl;
    //       std::cout<<"linkU                "<< allocated_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"rowU                 "<< allocated_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"startU               "<< startU.memory()/mem_mat<<std::endl;
    //       std::cout<<"linkL                "<< allocated_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"colL                 "<< allocated_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"startL               "<< startL.memory()/mem_mat<<std::endl;
    //       std::cout<<"Dinv                 "<< Dinv.memory()/mem_mat<<std::endl;
    //       std::cout<<"perm                 "<< perm.memory()/mem_mat<<std::endl;
    //       std::cout<<"permrows             "<< permrows.memory()/mem_mat<<std::endl;
    //       std::cout<<"inverse_perm         "<< inverse_perm.memory()/mem_mat<<std::endl;
    //       std::cout<<"inverse_permrows     "<< inverse_permrows.memory()/mem_mat<<std::endl;
    //       std::cout<<"U                    "<< memory_U_allocated/mem_mat<<std::endl;
    //       std::cout<<"L                    "<< memory_L_allocated/mem_mat<<std::endl;
    //       std::cout<<"Anew                 "<< memory_Anew_allocated/mem_mat<<std::endl;
    //       std::cout<<"droppedL_data        "<< droppedL_data_memory/mem_mat<<std::endl;
    //       std::cout<<"droppedL_colindex    "<< droppedL_colindex_memory/mem_mat<<std::endl;
    //       std::cout<<"linkU (used)         "<< used_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"rowU  (used)         "<< used_mem_lists_U/mem_mat<<std::endl;
    //       std::cout<<"linkL (used)         "<< used_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"colL  (used)         "<< used_mem_lists_L/mem_mat<<std::endl;
    //       std::cout<<"U     (used)         "<< U.memory()/mem_mat<<std::endl;
    //       std::cout<<"L     (used)         "<< memory()/mem_mat<<std::endl;
    //       std::cout<<"Anew  (used)         "<< Anew.memory()/mem_mat<<std::endl;
    //       std::cout<<"total_memory         "<< total_memory_allocated/mem_mat<<std::endl;
    //       std::cout<<"total_memory (used)  "<< total_memory_used/mem_mat<<std::endl;
    //       std::cout<<std::endl;
#ifdef STATISTICS
    // Statistics for A
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Arow.row_density();
    for(k=0;k<n;k++){
        help =  Arow.pointer[k+1]-Arow.pointer[k];
        if (max_total < help) max_total = help;
        if (min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for A"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number: "<<average_total<<std::endl;
    std::cout<<"       Minimum Number     in Row:    "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Row:    "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Row:    "<<stand_dev_total<<std::endl;
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Acol.column_density();
    for(k=0;k<n;k++){
        help =  Acol.pointer[k+1]-Acol.pointer[k];
        if(max_total < help) max_total = help;
        if(min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<"       Minimum Number     in Column: "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Column: "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Column: "<<stand_dev_total<<std::endl;
    // Statistics for L
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) L_kept.read(k))/((Real) L_total.read(k));
        if(max_total < L_total.read(k)) max_total =L_total.read(k);
        if(min_total > L_total.read(k)) min_total =L_total.read(k);
        if(max_kept < L_kept.read(k)) max_kept =L_kept.read(k);
        if(min_kept > L_kept.read(k)) min_kept =L_kept.read(k);
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += L_total.read(k);
        sum2 += L_kept.read(k);
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) L_kept.read(k))/((Real) L_total.read(k));
        sum1 += (L_total.read(k)-average_total)*(L_total.read(k)-average_total);
        sum2 += (L_kept.read(k)-average_total)*(L_kept.read(k)-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for L"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;
    // Statistics for U
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) U_kept.read(k))/((Real) U_total.read(k));
        if(max_total < U_total.read(k)) max_total =U_total.read(k);
        if(min_total > U_total.read(k)) min_total =U_total.read(k);
        if(max_kept < U_kept.read(k)) max_kept =U_kept.read(k);
        if(min_kept > U_kept.read(k)) min_kept =U_kept.read(k);
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += U_total.read(k);
        sum2 += U_kept.read(k);
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) U_kept.read(k))/((Real) U_total.read(k));
        sum1 += (U_total.read(k)-average_total)*(U_total.read(k)-average_total);
        sum2 += (U_kept.read(k)-average_total)*(U_kept.read(k)-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for U"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;

#endif
    //std::cout<<"L"<<std::endl<<expand()<<std::endl;
    //std::cout<<"U"<<std::endl<<U.expand()<<std::endl;
    //std::cout<<"Anew"<<std::endl<<Anew.expand()<<std::endl;
    //std::cout<<"Dinv"<<std::endl<<Dinv<<std::endl;
    return true;
}

template<class T> bool matrix_sparse<T>::partialILUC(const matrix_sparse<T>& Arow, matrix_sparse<T>& Anew, const iluplusplus_precond_parameter& IP, bool force_finish, matrix_sparse<T>& U, vector_dense<T>& Dinv, Integer last_row_to_eliminate, Real threshold, Integer& zero_pivots, Real& time_self, Real mem_factor, Real& total_memory_allocated, Real& total_memory_used){
    total_memory_allocated = 0.0;
    time_self = 0.0;
    Integer n = Arow.columns();
    if(!Arow.square_check()){
        std::cerr<<"matrix_sparse::partialILUC: argument matrix must be square. Returning 0x0 matrices."<<std::endl<<std::flush;
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        return false;
    }
    if(n==0){
        reformat(0,0,0,COLUMN);
        U.reformat(0,0,0,ROW);
        Dinv.resize_without_initialization(0);
        Anew.reformat(0,0,0,ROW);
        return true;
    }
    clock_t time_begin, time_end;
    time_begin=clock();
    Integer bandwidth, bandwidth_L, bandwidth_U;
    Integer i,j,k;//help;
    Integer h,pos;
    Integer max_fill_in;
    if(IP.get_MAX_FILLIN_IS_INF())  max_fill_in = n;
    else max_fill_in = IP.get_fill_in();
    if(max_fill_in<1) max_fill_in = 1;
    if(max_fill_in>n) max_fill_in = n;
    T pivot = 0.0;  // dummy initialization
    zero_pivots=0;
    Real norm_U,norm; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
    Real max_inv_piv=0.0;
    if(IP.get_DROP_TYPE_L()==4||IP.get_DROP_TYPE_U()==4) bandwidth=Arow.bandwidth(); else bandwidth=0;
    switch (IP.get_DROP_TYPE_L()){
        case 3: bandwidth_L = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_L = bandwidth; break;
        default: bandwidth_L = 0;
    }
    switch (IP.get_DROP_TYPE_U()){
        case 3: bandwidth_U = (Integer) (n*IP.get_BANDWIDTH_MULTIPLIER())+IP.get_BANDWIDTH_OFFSET(); break;
        case 4: bandwidth_U = bandwidth; break;
        default: bandwidth_U = 0;
    }
    if (threshold > 500.0) threshold=0.0;
    else threshold=std::exp(-threshold*std::log(10.0));
    Real threshold_Schur_factor = std::exp(-IP.get_THRESHOLD_SHIFT_SCHUR()*std::log(10.0));
    Real post_fact_threshold;
    if  (IP.get_POST_FACT_THRESHOLD() > 500.0) post_fact_threshold = 0.0; 
    else post_fact_threshold = threshold*std::exp(-IP.get_POST_FACT_THRESHOLD()*std::log(10.0));
    if(last_row_to_eliminate+1>n) last_row_to_eliminate = n-1;
    if(last_row_to_eliminate<0) last_row_to_eliminate = 0;
    bool use_improved_SCHUR = (IP.get_SCHUR_COMPLEMENT()>0);
    bool use_weightsLU = IP.get_USE_WEIGHTED_DROPPING() || IP.get_USE_WEIGHTED_DROPPING2();
    bool end_level_now = false;  // indicates if next iteration in k-loop starts a new level, i.e. calculations of Schur complement begin.
    bool eliminate = true;       // indicates if standard elimination is being performed or Schur complement being calculated
    Integer k_Anew,n_Anew=0; // set later
    Integer reserved_memory_Anew=0; // will be set later
    T  xplus, xminus, yplus, yminus,vi;
    Real nuplus,numinus;
    Integer nplus, nminus;
    Real weightL, weightU;
#ifdef VERBOSE
    clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_9;
    Real time_init=0.0;
    Real time_read=0.0;
    Real time_calc_L=0.0;
    Real time_scu_L=0.0;  // sorting, copying, updating access information
    Real time_calc_U=0.0;
    Real time_scu_U=0.0;
    Real time_zeroset=0.0;
    Real time_compress=0.0;
    Real time_resort=0.0;
    time_0 = clock();
#endif
    sorted_vector row_reorder_weight;
    bool use_norm_row_U = false;
    vector_dense<Real> norm_row_U;
    Real droppedL_data_memory = 0.0;
    Real droppedL_colindex_memory = 0.0;
    vector_dense<T> vxL,vyL,vxU,vyU,xL,yL,xU,yU;
    vector_dense<Real> weightsL,weightsU;
    index_list list_L, list_U;
    index_list rejected_L, rejected_U;
    Integer reserved_memory_L = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Arow.non_zeroes()));
    Integer reserved_memory_U = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Arow.non_zeroes()));
    Integer reserved_memory_droppedU = max(n,(Integer) min(((Real)(max_fill_in))*((Real) n), mem_factor*Arow.non_zeroes()));
    array<Integer> firstU, firstUdropped, firstL, firstA, listA, headA, listU,listUdropped, listL;
    array< std::queue<T> > droppedL_data;
    array< std::queue<Integer> > droppedL_colindex;
    matrix_sparse<T> droppedU;
    vector_sparse_dynamic<T> w,z;
#ifdef STATISTICS
    vector_dense<Integer> L_total,L_kept,U_total,U_kept;
    Real average_total,average_kept,average_prop, min_prop, max_prop, stand_dev_kept, stand_dev_prop, stand_dev_total;
    Integer min_total, max_total, min_kept, max_kept, help;
    Real sum1, sum2, sum3, prop;
#endif
    firstU.erase_resize_data_field(n);
    firstL.erase_resize_data_field(n);
    firstA.erase_resize_data_field(n);
    listA.erase_resize_data_field(n);
    headA.erase_resize_data_field(n);
    listU.erase_resize_data_field(n);
    listL.erase_resize_data_field(n);
    Dinv.resize(n,1.0);
    w.resize(n);
    z.resize(n);
    U.reformat(n,n,reserved_memory_U,ROW);
    reformat(n,n,reserved_memory_L,COLUMN);
    if(use_improved_SCHUR){
        firstUdropped.erase_resize_data_field(n);
        listUdropped.erase_resize_data_field(n);
    }
    if(IP.get_FINAL_ROW_CRIT() <= -1){
        row_reorder_weight.resize(n); 
        if(n>0) row_reorder_weight.remove(0);
    }
    if(IP.get_FINAL_ROW_CRIT() == -3 || IP.get_FINAL_ROW_CRIT() == -4) {
        use_norm_row_U=true; 
        norm_row_U.resize(n,0.0);
    }
    if(use_improved_SCHUR){
        droppedU.reformat(n,n,reserved_memory_droppedU,ROW);
        droppedL_data.resize(n);
        droppedL_colindex.resize(n);
    }
    if(IP.get_USE_INVERSE_DROPPING()){
        xL.resize(n,0);
        yL.resize(n,0);
        vxL.resize(n,0);
        vyL.resize(n,0);
        xU.resize(n,0);
        yU.resize(n,0);
        vxU.resize(n,0);
        vyU.resize(n,0);
    }
    if(use_weightsLU){
        weightsL.resize(n,IP.get_INIT_WEIGHTS_LU());
        weightsU.resize(n,IP.get_INIT_WEIGHTS_LU());  // set equal to 1 for diagonal element
    }
#ifdef STATISTICS
    L_total.resize(n,0); L_kept.resize(n,0); U_total.resize(n,0); U_kept.resize(n,0);
#endif
    initialize_sparse_matrix_fields(n,Arow.pointer,Arow.indices,listA,headA,firstA);
    initialize_triangular_fields(n,listL);
    initialize_triangular_fields(n,listU);
    if(use_improved_SCHUR){
        initialize_triangular_fields(n,listUdropped);
    }
    // (1.) begin for k
#ifdef VERBOSE
    time_1 = clock();
    time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
#endif
    for(k=0;k<n;k++){
#ifdef VERBOSE
        time_2=clock();
#endif
        // (2.) initialize z
        z.zero_reset();
#ifdef VERBOSE
        time_3=clock();
        time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
        // read row of A
        for(j=firstA[k];j<Arow.pointer[k+1];j++) z[Arow.indices[j]] = Arow.data[j];
#ifdef VERBOSE
        time_4=clock();
        time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
#endif
        // (3.) begin while
        h=listL[k];
        while(h!=-1){
            for(j=firstU[h];j<U.pointer[h+1];j++){
                z[U.indices[j]] -= data[firstL[h]]/Dinv[h]*U.data[j]; 
            }  // end for j

            if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of L, small of U):
                for(j=firstUdropped[h];j<droppedU.pointer[h+1];j++){
                    z[droppedU.indices[j]] -= data[firstL[h]]/Dinv[h]*droppedU.data[j];
                } // end for
            } //end if
            h=listL[h];
        } // end while (5.) in algorithm of Saad.
        if(use_improved_SCHUR && !eliminate){  // do improved elimination for Schur complement (large elements of U, small of L):
            while(droppedL_data[k].size()>0){
                h = droppedL_colindex[k].front();  // read (h = corresponding column index in a fixed row of droppedL)
                for(j=firstU[h];j<U.pointer[h+1];j++){
                    z[U.indices[j]] -= droppedL_data[k].front()/Dinv[h]*U.data[j];
                    // if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
                } // end for
                droppedL_colindex[k].pop();  // remove
                droppedL_data[k].pop();
            }  // end while
        } // end if
#ifdef VERBOSE
        time_5=clock();
        time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate && !force_finish && !IP.get_EXTERNAL_FINAL_ROW() && k > IP.get_MIN_ELIM_FACTOR()*n && IP.get_SMALL_PIVOT_TERMINATES() && fabs(z.read(k)) < IP.get_MIN_PIVOT()){  // terminate level because pivot is too small.
            eliminate = false;
            end_level_now = true;
            threshold *= threshold_Schur_factor;
            last_row_to_eliminate = k-1;  // the current row will already be the first row of Anew
            n_Anew = n-k;
            //reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(mem_factor*Arow.non_zeroes()));
            reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Arow.non_zeroes()));
            Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
            if(use_improved_SCHUR){ 
                for(Integer p = 0; p < droppedL_data.dim(); p++) droppedL_data_memory += droppedL_data.read(p).size();
                for(Integer p = 0; p < droppedL_colindex.dim() ; p++) droppedL_colindex_memory += droppedL_colindex.read(p).size();
                droppedL_data_memory *= sizeof(T);
                droppedL_colindex_memory *= sizeof(Integer);
            }
        }
        if(eliminate){  // select pivot scale z/U
            pivot = z.read(k);
            Dinv[k]=1.0/z.read(k);
            z.scale(Dinv[k]);
            z[k]=0.0; // eliminate pivot for sorting
        }
        if(use_weightsLU){
            for(j=0;j<z.non_zeroes();j++) weightsU[z.get_pointer(j)] += fabs(z.get_data(j));
        }
        // (8.) read w
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
#endif
        w.zero_reset();
#ifdef VERBOSE
        time_7=clock();
        time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate){
            // read column of A
            h=headA[k];
            while(h!=-1){
                if(h>k) w[h]=Arow.data[firstA[h]];
                h=listA[h];
            }
            // end while
            h=listU[k];
            while(h!=-1){
                // h is current row index of k-th column of U
                for(j=firstL[h];j<pointer[h+1];j++){
                    // (8.) in the algorithm of Saad.
                    w[indices[j]] -= U.data[firstU[h]]/Dinv[h]*data[j];
                }  // end for j
                h=listU[h];
            } // end while (9.) in algorithm of Saad.
        } // end if 
        w.scale(Dinv[k]);
#ifdef VERBOSE
        time_9=clock();
        time_calc_L += (Real)(time_9-time_7)/(Real)CLOCKS_PER_SEC;
#endif
        if(use_weightsLU){
            for(j=0;j<w.non_zeroes();j++){ 
                //weightsL[w.get_pointer(j)] = max(weightsL[w.get_pointer(j)],fabs(w.get_data(j)));
                weightsL[w.get_pointer(j)] += fabs(w.get_data(j));
            }
        }
        if(IP.get_USE_INVERSE_DROPPING() && eliminate){
            if(k==0){
                xU[k]=1.0; yU[k]=1.0;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)]=vxU[z.get_pointer(j)]=z.get_data(j);
            } else {
                // initialise
                xplus  =  1.0 - vxU[k];
                xminus = -1.0 - vxU[k];
                nplus  = 0;
                nminus = 0;
                yplus  =  1.0 - vyU[k];
                yminus = -1.0 - vyU[k];
                nuplus  = 0.0;
                numinus = 0.0;
                // do x_k
                for(j=0;j<z.non_zeroes();j++) nuplus  += fabs(vxU[z.get_pointer(j)]+z.get_data(j)*xplus);
                for(j=0;j<z.non_zeroes();j++) numinus += fabs(vxU[z.get_pointer(j)]+z.get_data(j)*xminus);
                if(nuplus > numinus) xU[k] = xplus;
                else xU[k] = xminus;
                for(j=0;j<z.non_zeroes();j++) vxU[z.get_pointer(j)] +=  z.get_data(j)*xU[k];
                xU[k]=max(fabs(xplus),fabs(xminus));
                // do y_k
                for(j=0;j<z.non_zeroes();j++){
                    vi=vyU[z.get_pointer(j)];
                    if(fabs(vi+z.get_data(j)*yplus) > max(2.0*fabs(vi),(Real)0.5)) nplus++;
                    if(max(2.0*fabs(vi+z.get_data(j)*yplus),(Real) 0.5)<fabs(vi)) nplus--;
                    if(fabs(vi+z.get_data(j)*yminus) > max(2.0*fabs(vi),(Real) 0.5)) nminus++;
                    if(max(2.0*fabs(vi+z.get_data(j)*yminus),(Real) 0.5)<fabs(vi)) nminus--;
                }
                if(nplus > nminus) yU[k]=yplus;
                else yU[k]= yminus;
                for(j=0;j<z.non_zeroes();j++) vyU[z.get_pointer(j)] += z.get_data(j)*yU[k];
                yU[k]=max(fabs(yplus),fabs(yminus));
            }
        }   // values for dropping are now in xU[k],yU[k]
#ifdef STATISTICS
        L_total[k]= w.non_zeroes(); 
        U_total[k]= z.non_zeroes();
#endif
        if(!eliminate){
            z.take_largest_elements_by_abs_value_with_threshold(norm_U,list_U,max_fill_in,threshold,last_row_to_eliminate+1,n);
        } else {
            weightU=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()){norm = z.norm2(); if(norm==0.0) norm=1e-16; weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP()/norm);} 
            if(IP.get_USE_STANDARD_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_STANDARD_DROP2());  // drop if |w_i|<tau
            if(IP.get_USE_INVERSE_DROPPING())  weightU = IP.combine(weightU,IP.get_WEIGHT_INVERSE_DROP()*max(fabs(xU[k]),fabs(yU[k])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_WEIGHTED_DROP()*weightsU[k]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP()*w.norm1());
            if(IP.get_USE_ERR_PROP_DROPPING2()) weightU = IP.combine(weightU,IP.get_WEIGHT_ERR_PROP_DROP2()*w.norm1()/fabs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING()) weightU = IP.combine(weightU,IP.get_WEIGHT_PIVOT_DROP()*fabs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG()) weightU *= fabs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG()){max_inv_piv = max(max_inv_piv,fabs(Dinv[k])); weightU *= max_inv_piv;}
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k+1,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightsU,weightU,max_fill_in-1,threshold,k+1,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,rejected_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_U does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            } else {
                switch (IP.get_DROP_TYPE_U()){
                    case 0: z.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k+1,n); break; // usual dropping
                    case 1: z.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break; // positional dropping
                    case 2: z.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_U,weightsU,weightU,max_fill_in-1,threshold,k+1,n); // weighted dropping
                    case 3: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    case 4: z.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_U,weightU,max_fill_in-1,threshold,k,n,k+1,bandwidth_U,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_U does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            }
        }
#ifdef STATISTICS
        U_kept[k]= list_U.dimension();
#endif
        // update U or Anew
        if(eliminate){
            if(U.pointer[k]+list_U.dimension()+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+list_U.dimension()+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for U at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<U.pointer[k]+list_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            U.indices[U.pointer[k]]=k;
            for(j=0;j<list_U.dimension();j++){
                pos=U.pointer[k]+j+1;
                U.data[pos]=z.read(list_U[j]);
                U.indices[pos]=list_U[j];
                if(use_norm_row_U) norm_row_U[k] += fabs(U.data[pos]);
                //h=startU[U.indices[pos]];
                //startU[U.indices[pos]]=pos;
                //linkU[pos]=h;
                //rowU[pos]=k;
            }
            U.pointer[k+1]=U.pointer[k]+list_U.dimension()+1;
            if(pivot == 0.0){
                zero_pivots++;
                Dinv[k]=1.0;
#ifdef VERBOSE
                std::cerr<<"matrix_sparse::partialILUC: Preconditioner does not exist (zero pivot). Setting diagonal to 1."<<std::endl;
#endif
            }
            if(use_improved_SCHUR){ // update droppedU
                if(droppedU.pointer[k]+rejected_U.dimension()>reserved_memory_droppedU){
                    reserved_memory_droppedU = 2*(droppedU.pointer[k]+rejected_U.dimension());
                    droppedU.enlarge_fields_keep_data(reserved_memory_droppedU);
                    // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for droppedU at position 1"<<std::endl;
                    // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<droppedU.pointer[k]+rejected_U.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                    // reformat(0,0,0,COLUMN);
                    // U.reformat(0,0,0,ROW);
                    // Dinv.resize_without_initialization(0);
                    // Anew.reformat(0,0,0,ROW);
                    // return false;
                }
                for(j=0;j<rejected_U.dimension();j++){
                    pos=droppedU.pointer[k]+j;
                    droppedU.data[pos]=z.read(rejected_U[j]);
                    droppedU.indices[pos]=rejected_U[j];
                }
                droppedU.pointer[k+1]=droppedU.pointer[k]+rejected_U.dimension();
            }  // end updating droppedU
        } else {
            k_Anew = k -last_row_to_eliminate-1;
            if(U.pointer[k]+1>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k]+1);
                U.enlarge_fields_keep_data(reserved_memory_U);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For Anew: Reserved memory for non-zero elements: "<<reserved_memory_Anew<<" Memory needed: "<<Anew.pointer[k_Anew]+list_U.dimension()<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            if(Anew.pointer[k_Anew]+list_U.dimension()>reserved_memory_Anew){
                reserved_memory_Anew = 2*(Anew.pointer[k_Anew]+list_U.dimension());
                Anew.enlarge_fields_keep_data(reserved_memory_Anew);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for U or Anew at position 3"<<std::endl;
                // std::cerr<<"For Anew: Reserved memory for non-zero elements: "<<reserved_memory_Anew<<" Memory needed: "<<Anew.pointer[k_Anew]+list_U.dimension()<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            U.data[U.pointer[k]]=1.0;
            Dinv[k]=1.0;
            U.indices[U.pointer[k]]=k;
            U.pointer[k+1]=U.pointer[k]+1;
            for(j=0;j<list_U.dimension();j++){
                pos=Anew.pointer[k_Anew]+j;
                Anew.data[pos]=z.read(list_U[j]);
                Anew.indices[pos]=list_U[j];
            }
            Anew.pointer[k_Anew+1]=Anew.pointer[k_Anew]+list_U.dimension();
        }
#ifdef VERBOSE
        time_6=clock();
        time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
#endif
        // (12.) sort and copy data to L
        if(eliminate){
            if(IP.get_USE_INVERSE_DROPPING()){
                if(k==0){
                    xL[k]=1.0; yL[k]=1.0;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)]=vxL[w.get_pointer(j)]=w.get_data(j);
                } else {
                    // initialise
                    xplus  =  1.0 - vxL[k];
                    xminus = -1.0 - vxL[k];
                    nplus  = 0;
                    nminus = 0;
                    yplus  =  1.0 - vyL[k];
                    yminus = -1.0 - vyL[k];
                    nuplus  = 0.0;
                    numinus = 0.0;
                    // do x_k
                    for(j=0;j<w.non_zeroes();j++) nuplus  += fabs(vxL[w.get_pointer(j)]+w.get_data(j)*xplus);
                    for(j=0;j<w.non_zeroes();j++) numinus += fabs(vxL[w.get_pointer(j)]+w.get_data(j)*xminus);
                    if(nuplus > numinus) xL[k] = xplus;
                    else xL[k] = xminus;
                    for(j=0;j<w.non_zeroes();j++) vxL[w.get_pointer(j)] +=  w.get_data(j)*xL[k];
                    xL[k]=max(fabs(xplus),fabs(xminus));
                    // do y_k
                    for(j=0;j<w.non_zeroes();j++){
                        vi=vyL[w.get_pointer(j)];
                        if(fabs(vi+w.get_data(j)*yplus) > max(2.0*fabs(vi),(Real) 0.5)) nplus++;
                        if(max(2.0*fabs(vi+w.get_data(j)*yplus),(Real) 0.5)<fabs(vi)) nplus--;
                        if(fabs(vi+w.get_data(j)*yminus) > max(2.0*fabs(vi),(Real) 0.5)) nminus++;
                        if(max(2.0*fabs(vi+w.get_data(j)*yminus),(Real) 0.5)<fabs(vi)) nminus--;
                    }
                    if(nplus > nminus) yL[k]=yplus;
                    else yL[k]= yminus;
                    for(j=0;j<w.non_zeroes();j++) vyL[w.get_pointer(j)] += w.get_data(j)*yL[k];
                    yL[k]=max(fabs(yplus),fabs(yminus));
                }  // values for dropping are now in xL[k],yL[k]
            }
            weightL=IP.get_NEUTRAL_ELEMENT();
            if(IP.get_USE_STANDARD_DROPPING()) {norm = w.norm2(); if(norm==0.0) norm=1e-16; weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP()/norm);}
            if(IP.get_USE_STANDARD_DROPPING2()) weightL = IP.combine(weightL,IP.get_WEIGHT_STANDARD_DROP2());
            if(IP.get_USE_INVERSE_DROPPING())  weightL = IP.combine(weightL,IP.get_WEIGHT_INVERSE_DROP()*max(fabs(xL[k]),fabs(yL[k])));
            if(IP.get_USE_WEIGHTED_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_WEIGHTED_DROP()*weightsL[k]);
            if(IP.get_USE_ERR_PROP_DROPPING()) weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP()*z.norm1());
            if(IP.get_USE_ERR_PROP_DROPPING2())weightL = IP.combine(weightL,IP.get_WEIGHT_ERR_PROP_DROP2()*z.norm1()/fabs(Dinv[k]));
            if(IP.get_USE_PIVOT_DROPPING())weightL = IP.combine(weightL,IP.get_WEIGHT_PIVOT_DROP()*fabs(Dinv[k]));
            if(IP.get_SCALE_WEIGHT_INVDIAG())  weightL *= fabs(Dinv[k]);
            if(IP.get_SCALE_WGT_MAXINVDIAG())  weightL *= max_inv_piv;
            if(use_improved_SCHUR){
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightsL,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,rejected_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_L does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            } else {
                switch (IP.get_DROP_TYPE_L()){
                    case 0: w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 1: w.take_single_weight_pos_drop_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n,k,last_row_to_eliminate); break;
                    case 2: w.take_single_weight_weighted_largest_elements_by_abs_value_with_threshold(IP,list_L,weightsL,weightL,max_fill_in-1,threshold,k+1,n); break;
                    case 3: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    case 4: w.take_single_weight_bw_largest_elements_by_abs_value_with_threshold(IP,list_L,weightL,max_fill_in-1,threshold,k+1,n,k,bandwidth_L,last_row_to_eliminate); break;
                    default:
                            std::cerr<<"matrix_sparse::partialILUC: DROP_TYPE_L does not have permissible value"<<std::endl;
                            reformat(0,0,0,COLUMN);
                            U.reformat(0,0,0,ROW);
                            Dinv.resize_without_initialization(0);
                            Anew.reformat(0,0,0,ROW);
                            return false;
                }
            }
#ifdef STATISTICS
            L_kept[k]= list_L.dimension();
#endif
            if(pointer[k]+list_L.dimension()+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+list_L.dimension()+1);
                enlarge_fields_keep_data(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for L at position 1"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=k;
            for(j=0;j<list_L.dimension();j++){
                pos = pointer[k]+j+1;
                data[pos] = w.read(list_L[j]); // scaling has already been performed previously
                indices[pos] = list_L[j];
            } // end for j
            pointer[k+1]=pointer[k]+list_L.dimension()+1;
            if(use_improved_SCHUR){ // update droppedL
                for(j=0;j<rejected_L.dimension();j++){
                    pos = rejected_L[j]; // row index of current element
                    droppedL_colindex[pos].push(k);  // store corresponding column index = k
                    droppedL_data[pos].push(w.read(pos));  // store corresponding data element.
                }
            }  // end updating droppedL
        } else {  //  else branch of if(eliminate)
            if(pointer[k]+list_L.dimension()+1>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k]+list_L.dimension()+1);
                enlarge_fields_keep_data(reserved_memory_L);
                // std::cerr<<"matrix_sparse::partialILUC: memory reserved was insufficient. Overflow for L at position 2"<<std::endl;
                // std::cerr<<"Reserved memory for non-zero elements: "<<reserved_memory<<" Memory needed: "<<pointer[k]+list_L.dimension()+1<<" in step "<<k<<" out of "<<n<<" steps."<<std::endl;
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            }
            // copy data
            data[pointer[k]]=1.0;
            indices[pointer[k]]=k;
            pointer[k+1]=pointer[k]+1;
        }  //  end:  if(eliminate)

        if(eliminate){
            update_sparse_matrix_fields(k, Arow.pointer,Arow.indices,listA,headA,firstA);
            update_triangular_fields(k, U.pointer,U.indices,listU,firstU);
            if(use_improved_SCHUR) update_triangular_fields(k, droppedU.pointer,droppedU.indices,listUdropped,firstUdropped);
        }
        update_triangular_fields(k, pointer,indices,listL,firstL);
#ifdef VERBOSE
        time_0=clock();
        time_scu_L += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
#endif
        if(eliminate && IP.get_FINAL_ROW_CRIT() < 11 && !force_finish){
            if(IP.get_EXTERNAL_FINAL_ROW()){
                if (k >= last_row_to_eliminate && k >= IP.get_EXT_MIN_ELIM_FACTOR()*n ){
                    end_level_now = true;
                }  // end if (last_row_to_eliminate == k)
            }
            if(end_level_now){
                eliminate = false;
                threshold *= threshold_Schur_factor;
                last_row_to_eliminate = k;
                n_Anew = n-k-1;
                //reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(mem_factor*Arow.non_zeroes()));
                reserved_memory_Anew = (Integer) min((((Real)(max_fill_in))*((Real)(n_Anew))),(2.0*((Real)n_Anew/(Real) n)*mem_factor*Arow.non_zeroes()));
                Anew.reformat(n_Anew,n_Anew,reserved_memory_Anew,ROW);
                if(use_improved_SCHUR){ 
                    for(Integer p = 0; p < droppedL_data.dim(); p++) droppedL_data_memory += droppedL_data.read(p).size();
                    for(Integer p = 0; p < droppedL_colindex.dim() ; p++) droppedL_colindex_memory += droppedL_colindex.read(p).size();
                    droppedL_data_memory *= sizeof(T);
                    droppedL_colindex_memory *= sizeof(Integer);
                }
            } // end if(end_level_now)
        } // end if (eliminate)
        if (eliminate && IP.get_REQUIRE_ZERO_SCHUR() && IP.get_REQ_ZERO_SCHUR_SIZE()>= n-k-1){
            eliminate = false;
            if(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_L){
                reserved_memory_L = 2*(pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                enlarge_fields_keep_data(reserved_memory_L);
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            } 
            if(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE()>reserved_memory_U){
                reserved_memory_U = 2*(U.pointer[k+1]+IP.get_REQ_ZERO_SCHUR_SIZE());
                U.enlarge_fields_keep_data(reserved_memory_U);
                // reformat(0,0,0,COLUMN);
                // U.reformat(0,0,0,ROW);
                // Dinv.resize_without_initialization(0);
                // Anew.reformat(0,0,0,ROW);
                // return false;
            } 
            n_Anew = IP.get_REQ_ZERO_SCHUR_SIZE();
            Anew.reformat(n_Anew,n_Anew,0,ROW);
            for(j=k+1;j<n;j++){
                data[pointer[j]]=1.0;
                indices[pointer[j]]=j;
                pointer[j+1]=pointer[j]+1;
                U.data[U.pointer[j]]=1.0;
                Dinv[j]=1.0;
                U.indices[U.pointer[j]]=j;
                U.pointer[j+1]=U.pointer[j]+1;
            }
            break;
        }  // end if
    }  // (13.) end for k
#ifdef VERBOSE
    time_2 = clock();
#endif
    Real memory_L_allocated = memory();
    Real memory_U_allocated = U.memory();
    Real memory_Anew_allocated = Anew.memory();
    compress();
    U.compress();
    if(eliminate) Anew.reformat(0,0,0,ROW); // if eliminated till end, then Anew is a 0x0 matrix.
    else {
        if(Anew.nnz>0){
            Anew.compress();
            // resort and shift indices to standard
            for (i=0; i<Anew.rows(); i++)
                for(j=Anew.pointer[i]; j<Anew.pointer[i+1]; j++)
                    Anew.indices[j] -= last_row_to_eliminate+1;
            Anew.normal_order();
            Anew.number_columns=n_Anew; // originally, Anew has n columns
        } else {
            Anew.reformat(n_Anew,n_Anew,0,ROW);
        }
    }
    if(IP.get_USE_POS_COMPRESS()){
        positional_compress(IP,post_fact_threshold);
        U.positional_compress(IP,post_fact_threshold);
    }
#ifdef VERBOSE
    time_3=clock();
    time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
#endif
#ifdef VERBOSE
    time_4=clock();
    time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
    std::cout<<"    partialILUC-Times: "<<std::endl;
    std::cout<<"        initialization:                           "<<time_init<<std::endl;
    std::cout<<"        reading matrix:                           "<<time_read<<std::endl;
    std::cout<<"        sparse zero set:                          "<<time_zeroset<<std::endl;
    std::cout<<"        calculating L:                            "<<time_calc_L<<std::endl;
    std::cout<<"        calculating U:                            "<<time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info L: "<<time_scu_L<<std::endl;
    std::cout<<"        sorting, copying, updating access info U: "<<time_scu_U<<std::endl;
    std::cout<<"        compressing:                              "<<time_compress<<std::endl;
    std::cout<<"        resorting:                                "<<time_resort<<std::endl;
    std::cout<<"      Total times:"<<std::endl;
    std::cout<<"        calculations:                             "<<time_calc_L+time_calc_U<<std::endl;
    std::cout<<"        sorting, copying, updating access info:   "<<time_scu_L+time_scu_U<<std::endl;
    std::cout<<"        other administration:                     "<<time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Grand total                                 "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort<<std::endl;
    std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
#endif
    time_end=clock();
    time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
    total_memory_allocated = w.memory() + z.memory() +  norm_row_U.memory() + row_reorder_weight.memory() + list_L.memory() +
        firstU.memory() + firstL.memory() + firstA.memory() + listA.memory() + headA.memory() + listU.memory() +
        listL.memory() + list_U.memory() + rejected_L.memory() + rejected_U.memory() + droppedU.memory() + vxL.memory() +
        vyL.memory() + vxU.memory() + vyU.memory() + xL.memory() + yL.memory() + xU.memory() + yU.memory() +
        Dinv.memory() + droppedL_data_memory + droppedL_colindex_memory;
    total_memory_used = total_memory_allocated;
    total_memory_allocated += memory_U_allocated + memory_L_allocated + memory_Anew_allocated;
    total_memory_used +=  U.memory() + memory() + Anew.memory();
#ifdef STATISTICS
    // Statistics for A
    matrix_sparse<T> Acol;
    Acol.change_orientation_of_data(Arow);
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Arow.row_density();
    for(k=0;k<n;k++){
        help =  Arow.pointer[k+1]-Arow.pointer[k];
        if (max_total < help) max_total = help;
        if (min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for A"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number: "<<average_total<<std::endl;
    std::cout<<"       Minimum Number     in Row:    "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Row:    "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Row:    "<<stand_dev_total<<std::endl;
    sum1 = 0.0;   max_total=0; min_total=n;
    average_total = Acol.column_density();
    for(k=0;k<n;k++){
        help =  Acol.pointer[k+1]-Acol.pointer[k];
        if(max_total < help) max_total = help;
        if(min_total > help) min_total = help;
        sum1 += (help-average_total)*(help-average_total);
    }
    stand_dev_total = sqrt(sum1/n);
    std::cout<<"       Minimum Number     in Column: "<<min_total<<std::endl;
    std::cout<<"       Maximum Number     in Column: "<<max_total<<std::endl;
    std::cout<<"       Standard Deviation in Column: "<<stand_dev_total<<std::endl;
    // Statistics for L
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) L_kept.read(k))/((Real) L_total.read(k));
        if(max_total < L_total.read(k)) max_total =L_total.read(k);
        if(min_total > L_total.read(k)) min_total =L_total.read(k);
        if(max_kept < L_kept.read(k)) max_kept =L_kept.read(k);
        if(min_kept > L_kept.read(k)) min_kept =L_kept.read(k);
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += L_total.read(k);
        sum2 += L_kept.read(k);
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(L_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) L_kept.read(k))/((Real) L_total.read(k));
        sum1 += (L_total.read(k)-average_total)*(L_total.read(k)-average_total);
        sum2 += (L_kept.read(k)-average_total)*(L_kept.read(k)-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for L"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;
    // Statistics for U
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; min_kept=n; max_total=0; min_total=n; max_kept=0; min_prop=1.0; max_prop=0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) U_kept.read(k))/((Real) U_total.read(k));
        if(max_total < U_total.read(k)) max_total =U_total.read(k);
        if(min_total > U_total.read(k)) min_total =U_total.read(k);
        if(max_kept < U_kept.read(k)) max_kept =U_kept.read(k);
        if(min_kept > U_kept.read(k)) min_kept =U_kept.read(k);
        if(max_prop < prop) max_prop = prop;
        if(min_prop > prop) min_prop = prop;
        sum1 += U_total.read(k);
        sum2 += U_kept.read(k);
        sum3 += prop;
    }
    average_total     = ((Real) sum1) / ((Real)last_row_to_eliminate);
    average_kept      = ((Real) sum2) / ((Real)last_row_to_eliminate);
    average_prop = ((Real) sum3) / ((Real) last_row_to_eliminate);
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for(k=0;k<last_row_to_eliminate;k++){
        if(U_total.read(k) == 0) prop = 1.0;
        else prop = ((Real) U_kept.read(k))/((Real) U_total.read(k));
        sum1 += (U_total.read(k)-average_total)*(U_total.read(k)-average_total);
        sum2 += (U_kept.read(k)-average_total)*(U_kept.read(k)-average_total);
        sum3 += (prop-average_prop)*(prop-average_prop);
    }
    stand_dev_total = sqrt(sum1/last_row_to_eliminate);
    stand_dev_kept  = sqrt(sum2/last_row_to_eliminate);
    stand_dev_prop  = sqrt(sum3/last_row_to_eliminate);
    std::cout<<std::endl;
    std::cout<<"Statistical Data for U"<<std::endl<<std::endl;
    std::cout<<"   Absolute Data: "<<std::endl;
    std::cout<<"       Average Number     before dropping: "<<average_total<<std::endl;
    std::cout<<"       Average Number     after  dropping: "<<average_kept<<std::endl;
    std::cout<<"       Minimum Number     before dropping: "<<min_total<<std::endl;
    std::cout<<"       Minimum Number     after  dropping: "<<min_kept<<std::endl;
    std::cout<<"       Maximum Number     before dropping: "<<max_total<<std::endl;
    std::cout<<"       Maximum Number     after  dropping: "<<max_kept<<std::endl;
    std::cout<<"       Standard Deviation before dropping: "<<stand_dev_total<<std::endl;
    std::cout<<"       Standard Deviation after  dropping: "<<stand_dev_kept<<std::endl;
    std::cout<<"   Relative Data: "<<std::endl;
    std::cout<<"       Average            Proportion kept:            "<<average_prop<<std::endl;
    std::cout<<"       Standard Deviation Proportion kept:            "<<stand_dev_prop<<std::endl;

#endif
    return true;
}


template<class T> bool matrix_sparse<T>::preprocessed_partialILUCDP(const iluplusplus_precond_parameter& IP, bool force_finish, const matrix_sparse<T>& A, matrix_sparse<T>& Acoarse, matrix_sparse<T>& U, vector_dense<T>& Dinv,
          index_list& permutation_rows, index_list& permutation_columns, index_list& inverse_permutation_rows, index_list& inverse_permutation_columns, vector_dense<T>& D_l, vector_dense<T>& D_r,
          Integer max_fill_in, Real threshold, Real perm_tol, Integer& zero_pivots, Real& setup_time, Real mem_factor, Real& total_memory_allocated, Real& total_memory_used)
              {
                  bool use_ILUC;
                  if( (IP.get_PERMUTE_ROWS() == 0 || (IP.get_PERMUTE_ROWS() == 1 && !IP.get_EXTERNAL_FINAL_ROW()))  && (!IP.get_BEGIN_TOTAL_PIV() || (IP.get_BEGIN_TOTAL_PIV() && IP.get_TOTAL_PIV() == 0) ) && IP.get_perm_tol() > 500.0) use_ILUC = true;
                  else use_ILUC = false;
                  clock_t time_1,time_2;
                  time_1 = clock();
                  matrix_sparse<T> Arow2,Acol;
                  bool factorization_exists;
                  index_list pr1, pr2, ipr1,ipr2,pc1,pc2,ipc1,ipc2;
                  Real partial_setup_time;
                  setup_time = 0.0;
                  Integer last_row_to_eliminate,bp,bpr,epr,end_PQ;
                  if(A.orient() == ROW){
                      end_PQ = Arow2.preprocess(A,IP,pr1,pc1,ipr1,ipc1,D_l,D_r);
                  } else {
                      Arow2.change_orientation_of_data(A);
                      end_PQ = Arow2.preprocess(IP,pr1,pc1,ipr1,ipc1,D_l,D_r);
                  }
                  if (force_finish) {
                      last_row_to_eliminate = Arow2.rows()-1;
                  } else {
                      if (IP.get_EXTERNAL_FINAL_ROW()) last_row_to_eliminate = end_PQ-1;
                      else last_row_to_eliminate = (Arow2.rows()-1)/2;
                  }
                  switch (IP.get_PERMUTE_ROWS()) {
                      case 0:  bpr = 0; epr = 0; break;
                      case 1:  if(IP.get_EXTERNAL_FINAL_ROW() && (!force_finish)){bpr = 0; epr = 0;} else {bpr = end_PQ; epr = Arow2.rows()-1;} break;
                      case 2:  if(force_finish){bpr = 0; epr = Arow2.rows()-1;} else {bpr = 0; epr = last_row_to_eliminate;} break;
                      case 3:  bpr = 0; epr = Arow2.rows()-1; break;
                      default: std::cerr<<"matrix_sparse::preprocessed_partialILUCDP::choose permissible value for PERMUTE_ROWS!"<<std::endl; 
                               reformat(0,0,0,COLUMN);
                               U.reformat(0,0,0,ROW);
                               Dinv.resize_without_initialization(0);
                               D_l.resize_without_initialization(0);
                               D_r.resize_without_initialization(0);
                               Acoarse.reformat(0,0,0,ROW);
                               permutation_rows.resize(0);
                               permutation_columns.resize(0);
                               inverse_permutation_rows.resize(0);
                               inverse_permutation_columns.resize(0);
                               return false;
                  }
                  switch (IP.get_TOTAL_PIV()) {
                      case 0:  bp = Acol.rows(); break;
                      case 1:  if(force_finish){bp = end_PQ;} else {bp = last_row_to_eliminate+1;} break;
                      case 2:  bp = 0;  break;
                      default: std::cerr<<"matrix_sparse::preprocessed_partialILUCDP::choose permissible value for TOTAL_PIV!"<<std::endl;
                               reformat(0,0,0,COLUMN);
                               U.reformat(0,0,0,ROW);
                               Dinv.resize_without_initialization(0);
                               D_l.resize_without_initialization(0);
                               D_r.resize_without_initialization(0);
                               Acoarse.reformat(0,0,0,ROW);
                               permutation_rows.resize(0);
                               permutation_columns.resize(0);
                               inverse_permutation_rows.resize(0);
                               inverse_permutation_columns.resize(0);
                               return false;
                  }
#ifdef INFO
                  std::cout<<std::endl;
                  std::cout<<"  ** matrix statistics:"<<std::endl;
                  std::cout<<"     n                      = "<<Arow2.rows()<<std::endl;
                  std::cout<<"     nnz                    = "<<Arow2.actual_non_zeroes()<<std::endl;
                  std::cout<<"     density                = "<<Arow2.row_density()<<std::endl;
                  std::cout<<"  ** factorization parameters:"<<std::endl;
                  std::cout<<"     max. numb. nnz/row p   = "<<max_fill_in<<std::endl;
                  std::cout<<"     tau                    = "<<threshold<<std::endl;
                  std::cout<<"     perm tolerance         = "<<perm_tol<<std::endl;
                  std::cout<<"     begin permuting rows   = "<<bpr<<std::endl;
                  std::cout<<"     end   permuting rows   = "<<epr<<std::endl;
                  if(IP.get_EXTERNAL_FINAL_ROW())
                      std::cout<<"     last row to eliminate  = "<<last_row_to_eliminate;
                  else
                      std::cout<<"     last row to eliminate decided by factorization."<<std::endl;
                  std::cout<<std::endl;
#endif
                  if (use_ILUC){
                      factorization_exists = partialILUC(Arow2,Acoarse,IP,force_finish,U,Dinv,last_row_to_eliminate,threshold,zero_pivots,partial_setup_time,mem_factor,total_memory_allocated,total_memory_used);
                  } else {
                      Acol.change_orientation_of_data(Arow2);
                      factorization_exists = partialILUCDP(Arow2,Acol,Acoarse,IP,force_finish,U,Dinv,pc2,pr2,ipc2,ipr2,last_row_to_eliminate,threshold,bp,bpr,epr,zero_pivots,partial_setup_time,mem_factor,total_memory_allocated,total_memory_used);
                  }
                  if(!factorization_exists) return false;
#ifdef INFO
                  std::cout<<"     zero-pivots            = "<<zero_pivots<<std::endl;
                  std::cout<<"     local fill-in          = "<<((Real)(actual_non_zeroes()+U.actual_non_zeroes())- (Real) Acol.rows() )/((Real)Acol.actual_non_zeroes())<<std::endl;
#endif
                  if(use_ILUC){
                      permutation_columns=pc1;
                      permutation_rows=pr1;
                  } else {
                      permutation_columns.compose(pc1,pc2);
                      permutation_rows.compose(pr1,pr2);
                  }
                  inverse_permutation_columns.invert(permutation_columns);
                  inverse_permutation_rows.invert(permutation_rows);
                  time_2 = clock();
                  setup_time = ((Real)time_2-(Real)time_1)/(Real)CLOCKS_PER_SEC;
                  return true;
}


/*
template<class T> bool matrix_sparse<T>::ILUCDPinv(const matrix_sparse<T>& Arow, const matrix_sparse<T>& Acol, matrix_sparse<T>& U, index_list& perm, index_list& permrows, Integer max_fill_in, Real threshold, Real perm_tol,  Integer bpr, Integer& zero_pivots, Real& time_self, Real mem_factor){
    try {
      clock_t time_begin, time_end;
      time_begin=clock();
      if (threshold > 500.0) threshold=0.0;
      else threshold=std::exp(-threshold*std::log(10.0));
      if (perm_tol > 500.0) perm_tol=0.0;
      else perm_tol=std::exp(-perm_tol*std::log(10.0));
      #ifdef VERBOSE
          clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_8,time_9;
          Real time_init=0.0;
          Real time_read=0.0;
          Real time_calc_L=0.0;
          Real time_scu_L=0.0;  // sorting, copying, updating access information
          Real time_calc_U=0.0;
          Real time_scu_U=0.0;
          Real time_zeroset=0.0;
          Real time_compress=0.0;
          Real time_resort=0.0;
          Real time_weights=0.0;
          time_0 = clock();
      #endif
      if(non_fatal_error(!Arow.square_check(),"matrix_sparse::ILUCDPinv: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      if(non_fatal_error(!Acol.square_check(),"matrix_sparse::ILUCDPinv: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      if(non_fatal_error(Acol.rows()!=Arow.rows(),"matrix_sparse::ILUCDPinv: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      Integer n = Acol.columns();
      Integer a,b,k,i,j,p,current_row_col_U,current_col_row_L;
      Integer h,pos, selected_row;
      T current_data_row_L,current_data_col_U;
      zero_pivots=0;
      Real norm_L,norm_U; // this variable is needed to call take_largest_elements_by_absolute_value, but serves no purpose in this routine.
      vector_sparse_dynamic<T> w(n), z(n);
      vector_dense<bool> non_pivot(n,true);
      vector_dense<bool> unused_rows(n,true);
      vector_dense<Integer> numb_el_row_L(n,0), pointer_num_el_row_L(n+1,n);
      vector_dense<T> weights_L(n+1);
      vector_dense<T> weights_U(n+1);
      Real xiplus_L,ximinus_L;
      pointer_num_el_row_L[0]=0;
      index_list list_L, list_U;
      index_list inverse_perm(n), inverse_permrows(n);
      if(max_fill_in<1) max_fill_in = 1;
      if(max_fill_in>n) max_fill_in = n;
      Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*Acol.non_zeroes());
      array<Integer> linkU(reserved_memory); //h=link[startU[i]]] points to second 2nd element, link[h] to next, etc.
      array<Integer> rowU(reserved_memory);   // row indices of elements of U.data.
      array<Integer> startU(n); // startU[i] points to start of points to an index of data belonging to column i 
      array<Integer> linkL(reserved_memory); //h=link[startL[i]]] points to second 2nd element, link[h] to next, etc.
      array<Integer> colL(reserved_memory);  // column indices of elements of data.
      array<Integer> startL(n); // startL[i] points to start of points to an index of data belonging to row i 
      U.reformat(n,n,reserved_memory,ROW);
      U.pointer[0]=0;
      weights_L[0]=1.0;
      reformat(n,n,reserved_memory,COLUMN);
      pointer[0]=0;
      perm.resize(n);
      permrows.resize(n);
      for(k=0;k<n;k++) startU[k]=-1;
      for(k=0;k<n;k++) startL[k]=-1;
      // (1.) begin for k
      #ifdef VERBOSE
          time_1 = clock();
          time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
      #endif
      for(k=0;k<n;k++){
          #ifdef VERBOSE
              time_2=clock();
          #endif
          // (2.) initialize z
          selected_row = permrows[k];
        unused_rows[selected_row]=false;
          z.zero_reset();
          #ifdef VERBOSE
              time_3=clock();
              time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
          #endif
          // read row of A
          for(i=Arow.pointer[selected_row];i<Arow.pointer[selected_row+1];i++){
              if(non_pivot[Arow.indices[i]]) z[Arow.indices[i]] = Arow.data[i];
          }     // end for i
          #ifdef VERBOSE
              time_4=clock();
              time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
          #endif
          // (3.) begin while
          h=startL[selected_row]; // h=startL[permrows[k]];
          while(h!=-1){
              current_col_row_L=colL[h];
              current_data_row_L=data[h];
              h=linkL[h];
              for(j=U.pointer[current_col_row_L];j<U.pointer[current_col_row_L+1];j++){
                  if(non_pivot[U.indices[j]]) z[U.indices[j]] -= current_data_row_L*U.data[j];
              } // end for
          }   // (5.) end while
          #ifdef VERBOSE
              time_5=clock();
              time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
          #endif
          // (6.) sort and copy data to U; update information for accessing columns of U
          z.take_single_weight_largest_elements_by_abs_value_with_threshold_pivot_last(list_U,weights_U,max_fill_in,threshold,perm[k],perm_tol);
              // dropping too stringent?
          if(list_U.dimension()==0){
              if(threshold>0.0)
                  #ifdef VERBOSE
                      std::cout<<"Dropping too stringent, selecting elements without threshold."<<std::endl;
                  #endif
                  z.take_largest_elements_by_abs_value_with_threshold_pivot_last(norm_U,list_U,max_fill_in,0.0,perm[k],perm_tol);
          }
          // still no non-zero elements?
          if(list_U.dimension()==0){
              #ifdef VERBOSE
                 std::cout<<"Obtained a zero row, setting an arbitrary element to 1."<<std::endl;
              #endif
              zero_pivots++;
              z[perm[k]]=1.0;
              list_U.resize(1);
              list_U[0]=perm[k];
          } // end if
          if(U.pointer[k]+list_U.dimension()>reserved_memory){
              std::cerr<<"matrix_sparse::ILUCDPinv: memory reserved was insufficient."<<std::endl;
              return false;
          }
          // copy data, update access information.
          // copy pivot
          U.data[U.pointer[k]]=z.read(list_U[list_U.dimension()-1]);
          U.indices[U.pointer[k]]=list_U[list_U.dimension()-1];
          for(j=1;j<list_U.dimension();j++){
              pos=U.pointer[k]+j;
              U.data[pos]=z.read(list_U[list_U.dimension()-1-j]);
              U.indices[pos]=list_U[list_U.dimension()-1-j];
              h=startU[U.indices[pos]];
              startU[U.indices[pos]]=pos;
              linkU[pos]=h;
              rowU[pos]=k;
          }
          U.pointer[k+1]=U.pointer[k]+list_U.dimension();
          if(U.data[U.pointer[k]]==0){
              std::cerr<<"matrix_sparse::ILUCDPinv: Pivot is zero, because pivoting was not permitted. Preconditioner does not exist."<<std::endl;
              std::cout<<"dim list_U "<<list_U.dimension()<<std::endl;
              std::cout<<"last element corresponding to pivot: "<<z[perm[k]]<<std::endl; 
              return false;
          }
          // store positions of columns of U, but without pivot
          // update non-pivots.
          // (7.) update permutations
          p=inverse_perm[U.indices[U.pointer[k]]];
          inverse_perm.switch_index(perm[k],U.indices[U.pointer[k]]);
          perm.switch_index(k,p);
          non_pivot[U.indices[U.pointer[k]]]=false;
          #ifdef VERBOSE
              time_6=clock();
              time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
          #endif
           // (8.) read w
          w.zero_reset();
          #ifdef VERBOSE
              time_7=clock();
              time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
          #endif
          // read column of A
          for(i=Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
              if(unused_rows[Acol.indices[i]])
                  w[Acol.indices[i]] = Acol.data[i];
          }     // end for i
          #ifdef VERBOSE
              time_8=clock();
              time_read += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
          #endif
          // (9.) begin while
          h=startU[perm[k]];
          while(h!=-1){
              current_row_col_U=rowU[h];
              current_data_col_U=U.data[h];
              h=linkU[h];
             // (10.) w = w - U(i,perm(k))*l_i
              for(j=pointer[current_row_col_U];j<pointer[current_row_col_U+1];j++){
                  if(unused_rows[indices[j]]) w[indices[j]] -= current_data_col_U*data[j];
              } // end for
          }   // (11.) end while
         #ifdef VERBOSE
              time_9=clock();
              time_calc_L += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
          #endif
          // (12.) sort and copy data to L
          // sort
          w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP, list_L,fabs(weights_L.read(k)),max_fill_in-1,threshold,0,n);
          if(pointer[k]+list_L.dimension()+1>reserved_memory){
              std::cerr<<"matrix_sparse::ILUCDPinv: memory reserved was insufficient."<<std::endl;
              return false;
          }
          // copy data
          data[pointer[k]]=1.0;
          indices[pointer[k]]=selected_row;
          for(j=0;j<list_L.dimension();j++){
              pos = pointer[k]+j+1;
              data[pos] = w.read(list_L[j])/U.data[U.pointer[k]];
              b = indices[pos] = list_L[j];
              h=startL[b];
              startL[b]=pos;
              linkL[pos]=h;
              colL[pos]=k;
              // begin updating fields for number elements of row of L
              if (b > bpr) {
                  b = inverse_permrows[b];
                  a = --pointer_num_el_row_L[++numb_el_row_L[b]];
                  inverse_permrows.switch_index(permrows[a],permrows[b]);
                  permrows.switch_index(a,b);
                  numb_el_row_L.switch_entry(a,b);
              }
              // end updating fields
          } // end for j
          // sort permrows if necessary, i.e. if num_el_row_L increases at next iteration.
              if(pointer_num_el_row_L[numb_el_row_L[k]+1] == k+1) 
                  permrows.quicksort_with_inverse(inverse_permrows,pointer_num_el_row_L[numb_el_row_L[k]+1],pointer_num_el_row_L[numb_el_row_L[k]+2]-1);
          // end sorting
          pointer[k+1]=pointer[k]+list_L.dimension()+1;
          //if (k == bpr) threshold /= 100.0;
         #ifdef VERBOSE
              time_0=clock();
              time_scu_L += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
          #endif
           // update weights
          for(j=pointer[k]+1;j<pointer[k+1];j++){
               weights_L[indices[j]] -= weights_L[k]*data[j];
          }
          xiplus_L=1.0+weights_L.read(k+1);
          ximinus_L=-1.0+weights_L.read(k+1);
          if(fabs(xiplus_L)<fabs(ximinus_L))weights_L[k+1]=ximinus_L;
          else weights_L[k+1]=xiplus_L;
          for(j=U.pointer[k]+1;j<U.pointer[k+1];j++){
              weights_U[U.indices[j]] -= weights_U[perm[k]]*U.data[j];
          }
          #ifdef VERBOSE
              time_1=clock();
              time_weights += (Real)(time_1-time_0)/(Real)CLOCKS_PER_SEC;
          #endif
      }  // (13.) end for k
      #ifdef VERBOSE
          time_2 = clock();
      #endif
      compress();
      U.compress();
      #ifdef VERBOSE
          time_3=clock();
          time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
      #endif
      #ifdef VERBOSE
          time_4=clock();
          time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
          std::cout<<"    ILUCDPinv-Times: "<<std::endl;
          std::cout<<"        initialization:                           "<<time_init<<std::endl;
          std::cout<<"        reading matrix:                           "<<time_read<<std::endl;
          std::cout<<"        sparse zero set:                          "<<time_zeroset<<std::endl;
          std::cout<<"        calculating L:                            "<<time_calc_L<<std::endl;
          std::cout<<"        calculating U:                            "<<time_calc_U<<std::endl;
          std::cout<<"        sorting, copying, updating access info L: "<<time_scu_L<<std::endl;
          std::cout<<"        sorting, copying, updating access info U: "<<time_scu_U<<std::endl;
          std::cout<<"        compressing:                              "<<time_compress<<std::endl;
          std::cout<<"        resorting:                                "<<time_resort<<std::endl;
          std::cout<<"        updating weights:                         "<<time_weights<<std::endl;
          std::cout<<"      Total times:"<<std::endl;
          std::cout<<"        calculations:                             "<<time_calc_L+time_calc_U<<std::endl;
          std::cout<<"        sorting, copying, updating access info:   "<<time_scu_L+time_scu_U<<std::endl;
          std::cout<<"        other administration:                     "<<time_init+time_read+time_zeroset+time_compress+time_resort+time_weights<<std::endl;
          std::cout<<"      Grand total                                 "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort+time_weights<<std::endl;
          std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
      #endif
      time_end=clock();
      time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
      return true;
   }
   catch(iluplusplus_error ippe){
      std::cerr << "matrix_sparse<T>:: ILUCDPinv: "<<ippe.error_message() << std::endl;
      throw;
   }
  }
*/

/*
template<class T> bool matrix_sparse<T>::ILUCP4inv(const matrix_sparse<T>& Acol, matrix_sparse<T>& U, index_list& perm, Integer max_fill_in, Real threshold, Real perm_tol,Integer rp, Integer& zero_pivots, Real& time_self, Real mem_factor){
  try {
      clock_t time_begin, time_end;
      time_begin=clock();
      if (threshold > 500.0) threshold=0.0;
      else threshold=std::exp(-threshold*std::log(10.0));
      if (perm_tol > 500.0) perm_tol=0.0;
      else perm_tol=std::exp(-perm_tol*std::log(10.0));
      #ifdef VERBOSE
          clock_t time_0, time_1, time_2, time_3, time_4,time_5,time_6,time_7,time_8,time_9;
          Real time_init=0.0;
          Real time_read=0.0;
          Real time_calc_L=0.0;
          Real time_scu_L=0.0;  // sorting, copying, updating access information
          Real time_calc_U=0.0;
          Real time_scu_U=0.0;
          Real time_zeroset=0.0;
          Real time_compress=0.0;
          Real time_resort=0.0;
          Real time_weights=0.0;
          time_0 = clock();
      #endif
      if(non_fatal_error(!Acol.square_check(),"matrix_sparse::ILUCP4inv: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
      Integer n = Acol.columns();
      Integer k,i,j,p,current_row_col_U;
      Integer h,pos;
      T current_data_col_U;
      zero_pivots=0;
      vector_sparse_dynamic<T> w(n), z(n);
      vector_dense<bool> non_pivot(n,true);
      index_list list_L, list_U;
      vector_dense<T> weights_L(n+1);
      vector_dense<T> weights_U(n+1);
      Real xiplus_L,ximinus_L;
      array<Integer> firstL(n);
      array<Integer>listL(n);
      array<Integer> firstA(n);
      array<Integer> listA(n);
      array<Integer> headA(n);
      index_list inverse_perm(n);
      if(max_fill_in<1) max_fill_in = 1;
      if(max_fill_in>n) max_fill_in = n;
      Integer reserved_memory = min(max_fill_in*n, (Integer) mem_factor*Acol.non_zeroes());
      array<Integer> linkU(reserved_memory); //h=link[startU[i]]] points to second 2nd element, link[h] to next, etc.
      array<Integer> rowU(reserved_memory); // row indices of elements of U.data.
      array<Integer> startU(reserved_memory); // startU[i] points to start of points to an index of U.data belonging to column i 
      U.reformat(n,n,reserved_memory,ROW);
      U.pointer[0]=0;
      reformat(n,n,reserved_memory,COLUMN);
      pointer[0]=0;
      perm.resize(n);
      weights_L[0]=1.0;
      initialize_triangular_fields(n,listL);
      initialize_sparse_matrix_fields(n,Acol.pointer,Acol.indices,listA,headA,firstA);
      for(k=0;k<n;k++) startU[k]=-1;
      // (1.) begin for k
      #ifdef VERBOSE
          time_1 = clock();
          time_init = ((Real)time_1-(Real)time_0)/(Real)CLOCKS_PER_SEC;
      #endif
      for(k=0;k<n;k++){
          #ifdef VERBOSE
              time_2=clock();
              time_scu_L += (Real)(time_2-time_1)/(Real)CLOCKS_PER_SEC;
          #endif
          if (k == rp) perm_tol = 1.0;  // permute always
          // (2.) initialize z
          z.zero_reset();
          #ifdef VERBOSE
              time_3=clock();
              time_zeroset += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
          #endif
          // read row of A
          h=headA[k];
          while(h!=-1){
              if(non_pivot[h]) z[h]=Acol.data[firstA[h]];
              h=listA[h];
          }
          #ifdef VERBOSE
              time_4=clock();
              time_read += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
          #endif
          // (3.) begin while
          h=listL[k];
          while(h!=-1){
              // h is current column index of k-th row of L
              for(j=U.pointer[h];j<U.pointer[h+1];j++){
                      if(non_pivot[U.indices[j]]){
                         z[U.indices[j]] -= data[firstL[h]]*U.data[j];
                      } // end if
              } // end for
            h=listL[h];
          } // end while (5.) 
          #ifdef VERBOSE
              time_5=clock();
              time_calc_U += (Real)(time_5-time_4)/(Real)CLOCKS_PER_SEC;
          #endif
          // (6.) sort and copy data to U; update information for accessing columns of U
          z.take_single_weight_largest_elements_by_abs_value_with_threshold_pivot_last(list_U,weights_U,max_fill_in,threshold,perm[k],perm_tol);
          // still no non-zero elements?
          if(list_U.dimension()==0){
              #ifdef VERBOSE
                 std::cout<<"Obtained a zero row, setting an arbitrary element to 1."<<std::endl;
              #endif
              zero_pivots++;
              z[perm[k]]=1.0;
              list_U.resize(1);
              list_U[0]=perm[k];
          } // end if
          if(U.pointer[k]+list_U.dimension()>reserved_memory){
              std::cerr<<"matrix_sparse::ILUCP4inv: memory reserved was insufficient."<<std::endl;
              return false;
          }
          // copy data, update access information.
          // copy pivot
          U.data[U.pointer[k]]=z.read(list_U[list_U.dimension()-1]);
          U.indices[U.pointer[k]]=list_U[list_U.dimension()-1];
          for(j=1;j<list_U.dimension();j++){
              pos=U.pointer[k]+j;
              U.data[pos]=z.read(list_U[list_U.dimension()-1-j]);
              U.indices[pos]=list_U[list_U.dimension()-1-j];
              h=startU[U.indices[pos]];
              startU[U.indices[pos]]=pos;
              linkU[pos]=h;
              rowU[pos]=k;
          }
          U.pointer[k+1]=U.pointer[k]+list_U.dimension();
          if(U.data[U.pointer[k]]==0){
              std::cerr<<"matrix_sparse::ILUCP4inv: Pivot is zero. Preconditioner does not exist."<<std::endl;
              return false;
          }
          // store positions of columns of U, but without pivot
          // update non-pivots.
          // (7.) update permutations
          p=inverse_perm[U.indices[U.pointer[k]]];
          inverse_perm.switch_index(perm[k],U.indices[U.pointer[k]]);
          perm.switch_index(k,p);
          non_pivot[U.indices[U.pointer[k]]]=false;
          #ifdef VERBOSE
              time_6=clock();
              time_scu_U += (Real)(time_6-time_5)/(Real)CLOCKS_PER_SEC;
          #endif
           // (8.) read w
          w.zero_reset();
          #ifdef VERBOSE
              time_7=clock();
              time_zeroset += (Real)(time_7-time_6)/(Real)CLOCKS_PER_SEC;
          #endif
          for(i=Acol.pointer[perm[k]];i<Acol.pointer[perm[k]+1];i++){
              if(Acol.indices[i]>k)
                  w[Acol.indices[i]] = Acol.data[i];
          }     // end for i
          #ifdef VERBOSE
              time_8=clock();
              time_read += (Real)(time_8-time_7)/(Real)CLOCKS_PER_SEC;
          #endif
          // (9.) begin while
          h=startU[perm[k]];
          while(h!=-1){
              current_row_col_U=rowU[h];
              current_data_col_U=U.data[h];
              h=linkU[h];
              // (10.) w = w - U(i,perm(k))*l_i
              for(j=pointer[current_row_col_U];j<pointer[current_row_col_U+1];j++){
                  w[indices[j]] -= current_data_col_U*data[j];
              } // end for
          }   // (11.) end while
          #ifdef VERBOSE
              time_9=clock();
              time_calc_L += (Real)(time_9-time_8)/(Real)CLOCKS_PER_SEC;
          #endif
         // (12.) sort and copy data to L
         // sort
          w.take_single_weight_largest_elements_by_abs_value_with_threshold(IP, list_L,fabs(weights_L.read(k)),max_fill_in-1,threshold,0,n);
          if(pointer[k]+list_L.dimension()+1>reserved_memory){
              std::cerr<<"matrix_sparse::ILUCP4inv: memory reserved was insufficient."<<std::endl;
              return false;
          }
          // copy data
          data[pointer[k]]=1.0;
          indices[pointer[k]]=k;
          for(j=0;j<list_L.dimension();j++){
              data[pointer[k]+j+1] = w.read(list_L[j])/U.data[U.pointer[k]];
              indices[pointer[k]+j+1] = list_L[j];
          } // end for j
          pointer[k+1]=pointer[k]+list_L.dimension()+1;
          update_sparse_matrix_fields(k, Acol.pointer,Acol.indices,listA,headA,firstA);
          update_triangular_fields(k, pointer,indices,listL,firstL);
          #ifdef VERBOSE
              time_0=clock();
              time_scu_L += (Real)(time_0-time_9)/(Real)CLOCKS_PER_SEC;
          #endif
          // update weights
          for(j=pointer[k]+1;j<pointer[k+1];j++){
               weights_L[indices[j]] -= weights_L[k]*data[j];
          }
          xiplus_L=1.0+weights_L.read(k+1);
          ximinus_L=-1.0+weights_L.read(k+1);
          if(fabs(xiplus_L)<fabs(ximinus_L))weights_L[k+1]=ximinus_L;
          else weights_L[k+1]=xiplus_L;
          for(j=U.pointer[k]+1;j<U.pointer[k+1];j++){
              weights_U[U.indices[j]] -= weights_U[perm[k]]*U.data[j];
          }
          #ifdef VERBOSE
              time_1=clock();
              time_weights += (Real)(time_1-time_0)/(Real)CLOCKS_PER_SEC;
          #endif
         // other calculations needed to update weights_U are done while selecting elements to keep for U.
      }  // (13.) end for k
      #ifdef VERBOSE
          time_2 = clock();
      #endif
      compress();
      U.compress();
      #ifdef VERBOSE
          time_3=clock();
          time_compress += (Real)(time_3-time_2)/(Real)CLOCKS_PER_SEC;
      #endif
      #ifdef VERBOSE
          time_4=clock();
          time_resort += (Real)(time_4-time_3)/(Real)CLOCKS_PER_SEC;
          std::cout<<"    ILUCP4inv-Times: "<<std::endl;
          std::cout<<"        initialization:                           "<<time_init<<std::endl;
          std::cout<<"        reading matrix:                           "<<time_read<<std::endl;
          std::cout<<"        sparse zero set:                          "<<time_zeroset<<std::endl;
          std::cout<<"        calculating weights:                      "<<time_weights<<std::endl;
          std::cout<<"        calculating L:                            "<<time_calc_L<<std::endl;
          std::cout<<"        calculating U:                            "<<time_calc_U<<std::endl;
          std::cout<<"        sorting, copying, updating access info L: "<<time_scu_L<<std::endl;
          std::cout<<"        sorting, copying, updating access info U: "<<time_scu_U<<std::endl;
          std::cout<<"        compressing:                              "<<time_compress<<std::endl;
          std::cout<<"        resorting:                                "<<time_resort<<std::endl;
          std::cout<<"      Total times:"<<std::endl;
          std::cout<<"        calculations:                             "<<time_calc_L+time_calc_U<<std::endl;
          std::cout<<"        sorting, copying, updating access info:   "<<time_scu_L+time_scu_U<<std::endl;
          std::cout<<"        other administration:                     "<<time_init+time_read+time_zeroset+time_compress+time_resort+time_weights<<std::endl;
          std::cout<<"      Grand total                                 "<<time_calc_L+time_calc_U+time_scu_L+time_scu_U+time_init+time_read+time_zeroset+time_compress+time_resort+time_weights<<std::endl;
          std::cout<<"      Encountered "<<zero_pivots<<" zero pivots that were set to 1."<<std::endl;
      #endif
      time_end=clock();
      time_self=((Real)time_end-(Real)time_begin)/(Real)CLOCKS_PER_SEC;
      return true;
   }
   catch(iluplusplus_error ippe){
      std::cerr << "matrix_sparse<T>:: ILUCP4inv: "<<ippe.error_message() << std::endl;
      throw;
   }
  }
*/

#ifdef ILUPLUSPLUS_USES_SPARSPAK

template<class T> void matrix_sparse<T>::rcm(){
    matrix_sparse<T> A;
    A.rcm(*this);
    copy_and_destroy(A);
 }

template<class T> void matrix_sparse<T>::rcm(const matrix_sparse<T>& A){
    index_list P;
    RCM<T>(A,P);
    permute(A,P,P);
 }

template<class T> void matrix_sparse<T>::rcm(index_list& P) const {
    RCM<T>(*this,P);
}

template<class T> void matrix_sparse<T>::rcm(index_list& P, Integer b, Integer e) const {
    RCM<T>(*this,P,b,e);
}

#endif

template<class T> Integer matrix_sparse<T>::choose_ddPQ(const iluplusplus_precond_parameter& IP, index_list& P, index_list& Q) const {
    switch(IP.get_PQ_ALGORITHM()){
        case 0:   return ddPQ(P,Q,IP.get_PQ_THRESHOLD());        // standard greedy PQ;
        case 1:   return ddPQ_dyn_av(P,Q,IP.get_PQ_THRESHOLD()); // dynamic averaging;
        case 3:   return symm_ddPQ_dyn_av(P,Q,IP.get_PQ_THRESHOLD()); // symmetrized dynamic averaging;
        default:  return ddPQ(P,Q,IP.get_PQ_THRESHOLD());        // standard greedy PQ;
    }
}


template<class T> Integer matrix_sparse<T>::ddPQ(index_list& P, index_list& Q, Real tau) const
{
    if(non_fatal_error(!square_check(),"matrix_sparse::ddPQ: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer j,k,count,Qcount,pos;
    Integer n = columns();
    //Real tau = 0.01;
    Real current_max;
    Real divisor=0.0;
    index_list I(n);
    std::vector<Integer> J(n);
    vector_dense<Real> W(n);
    P.resize_with_constant_value(n,-1);
    Q.resize_with_constant_value(n,-1);
    for(k=0; k<n; k++) {
        current_max = 0.0;
        W[k] = 0.0;
        J[k] = 0;

        // find column J[k] with maximum absolute value in row k
        // W[k] is L_1 norm of row k
        for(j=pointer[k]; j<pointer[k+1]; j++){
            W[k] += std::abs(data[j]);
            if (std::abs(data[j]) > current_max) {
                current_max = std::abs(data[j]);
                J[k] = indices[j];
            }
        }
        divisor = W[k] * (pointer[k+1]-pointer[k]);
        //divisor = W[k];
        if(divisor == 0.0)
            W[k] = 0.0;
        else
            W[k] = -current_max / divisor;
    }
    W.quicksort(I,0,n-1);
    J = permute_vec(J, I);
    count = -1;
    for (k=0;k<n;k++){
        if ((P[I[k]] == -1) && (Q[J[k]] == -1) && (-W[k] >= tau)) {
            count++;
            P[I[k]] = count;
            Q[J[k]] = count;
        }
    }
    pos = Qcount = count;
    for (k=0;k<n;k++){
        if(P[k]<0){
            count++;
            P[k] = count;
        }
    }
    for (k=0;k<n;k++){
        if(Q[k]<0){
            Qcount++;
            Q[k] = Qcount;
        }
    }
    return pos+1;
}


template<class T> Integer matrix_sparse<T>::ddPQ_dyn_av(index_list& P, index_list& Q, Real tau) const {
    if(non_fatal_error(!square_check(),"matrix_sparse::ddPQ_dyn_av: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer i,j,k,count,Qcount,pos;
    Integer n = columns();
    Real current_max;
    Real divisor = 0.0;
    index_list I(n);
    index_list J(n);
    vector_dense<Real> W(n);
    vector_dense<Integer> pos_dom_element(n);
    P.resize_with_constant_value(n,-1);
    Q.resize_with_constant_value(n,-1);
    Real rho,h;
    Integer nz;
    // preselection
    for(i=0;i<n;i++){
        current_max = W[i] = 0.0;
        J[i] = 0;
        for(j=pointer[i];j<pointer[i+1];j++){
            W[i] += fabs(data[j]);
            if (fabs(data[j])> current_max){
                current_max = fabs(data[j]);
                J[i] = indices[j];
                pos_dom_element[i] = j;
            }
        }
        divisor = W[i]*(pointer[i+1]-pointer[i]);
        //divisor = W[i];
        if(equal_to_zero(divisor)) W[i]=0.0;
        else W[i] = -current_max /divisor;
    }
    W.quicksort(I,0,n-1);
    J = J.permute(I); 
    count = -1;
    for (i=0;i<n;i++){
        if ((P[I[i]] == -1) && (Q[J[i]] == -1) && (-W[i] >= tau)) {
            rho=fabs(data[pos_dom_element[I[i]]]);
            nz = pointer[I[i]+1]-pointer[I[i]]-1;
            for (k=pointer[I[i]];k<pointer[I[i]+1];k++){
                if(Q[indices[k]]>=0 && k != pos_dom_element[I[i]]) { // indices[k] belongs to  SB
                    rho -= fabs(data[k]);
                    nz--;
                }
                if(Q[indices[k]]== -2 && k != pos_dom_element[I[i]]) { // indices[k] belongs to SF
                    nz--;
                }
            }
            if(rho<0) continue;
            count++;
            P[I[i]] = count;
            Q[J[i]] = count;
            for (k=pointer[I[i]];k<pointer[I[i]+1];k++){
                if(Q[indices[k]]== -1 && k != pos_dom_element[I[i]]){
                    h = fabs(data[ pos_dom_element[I[i]]]);
                    if(nz*h>rho) Q[indices[k]]= -2;
                    else rho -= h;
                }
                nz--;
            }
        }
    }
    // complete arbitarily
    pos = Qcount = count;
    for (i=0;i<n;i++){
        if(P[i]<0){
            count++;
            P[i] = count;
        }
    }
    for (i=0;i<n;i++){
        if(Q[i]<0){
            Qcount++;
            Q[i] = Qcount;
        }
    }
    return pos+1;
}


template<class T> Integer matrix_sparse<T>::symm_ddPQ_dyn_av(index_list& P, index_list& Q, Real tau) const {
    if(non_fatal_error(!square_check(),"matrix_sparse::symm_ddPQ_dyn_av: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#ifdef DEBUG
    if(non_fatal_error(!test_I_matrix(),"matrix_sparse::symm_ddPQ_dyn_av: argument must be an I-matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    Integer i,j,k,count,Qcount,pos;
    Integer n = columns();
    Real current_max;
    Real divisor = 0.0;
    index_list I(n);
    index_list J(n);
    vector_dense<Real> W(n);
    vector_dense<Integer> pos_dom_element(n);
    P.resize_with_constant_value(n,-1);
    Q.resize_with_constant_value(n,-1);
    Real rho,h;
    Integer nz;
    // preselection
    for(i=0;i<n;i++){
        current_max = W[i] = 0.0;
        J[i] = 0;
        for(j=pointer[i];j<pointer[i+1];j++){
            W[i] += fabs(data[j]);
            if (indices[j]==i){
                J[i] = indices[j];
                pos_dom_element[i] = j;
            }
        }
        divisor = W[i]*(pointer[i+1]-pointer[i]);
        //divisor = W[i];
        if(equal_to_zero(divisor)) W[i]=0.0;
        else W[i] = -current_max /divisor;
    }
    W.quicksort(I,0,n-1);
    J = J.permute(I); 
    count = -1;
    for (i=0;i<n;i++){
        if ((P[I[i]] == -1) && (Q[J[i]] == -1) && (-W[i] >= tau)) {
            rho=fabs(data[pos_dom_element[I[i]]]);
            nz = pointer[I[i]+1]-pointer[I[i]]-1;
            for (k=pointer[I[i]];k<pointer[I[i]+1];k++){
                if(Q[indices[k]]>=0 && k != pos_dom_element[I[i]]) { // indices[k] belongs to  SB
                    rho -= fabs(data[k]);
                    nz--;
                }
                if(Q[indices[k]]== -2 && k != pos_dom_element[I[i]]) { // indices[k] belongs to SF
                    nz--;
                }
            }
            if(rho<0) continue;
            count++;
            P[I[i]] = count;
            Q[J[i]] = count;
            for (k=pointer[I[i]];k<pointer[I[i]+1];k++){
                if(Q[indices[k]]== -1 && k != pos_dom_element[I[i]]){
                    h = fabs(data[ pos_dom_element[I[i]]]);
                    if(nz*h>rho) Q[indices[k]]= -2;
                    else rho -= h;
                }
                nz--;
            }
        }
    }
    // complete arbitarily
    pos = Qcount = count;
    for (i=0;i<n;i++){
        if(P[i]<0){
            count++;
            P[i] = count;
        }
    }
    for (i=0;i<n;i++){
        if(Q[i]<0){
            Qcount++;
            Q[i] = Qcount;
        }
    }
    return pos+1;
}



template<class T> Integer matrix_sparse<T>::ddPQ(index_list& P, index_list& Q, Integer from, Integer to, Real tau) const {
    // selects only from "from" to "to", including "from", excluding "to"
    if(non_fatal_error(!square_check(),"matrix_sparse::ddPQ: argument matrix must be square.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer j,k,nz,count,Qcount;
    Integer n = columns();
    Real current_max;
    Real t;
    Integer pos;
    index_list I(n);
    index_list J(n);
    vector_dense<Real> W(n);
    P.resize_with_constant_value(n,-1);
    Q.resize_with_constant_value(n,-1);
    for(k=from;k<to;k++){
        current_max = W[k] = 0.0;
        J[k] = 0;
        nz = 0;
        for(j=pointer[k];j<pointer[k+1];j++){
            if(indices[j]>=from && indices[j]<to){
                nz++;
                W[k] += fabs(data[j]);
                if (fabs(data[j])> current_max){
                    current_max = fabs(data[j]);
                    J[k] = indices[j];
                }
            } // end if
        } // end for j
        t = W[k]*nz;
        // t = W[k];
        if (equal_to_zero(t))  W[k] = 0.0;
        else  W[k] = -current_max/t;
    }
    W.quicksort(I,from,to-1);
    J = J.permute(I); 
    count = from-1;
    for (k=from;k<to;k++){
        if ((P[I[k]] == -1) && (Q[J[k]] == -1) && (W[k] >= tau)) {
            count++;
            P[I[k]] = count;
            Q[J[k]] = count;
        }
    }
    pos = Qcount = count;
    for (k=from;k<to;k++){
        if(P[k]<0){
            count++;
            P[k] = count;
        }
    }
    for (k=from;k<to;k++){
        if(Q[k]<0){
            Qcount++;
            Q[k] = Qcount;
        }
    }
    for(k=0;k<from;k++) P[k]=k;
    for(k=0;k<from;k++) Q[k]=k;
    for(k=to;k<n;k++) P[k]=k;
    for(k=to;k<n;k++) Q[k]=k;
    return pos+1; // returns first index which was not selected by weights, i.e. first non-treated index
}



template<class T> void matrix_sparse<T>::symmetric_move_to_corner(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    vector_dense<Real> w(n);
    for(i = 0; i < pointer_size-1; i++){
        for(j = pointer[i]; j<pointer[i+1]; j++){
            //if(indices[j] != i){
            w[i] += fabs(data[j]);
            w[indices[j]]+= fabs(data[j]);
            //}
        }
    }
    P.resize(n);
    P.init();
    w.quicksort(P,0,n-1);
}

template<class T> void matrix_sparse<T>::weighted_symmetric_move_to_corner(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::weighted_symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    vector_dense<Real> w(n);
    vector_dense<Real> counter(n);
    for(i = 0; i < pointer_size-1; i++){
        for(j = pointer[i]; j<pointer[i+1]; j++){
            //if(indices[j] != i){
            w[i] += fabs(data[j]);
            w[indices[j]]+= fabs(data[j]);
            counter[indices[j]] += 1.0;
            //}
        }
    }
    for (i=0;i<n;i++) w[i] *= counter[i]+pointer[i+1]-pointer[i];
    P.resize(n);
    P.init();
    w.quicksort(P,0,n-1);
}


template<class T> void matrix_sparse<T>::weighted2_symmetric_move_to_corner(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::weighted2_symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    vector_dense<Real> colsum(n);
    vector_dense<Real> w(n);
    vector_dense<Real> colcounter(n);
    for(i = 0; i < pointer_size-1; i++){
        for(j = pointer[i]; j<pointer[i+1]; j++){
            //if(indices[j] != i){
            w[i] += fabs(data[j]);
            colsum[indices[j]]+= fabs(data[j]);
            colcounter[indices[j]] += 1.0;
            //}
        }
    }
    for (i=0;i<n;i++) w[i] = w[i]*(pointer[i+1]-pointer[i]) + colsum[i]*colcounter[i];
    P.resize(n);
    P.init();
    w.quicksort(P,0,n-1);
}

// symmetric PQ (P=Q) for I-matrices
template<class T> void matrix_sparse<T>::sym_ddPQ(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symm_ddPQ: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    vector_dense<Real> w(n);
    for(i = 0; i < pointer_size-1; i++){
        for(j = pointer[i]; j<pointer[i+1]; j++){
            w[i] += fabs(data[j]);
        }
    }
    for (i=0;i<n;i++) w[i] *= pointer[i+1]-pointer[i];
    P.resize(n);
    P.init();
    w.quicksort(P,0,n-1);
}


template<class T> void matrix_sparse<T>::symmetric_move_to_corner_improved(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    sorted_vector w;
    vector_dense<bool> unused(n,true); // indicates which indices have been used
    w.resize(n); // sets all weights to 0
    P.resize(n);
    matrix_sparse<T> A;
    A.change_orientation_of_data(*this);
    for(i = 0; i < pointer_size-1; i++){
        P[i] = w.index_min();
        w.remove_min();
        unused[P[i]] = false;
        for(j = pointer[P[i]]; j<pointer[P[i]+1]; j++){
            if(unused[indices[j]]) w.add(indices[j],fabs(data[j]));
        }
        for(j = A.pointer[P[i]]; j<A.pointer[P[i]+1]; j++){
            if(unused[A.indices[j]]) w.add(A.indices[j],fabs(A.data[j]));
        }
    }
}


template<class T> void matrix_sparse<T>::diagonally_dominant_symmetric_move_to_corner_improved(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::diagonally_dominant_symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j,current_index,counter=0;
    bool acceptable_index;
    sorted_vector w;
    vector_dense<Integer> unused(n,0); // indicates which indices have been used 0 = unused; 1= used, -1 = rejected
    vector_dense<Real> row_gap(n,2.0);
    vector_dense<Real> col_gap(n,2.0);
    w.resize(n); // sets all weights to 0
    P.resize(n);
    matrix_sparse<T> A;
    A.change_orientation_of_data(*this);
    for(i = 0; i < n; i++){
        current_index = w.index_min();
        acceptable_index = (row_gap[current_index]>=0) && (col_gap[current_index]>=0);
        j=pointer[current_index];
        while(acceptable_index && j<pointer[current_index+1]){
            if (unused[indices[j]]==1) acceptable_index = acceptable_index && (fabs(data[j])<= col_gap[indices[j]]);
            j++;
        }
        j = A.pointer[current_index];
        while(acceptable_index && j<A.pointer[current_index+1]){
            if (unused[A.indices[j]]==1) acceptable_index = acceptable_index && (fabs(A.data[j])<= row_gap[A.indices[j]]);
            j++;
        }
        if(acceptable_index){
            P[counter] = current_index;
            w.remove_min();
            unused[current_index] = 1;
            for(j = pointer[current_index]; j<pointer[current_index+1]; j++){
                if(equal_to_zero(unused[indices[j]])) w.add(indices[j],fabs(data[j]));
                col_gap[indices[j]] -= fabs(data[j]);
                //if(current_index != indices[j]) col_gap[indices[j]] -= fabs(data[j]);
            }
            for(j = A.pointer[current_index]; j<A.pointer[current_index+1]; j++){
                if(equal_to_zero(unused[A.indices[j]])) w.add(A.indices[j],fabs(A.data[j]));
                row_gap[A.indices[j]] -= fabs(A.data[j]);
                //if(current_index != A.indices[j]) row_gap[A.indices[j]] -= fabs(A.data[j]);
            }
            counter++;
        } else {
            unused[current_index] = -1;
            w.remove_min();
        }
    }
    // now do the rejected indices
    w.resize(n); // also clears everything
    for(i=0;i<counter;i++) w.remove(P[i]);
    for(i=0;i<n;i++){
        if(unused[i] == -1){
            for(j=pointer[i];j<pointer[i+1];j++) if(unused[indices[j]]!=-1) w.add(i,fabs(data[j]));
            for(j=A.pointer[i];j<A.pointer[i+1];j++) if(unused[A.indices[j]]!=-1) w.add(i,fabs(A.data[j]));
        }
    }
    for(i=counter;i<n;i++){
        P[i] = w.index_min();
        w.remove_min();
        unused[P[i]] = 0;
        for(j = pointer[P[i]]; j<pointer[P[i]+1]; j++){
            if(unused[indices[j]]==-1) w.add(indices[j],fabs(data[j]));
        }
        for(j = A.pointer[P[i]]; j<A.pointer[P[i]+1]; j++){
            if(unused[A.indices[j]]==-1) w.add(A.indices[j],fabs(A.data[j]));
        }
    }
}

template<class T> void matrix_sparse<T>::weighted_symmetric_move_to_corner_improved(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    sorted_vector w;
    vector_dense<Real> counter(n);
    vector_dense<bool> unused(n,true); // indicates which indices have been used
    w.resize(n); // sets all weights to 0
    P.resize(n);
    matrix_sparse<T> A;
    A.change_orientation_of_data(*this);
    for(i=0;i<n;i++) counter[i]=pointer[i+1]-pointer[i]+A.pointer[i+1]-A.pointer[i];
    for(i = 0; i < pointer_size-1; i++){
        P[i] = w.index_min();
        w.remove_min();
        unused[P[i]] = false;
        for(j = pointer[P[i]]; j<pointer[P[i]+1]; j++){
            if(unused[indices[j]]) w.add(indices[j],counter[indices[j]]*fabs(data[j]));
        }
        for(j = A.pointer[P[i]]; j<A.pointer[P[i]+1]; j++){
            if(unused[A.indices[j]]) w.add(A.indices[j],counter[indices[j]]*fabs(A.data[j]));
        }
    }
}


template<class T> void matrix_sparse<T>::weighted2_symmetric_move_to_corner_improved(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    sorted_vector w;
    vector_dense<Real> counterrows(n);
    vector_dense<Real> countercols(n);
    vector_dense<bool> unused(n,true); // indicates which indices have been used
    w.resize(n); // sets all weights to 0
    P.resize(n);
    matrix_sparse<T> A;
    A.change_orientation_of_data(*this);
    for(i=0;i<n;i++) counterrows[i]=pointer[i+1]-pointer[i];
    for(i=0;i<n;i++) countercols[i]=A.pointer[i+1]-A.pointer[i];
    for(i = 0; i < pointer_size-1; i++){
        P[i] = w.index_min();
        w.remove_min();
        unused[P[i]] = false;
        for(j = pointer[P[i]]; j<pointer[P[i]+1]; j++){
            if(unused[indices[j]]) w.add(indices[j],countercols[indices[j]]*fabs(data[j]));
        }
        for(j = A.pointer[P[i]]; j<A.pointer[P[i]+1]; j++){
            if(unused[A.indices[j]]) w.add(A.indices[j],counterrows[indices[j]]*fabs(A.data[j]));
        }
    }
}

template<class T> void matrix_sparse<T>::sp_symmetric_move_to_corner_improved(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j,posA;
    sorted_vector w;
    vector_dense<bool> unused(n,true); // indicates which indices have been used
    w.resize(n); // sets all weights to 0
    P.resize(n);
    matrix_sparse<T> A;
    A.change_orientation_of_data(*this);
    for(i = 0; i < pointer_size-1; i++){
        P[i] = w.index_min();
        w.remove_min();
        unused[P[i]] = false;
        posA = A.pointer[P[i]];
        for(j = pointer[P[i]]; j<pointer[P[i]+1]; j++){
            while(A.indices[posA] < indices[j] && posA < A.pointer[P[i]+1]) posA++;
            //if ( unused[indices[j]] &&  (A.indices[posA] == indices[j] && data[j]*A.data[posA] > 0.0) )   w.add(indices[j],data[j]*A.data[posA]);
            if ( unused[indices[j]] &&  (A.indices[posA] == indices[j]) )   w.add(indices[j],fabs(data[j])*fabs(A.data[posA]));
        }
    }
}


template<class T> void matrix_sparse<T>::symb_symmetric_move_to_corner_improved(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    sorted_vector w;
    vector_dense<bool> unused(n,true); // indicates which indices have been used
    w.resize(n); // sets all weights to 0
    P.resize(n);
    matrix_sparse<T> A;
    A.change_orientation_of_data(*this);
    for(i = 0; i < pointer_size-1; i++){
        P[i] = w.index_min();
        w.remove_min();
        unused[P[i]] = false;
        for(j = pointer[P[i]]; j<pointer[P[i]+1]; j++){
            if(unused[indices[j]]) w.add(indices[j],1.0);
        }
        for(j = A.pointer[P[i]]; j<A.pointer[P[i]+1]; j++){
            if(unused[A.indices[j]]) w.add(A.indices[j],1.0);
        }
    }
}


template<class T> void matrix_sparse<T>::sp_symmetric_move_to_corner(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::sp_symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer j,k,h;
    vector_dense<Real> w(n);
    vector_sparse_dynamic<T> x(n);
    vector_sparse_dynamic<T> y(n);
    array<Integer> firstA(n);
    array<Integer> listA(n);
    array<Integer> headA(n);
    initialize_sparse_matrix_fields(n,pointer,indices,listA,headA,firstA);
    for(k=0;k<n;k++){
        //for(j=pointer[k];j<pointer[k+1];j++) x[indices[j]] = fabs(data[j]);
        //for(j=pointer[k];j<pointer[k+1];j++) x[indices[j]] = data[j];
        for(j=pointer[k];j<pointer[k+1];j++) if (data[j] != 0.0) x[indices[j]] = 1.0;
        h=headA[k];
        while(h!=-1){
            //y[h]=data[firstA[h]];
            //y[h]=fabs(data[firstA[h]]);
            if (data[firstA[h]] != (T) 0) y[h] = 1.0;
            h=listA[h];
        }
        update_sparse_matrix_fields(k,pointer,indices,listA,headA,firstA);
        // not very good without absolute value in vectors //w[k] = (x*y); // scalar product (negative elements are good and should be eliminated first, because they increase the pivot)
        // not very good // w[k] = x.scalar_product_pos_factors(y); // scalar product (positive factors are bad)
        w[k] = std::real(x*y); // scalar product (negative elements are good and should be eliminated first, because they increase the pivot)
        x.zero_reset();
        y.zero_reset();
    } 
    P.resize(n);
    P.init();
    w.quicksort(P,0,n-1);
}

template<class T> void matrix_sparse<T>::unit_or_zero_diagonal(vector_dense<T>& D1) const {
    D1.resize(dim_along_orientation(),1.0);
    for(Integer k=0; k<pointer_size-1; k++)
        for(Integer j=pointer[k]; j<pointer[k+1]; j++)
            if(indices[j]==k && data[j] != (T) 0.0) D1[k]= data[j];   // data[j]/fabs(data[j]);

}

/*
template<class T> void matrix_sparse<T>::non_negative_diagonal(vector_dense<T>& D1) const {
  try {
      D1.resize(dim_along_orientation(),1.0);
      for(Integer k=0; k<pointer_size-1; k++)
          for(Integer j=pointer[k]; j<pointer[k+1]; j++)
              if(indices[j]==k && data[j]<0) D1[k]=-1.0;

  }
  catch(iluplusplus_error ippe){
    std::cerr<<"matrix_sparse<T>::non_negative_diagonal: "<<ippe.error_message()<<std::endl;
    throw;
  }
}
*/

template<class T> void matrix_sparse<T>::symb_symmetric_move_to_corner(index_list& P) const {
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::symb_symmetric_move_to_corner: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n = rows();
    Integer i,j;
    vector_dense<Real> w(n);
    for(i = 0; i < pointer_size-1; i++){
        for(j = pointer[i]; j<pointer[i+1]; j++){
            if(indices[j] != i){
                w[i] += 1.0;
                w[indices[j]]+= 1.0;
            }
        }
    }
    P.resize(n);
    P.init();
    w.quicksort(P,0,n-1);
}

template<class T> Integer matrix_sparse<T>::preprocess(
        const matrix_sparse<T>& A, const iluplusplus_precond_parameter& IP, index_list& P, index_list& Q,
        index_list& invP, index_list& invQ, vector_dense<T>& Drow, vector_dense<T>& Dcol){
    Integer k;
    vector_dense<T> D1,D2;
    index_list p1,p2,ip1,ip2;
    if(non_fatal_error(A.rows()!=A.columns(),"matrix_sparse::preprocess: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n=A.rows();
    Integer preprocessing_bad_at = n;
    P.resize(n);
    Q.resize(n);
    invP.resize(n);
    invQ.resize(n);
    Drow.resize(n,1.0);
    Dcol.resize(n,1.0);
    if(IP.get_PREPROCESSING().dimension()==0){
        *this = A;
        return n;
    } else {
        switch (IP.get_PREPROCESSING()[0]){
            case TEST_ORDERING:
                A.test_ordering(p1,p2);
                permute(A,p1,p2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  NORMALIZE_COLUMNS:
                normalize_columns(A,D2);
                D2.permute(invQ);
                Dcol.multiply(D2);
                preprocessing_bad_at = n;
                break;
            case   NORMALIZE_ROWS:
                normalize_rows(A,D1);
                D1.permute(invP);
                Drow.multiply(D1);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_SPARSPAK
            case  REVERSE_CUTHILL_MCKEE_ORDERING:
                A.rcm(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
#endif
            case  PQ_ORDERING:
                preprocessing_bad_at = A.choose_ddPQ(IP,ip1,ip2);
                p1.invert(ip1);
                p2.invert(ip2);
                permute(A,p1,p2,ip1,ip2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                //preprocessing_bad_at = n;
                break;
            case  MAX_WEIGHTED_MATCHING_ORDERING:
                A.maximal_weight_inverse_scales(p1,D1,D2);
                // preprocess matrix
                inverse_scale(A,D1,ROW);
                inverse_scale(D2,COLUMN);
                permute(p1,ROW);
                // incorporate left scaling
                D1.permute(invP);
                Drow.multiply(D1);
                // incorporate right scaling
                D2.permute(invQ);
                Dcol.multiply(D2);
                // incorporate permutation
                P.compose_right(p1);
                invP.invert(P);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_METIS
            case  METIS_NODE_ND_ORDERING:
                A.metis_node_nd(p1,ip1);
                permute(A,p1,p1,ip1,ip1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
#endif
            case UNIT_OR_ZERO_DIAGONAL_SCALING:
                A.unit_or_zero_diagonal(D1);
                inverse_scale(A,D1,ROW);
                Drow.multiply(D1);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_PARDISO
            case  PARDISO_MAX_WEIGHTED_MATCHING_ORDERING:
                A.pardiso_maximal_weight_inverse_scales(p1,D1,D2);
                // preprocess matrix
                inverse_scale(A,D1,ROW);
                inverse_scale(D2,COLUMN);
                permute(p1,ROW);
                // incorporate left scaling
                D1.permute(invP);
                Drow.multiply(D1);
                // incorporate right scaling
                D2.permute(invQ);
                Dcol.multiply(D2);
                // incorporate permutation
                P.compose_right(p1);
                invP.invert(P);
                preprocessing_bad_at = n;
                break;
#endif
                break;
            case  SPARSE_FIRST_ORDERING:
                A.sparse_first_ordering(p2);
                permute(A,p2,COLUMN);
                Q.compose_right(p2);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  DYN_AV_PQ_ORDERING:
                preprocessing_bad_at = A.ddPQ_dyn_av(ip1,ip2,IP.get_PQ_THRESHOLD());
                p1.invert(ip1);
                p2.invert(ip2);
                permute(A,p1,p2,ip1,ip2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                break;
            case  SYMM_MOVE_CORNER_ORDERING:
                A.symmetric_move_to_corner(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMM_PQ:
                A.sym_ddPQ(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMM_MOVE_CORNER_ORDERING_IM:
                A.symmetric_move_to_corner_improved(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMB_SYMM_MOVE_CORNER_ORDERING:
                A.symb_symmetric_move_to_corner(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMB_SYMM_MOVE_CORNER_ORDERING_IM:
                A.symb_symmetric_move_to_corner_improved(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SP_SYMM_MOVE_CORNER_ORDERING:
                A.sp_symmetric_move_to_corner(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SP_SYMM_MOVE_CORNER_ORDERING_IM:
                A.sp_symmetric_move_to_corner_improved(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT_SYMM_MOVE_CORNER_ORDERING:
                A.weighted_symmetric_move_to_corner(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT_SYMM_MOVE_CORNER_ORDERING_IM:
                A.weighted_symmetric_move_to_corner_improved(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT2_SYMM_MOVE_CORNER_ORDERING_IM:
                A.weighted2_symmetric_move_to_corner_improved(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT2_SYMM_MOVE_CORNER_ORDERING:
                A.weighted2_symmetric_move_to_corner(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case DD_SYMM_MOVE_CORNER_ORDERING_IM:
                A.diagonally_dominant_symmetric_move_to_corner_improved(p1);
                permute(A,p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            default:
                std::cerr<<"matrix_sparse<T>::preprocess: unknown error."<<std::endl;
                break;
        }   // end switch
    }  // end else
    for(k=1;k<IP.get_PREPROCESSING().dimension();k++){
        switch (IP.get_PREPROCESSING()[k]){
            case TEST_ORDERING:
                test_ordering(p1,p2);
                permute(p1,p2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  NORMALIZE_COLUMNS:
                normalize_columns(D2);
                D2.permute(invQ);
                Dcol.multiply(D2);
                preprocessing_bad_at = n;
                break;
            case   NORMALIZE_ROWS:
                normalize_rows(D1);
                D1.permute(invP);
                Drow.multiply(D1);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_SPARSPAK
            case  REVERSE_CUTHILL_MCKEE_ORDERING:
                rcm(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
#endif
            case  PQ_ORDERING:
                preprocessing_bad_at = choose_ddPQ(IP,ip1,ip2);
                p1.invert(ip1);
                p2.invert(ip2);
                permute(p1,p2,ip1,ip2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                //preprocessing_bad_at = n;
                break;
            case  MAX_WEIGHTED_MATCHING_ORDERING:
                maximal_weight_inverse_scales(p1,D1,D2);
                // preprocess matrix
                inverse_scale(D1,ROW);
                inverse_scale(D2,COLUMN);
                permute(p1,ROW);
                // incorporate left scaling
                D1.permute(invP);
                Drow.multiply(D1);
                // incorporate right scaling
                D2.permute(invQ);
                Dcol.multiply(D2);
                // incorporate permutation
                P.compose_right(p1);
                invP.invert(P);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_METIS
            case  METIS_NODE_ND_ORDERING:
                metis_node_nd(p1,ip1);
                permute(p1,p1,ip1,ip1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
#endif
            case UNIT_OR_ZERO_DIAGONAL_SCALING:
                unit_or_zero_diagonal(D1);
                inverse_scale(D1,ROW);
                Drow.multiply(D1);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_PARDISO
            case  PARDISO_MAX_WEIGHTED_MATCHING_ORDERING:
                pardiso_maximal_weight_inverse_scales(p1,D1,D2);
                // preprocess matrix
                inverse_scale(D1,ROW);
                inverse_scale(D2,COLUMN);
                permute(p1,ROW);
                // incorporate left scaling
                D1.permute(invP);
                Drow.multiply(D1);
                // incorporate right scaling
                D2.permute(invQ);
                Dcol.multiply(D2);
                // incorporate permutation
                P.compose_right(p1);
                invP.invert(P);
                preprocessing_bad_at = n;
                break;
#endif
                break;
            case  SPARSE_FIRST_ORDERING:
                sparse_first_ordering(p2);
                permute(p2,COLUMN);
                Q.compose_right(p2);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  DYN_AV_PQ_ORDERING:
                preprocessing_bad_at = ddPQ_dyn_av(ip1,ip2,IP.get_PQ_THRESHOLD());
                p1.invert(ip1);
                p2.invert(ip2);
                permute(p1,p2,ip1,ip2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                break;
            case  SYMM_MOVE_CORNER_ORDERING:
                symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMM_PQ:
                sym_ddPQ(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMM_MOVE_CORNER_ORDERING_IM:
                symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMB_SYMM_MOVE_CORNER_ORDERING:
                symb_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMB_SYMM_MOVE_CORNER_ORDERING_IM:
                symb_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SP_SYMM_MOVE_CORNER_ORDERING:
                sp_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SP_SYMM_MOVE_CORNER_ORDERING_IM:
                sp_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT_SYMM_MOVE_CORNER_ORDERING:
                weighted_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT_SYMM_MOVE_CORNER_ORDERING_IM:
                weighted_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT2_SYMM_MOVE_CORNER_ORDERING_IM:
                weighted2_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT2_SYMM_MOVE_CORNER_ORDERING:
                weighted2_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case DD_SYMM_MOVE_CORNER_ORDERING_IM:
                diagonally_dominant_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            default:
                std::cerr<<"matrix_sparse<T>::preprocess: unknown error."<<std::endl;
                break;
        }   // end switch
    }  // end for k
    return preprocessing_bad_at;
}

template<class T> Integer matrix_sparse<T>::preprocess(
        const iluplusplus_precond_parameter& IP, index_list& P, index_list& Q,
        index_list& invP, index_list& invQ, vector_dense<T>& Drow, vector_dense<T>& Dcol){
    Integer k;
    vector_dense<T> D1,D2;
    index_list p1,p2,ip1,ip2;
    if(non_fatal_error(rows()!=columns(),"matrix_sparse::preprocess: this routine requires a square matrix!")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n=rows();
    Integer preprocessing_bad_at = n;
    P.resize(n);
    Q.resize(n);
    invP.resize(n);
    invQ.resize(n);
    Drow.resize(n,1.0);
    Dcol.resize(n,1.0);
    for(k=0;k<IP.get_PREPROCESSING().dimension();k++){
        switch (IP.get_PREPROCESSING()[k]){
            case TEST_ORDERING:
                test_ordering(p1,p2);
                permute(p1,p2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  NORMALIZE_COLUMNS:
                normalize_columns(D2);
                D2.permute(invQ);
                Dcol.multiply(D2);
                preprocessing_bad_at = n;
                break;
            case   NORMALIZE_ROWS:
                normalize_rows(D1);
                D1.permute(invP);
                Drow.multiply(D1);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_SPARSPAK
            case  REVERSE_CUTHILL_MCKEE_ORDERING:
                rcm(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
#endif
            case  PQ_ORDERING:
                preprocessing_bad_at = choose_ddPQ(IP,ip1,ip2);
                p1.invert(ip1);
                p2.invert(ip2);
                permute(p1,p2,ip1,ip2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                //preprocessing_bad_at = n;
                break;
            case  MAX_WEIGHTED_MATCHING_ORDERING:
                maximal_weight_inverse_scales(p1,D1,D2);
                // preprocess matrix
                inverse_scale(D1,ROW);
                inverse_scale(D2,COLUMN);
                permute(p1,ROW);
                // incorporate left scaling
                D1.permute(invP);
                Drow.multiply(D1);
                // incorporate right scaling
                D2.permute(invQ);
                Dcol.multiply(D2);
                // incorporate permutation
                P.compose_right(p1);
                invP.invert(P);
                preprocessing_bad_at = n;
                break;
            case  SPARSE_FIRST_ORDERING:
                sparse_first_ordering(p2);
                permute(p2,COLUMN);
                Q.compose_right(p2);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_METIS
            case  METIS_NODE_ND_ORDERING:
                metis_node_nd(p1,ip1);
                permute(p1,p1,ip1,ip1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
#endif
            case UNIT_OR_ZERO_DIAGONAL_SCALING:
                unit_or_zero_diagonal(D1);
                inverse_scale(D1,ROW);
                Drow.multiply(D1);
                preprocessing_bad_at = n;
                break;
#ifdef ILUPLUSPLUS_USES_PARDISO
            case  PARDISO_MAX_WEIGHTED_MATCHING_ORDERING:
                pardiso_maximal_weight_inverse_scales(p1,D1,D2);
                // preprocess matrix
                inverse_scale(D1,ROW);
                inverse_scale(D2,COLUMN);
                permute(p1,ROW);
                // incorporate left scaling
                D1.permute(invP);
                Drow.multiply(D1);
                // incorporate right scaling
                D2.permute(invQ);
                Dcol.multiply(D2);
                // incorporate permutation
                P.compose_right(p1);
                invP.invert(P);
                preprocessing_bad_at = n;
                break;
#endif
            case  DYN_AV_PQ_ORDERING:
                preprocessing_bad_at = ddPQ_dyn_av(ip1,ip2,IP.get_PQ_THRESHOLD());
                p1.invert(ip1);
                p2.invert(ip2);
                permute(p1,p2,ip1,ip2);
                P.compose_right(p1);
                Q.compose_right(p2);
                invP.invert(P);
                invQ.invert(Q);
                break;
            case  SYMM_MOVE_CORNER_ORDERING:
                symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMM_PQ:
                sym_ddPQ(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMM_MOVE_CORNER_ORDERING_IM:
                symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMB_SYMM_MOVE_CORNER_ORDERING:
                symb_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SYMB_SYMM_MOVE_CORNER_ORDERING_IM:
                symb_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SP_SYMM_MOVE_CORNER_ORDERING:
                sp_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case  SP_SYMM_MOVE_CORNER_ORDERING_IM:
                sp_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT_SYMM_MOVE_CORNER_ORDERING:
                weighted_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT_SYMM_MOVE_CORNER_ORDERING_IM:
                weighted_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT2_SYMM_MOVE_CORNER_ORDERING_IM:
                weighted2_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case WGT2_SYMM_MOVE_CORNER_ORDERING:
                weighted2_symmetric_move_to_corner(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            case DD_SYMM_MOVE_CORNER_ORDERING_IM:
                diagonally_dominant_symmetric_move_to_corner_improved(p1);
                permute(p1,p1);
                P.compose_right(p1);
                Q.compose_right(p1);
                invP.invert(P);
                invQ.invert(Q);
                preprocessing_bad_at = n;
                break;
            default:
                std::cerr<<"matrix_sparse<T>::preprocess: unknown error."<<std::endl;
                break;
        }   // end switch
    }  // end for k
    return preprocessing_bad_at;
}


template<class T> void matrix_sparse<T>::permute(const matrix_sparse<T>& A, const index_list& perm)  {
    reformat(A.rows(),A.columns(),A.non_zeroes(),A.orient());
    Integer i,j,counter;
    counter = 0;
    for (i=0; i<dim_along_orientation(); i++){
        pointer[i] = counter;
        for(j=A.pointer[perm[i]]; j<A.pointer[perm[i]+1]; j++){
            data[counter] = A.data[j];
            indices[counter] = A.indices[j];
            counter++;
        }
    }
    pointer[pointer_size-1]=counter;
#ifdef DEBUG
    if(non_fatal_error(A.pointer[A.pointer_size-1] != counter,"matrix_sparse::permute: something is really strange.")) throw iluplusplus_error(UNKNOWN_ERROR);
    check_consistency();
#endif
}


template<class T> void matrix_sparse<T>::permute_against_orientation(const matrix_sparse<T>& A, const index_list& perm)  {
    index_list invperm;
    invperm.invert(perm);
    permute_against_orientation_with_invperm(A,invperm);
}

template<class T> void matrix_sparse<T>::permute_against_orientation_with_invperm(const matrix_sparse<T>& A, const index_list& invperm)  {
    reformat(A.rows(),A.columns(),A.non_zeroes(),A.orient());
    Integer i,j;
    for (i=0; i<dim_along_orientation(); i++){
        for(j=A.pointer[i]; j<A.pointer[i+1]; j++){
            indices[j] = invperm[A.indices[j]];
            data[j]= A.data[j];
        }
        pointer[i]=A.pointer[i];
    }
    pointer[dim_along_orientation()]=A.pointer[dim_along_orientation()];
    normal_order();
#ifdef DEBUG
    check_consistency();
#endif

}

template<class T> void matrix_sparse<T>::permute_along_and_against_orientation(const matrix_sparse<T>& A, const index_list& perm_along, const index_list& perm_against){
    index_list invperm_against;
    invperm_against.invert(perm_against);
    permute_along_with_perm_and_against_orientation_with_invperm(A,perm_along,invperm_against);
}


template<class T> void matrix_sparse<T>::permute_along_with_perm_and_against_orientation_with_invperm(const matrix_sparse<T>& A, const index_list& perm_along, const index_list& invperm_against){
    reformat(A.rows(),A.columns(),A.non_zeroes(),A.orient());
    Integer i,j,counter;
    counter = 0;
    for (i=0; i<dim_along_orientation(); i++){
        pointer[i] = counter;
        for(j=A.pointer[perm_along[i]]; j<A.pointer[perm_along[i]+1]; j++){
            data[counter] = A.data[j];
            indices[counter] = invperm_against[A.indices[j]];
            counter++;
        }
    }
    pointer[pointer_size-1]=counter;
    normal_order();
#ifdef DEBUG
    if(A.pointer[A.pointer_size-1] != counter){
        std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: something is really strange."<<std::endl;
        if(perm_along.check_if_permutation()) std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: first index_list is a permutation of size "<<perm_along.dim()<<std::endl;
        else std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: first index_list is NOT a permutation of size "<<perm_along.dim()<<std::endl;
        if(invperm_against.check_if_permutation()) std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: second index_list is a permutation of size "<<invperm_against.dim()<<std::endl;
        else std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: second index_list is NOT a permutation of size "<<invperm_against.dim()<<std::endl;
        if(check_consistency()) std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: matrix is consistent."<<std::endl;
        else std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: matrix is NOT consistent."<<std::endl;
        throw iluplusplus_error(UNKNOWN_ERROR);
    }
#endif
}


template<class T> void matrix_sparse<T>::permute(const matrix_sparse<T>& A, const index_list& perm, orientation_type O){
    if(O==A.orient()) permute(A,perm);
    else permute_against_orientation(A,perm);
}


template<class T> void matrix_sparse<T>::permute(const index_list& perm, orientation_type O){
    matrix_sparse<T> H;
    H.permute(*this,perm,O);
    interchange(H);
}

template<class T> void matrix_sparse<T>::permute(const matrix_sparse<T>& A, const index_list& permP, const index_list& permQ){
    index_list inverse;
    if(A.orientation == ROW){
        inverse.invert(permQ);
        permute_along_with_perm_and_against_orientation_with_invperm(A,permP,inverse);
    } else {
        inverse.invert(permP),
            permute_along_with_perm_and_against_orientation_with_invperm(A,permQ,inverse);
    }
}


template<class T> void matrix_sparse<T>::permute(const index_list& permP, const index_list& permQ){
    matrix_sparse<T> H;
    H.permute(*this,permP,permQ);
    copy_and_destroy(H);
}

template<class T> void matrix_sparse<T>::permute(const matrix_sparse<T>& A, const index_list& permP, const index_list& permQ, const index_list& invpermP, const index_list& invpermQ){
    if(A.orientation == ROW) permute_along_with_perm_and_against_orientation_with_invperm(A,permP,invpermQ);
    else permute_along_with_perm_and_against_orientation_with_invperm(A,permQ,invpermP);
}



template<class T> void matrix_sparse<T>::permute_efficiently(matrix_sparse<T>& H, const index_list& permP, const index_list& permQ, const index_list& invpermP, const index_list& invpermQ){
    H.permute(*this,permP,permQ,invpermP,invpermQ);
    interchange(H);
}

template<class T> void matrix_sparse<T>::permute(const index_list& permP, const index_list& permQ, const index_list& invpermP, const index_list& invpermQ){
    matrix_sparse<T> H;
    H.permute(*this,permP,permQ,invpermP,invpermQ);
    copy_and_destroy(H);
}


template<class T> void matrix_sparse<T>::permute_efficiently(matrix_sparse<T>& H, const index_list& permP, const index_list& permQ){
    H.permute(*this,permP,permQ);
    interchange(H);
}



#ifdef ILUPLUSPLUS_USES_METIS
template<class T> void matrix_sparse<T>::metis_node_nd(index_list& P, index_list& invP) const {
    METIS_NODE_ND<T>(*this,P,invP);
}

template<class T> void matrix_sparse<T>::metis_node_nd(Integer* P, Integer* invP) const {
    METIS_NODE_ND<T>(*this,P,invP);
}
#endif


template<class T> Integer matrix_sparse<T>::ddPQ(matrix_sparse<T>& A, orientation_type PQorient, Real tau) {
    Integer pos;
    matrix_sparse<T> B;
    index_list permP, permQ, invpermP, invpermQ;
    if(PQorient == A.orientation){
        pos = A.ddPQ(invpermP,invpermQ,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        if (A.orientation == COLUMN) permute(A,permQ,permP);
        else permute(A,permP,permQ);
    } else {
        B.change_orientation_of_data(A);
        pos = B.ddPQ(invpermP,invpermQ,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        if (A.orientation == COLUMN) permute(B,permQ,permP);
        else permute(B,permP,permQ);
    }
    return pos;
}


template<class T> Integer matrix_sparse<T>::ddPQ(matrix_sparse<T>& A, orientation_type PQorient, Integer from, Integer to, Real tau) {
    Integer pos;
    matrix_sparse<T> B;
    index_list permP, permQ, invpermP, invpermQ;
    if(PQorient == A.orientation){
        pos = A.ddPQ(invpermP,invpermQ,from,to,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        if (A.orientation == COLUMN) permute(A,permQ,permP);
        else permute(A,permP,permQ);
    } else {
        B.change_orientation_of_data(A);
        pos = B.ddPQ(invpermP,invpermQ,from,to,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        permute(B,permP,permQ);
    }
    return pos;
}



template<class T> Integer matrix_sparse<T>::ddPQ(matrix_sparse<T>& A, const vector_dense<T>& bold, vector_dense<T>& bnew, orientation_type PQorient, Real tau){
    Integer pos;
    matrix_sparse<T> B;
    index_list permP, permQ, invpermP, invpermQ;
    if(PQorient == A.orientation){
        pos = A.ddPQ(invpermP,invpermQ,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        if (A.orientation == COLUMN){ 
            permute(A,permQ,permP);
            bnew.permute(bold,permQ);
        } else {
            permute(A,permP,permQ);
            bnew.permute(bold,permP);
        }
    } else {
        B.change_orientation_of_data(A);
        pos = B.ddPQ(invpermP,invpermQ,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        permute(B,permP,permQ);
        bnew.permute(bold,permP);
    }
    return pos;
}

template<class T> Integer matrix_sparse<T>::ddPQ(matrix_sparse<T>& A, const vector_dense<T>& bold, vector_dense<T>& bnew, orientation_type PQorient, Integer from, Integer to, Real tau){
    Integer pos;
    matrix_sparse<T> B;
    index_list permP, permQ, invpermP, invpermQ;
    if(PQorient == A.orientation){
        pos = A.ddPQ(invpermP,invpermQ,from,to,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        if (A.orientation == COLUMN){ 
            permute(A,permQ,permP);
            bnew.permute(bold,permQ);
        } else {
            permute(A,permP,permQ);
            bnew.permute(bold,permP);
        }
    } else {
        B.change_orientation_of_data(A);
        pos = B.ddPQ(invpermP,invpermQ,from,to,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        permute(B,permP,permQ);
        bnew.permute(bold,permP);
    }
    return pos;
}

template<class T> Integer matrix_sparse<T>::multilevel_PQ(matrix_sparse<T>& A, const vector_dense<T>& bold, vector_dense<T>& bnew, orientation_type PQorient, Integer& level, Real tau){
    const Integer maxlevel = 4;
    if (!A.square_check()){
        std::cerr<<"matrix_sparse::multilevel_PQ: matrix must be square. Doing nothing"<<std::endl;
        return -1;
    }
    Integer n = A.rows();
    Integer pos=0;
    Integer pos_new;
    level=1;
    vector_dense<T> b;
    pos_new = ddPQ(A,bold,bnew,PQorient,tau);
    b = bnew;
#ifdef VERBOSE
    std::cout<<"multilevel PQ: end of level 1 at index "<<pos_new<<std::endl;
#endif
    A = *this;
    while(pos != pos_new && pos_new<n-1){
        pos = pos_new;
        pos_new = ddPQ(A,b,bnew,PQorient,pos_new,n,tau);
        A = *this;
        b = bnew;
        if(level == maxlevel) return pos;
        level++;
#ifdef VERBOSE
        if(pos_new != pos) std::cout<<"multilevel PQ: end of level "<<level<<" at index "<<pos_new<<std::endl;
#endif
    }
    return pos;
}


template<class T> bool matrix_sparse<T>::maximal_weight_inverse_scales(index_list& P, vector_dense<T>& D1, vector_dense<T>& D2) const {
    index_list invP;
    return find_pmwm(*this, invP.vec(), P.vec(), D1, D2);
}

template<class T> void matrix_sparse<T>::test_ordering(index_list& P, index_list& Q) const {
    P.resize(dimension());
    column_perm(*this, Q.vec());
}

template<class T> void matrix_sparse<T>::sparse_first_ordering(index_list& Q) const {
    column_perm(*this, Q.vec());
}

#ifdef ILUPLUSPLUS_USES_PARDISO
template<class T> int matrix_sparse<T>::pardiso_maximal_weight(int* P, double* D1, double* D2) const {
    int neqns = read_pointer_size()-1 ;
    int transpose=0;
    int partial = 0;
    int fast = 0;
    int pardiso_return;
    // old: return mps_pardiso(partial,neqns,pointer,indices,data,P,D1,D2,transpose);
    pardiso_return = mps_pardiso(fast,partial,neqns,pointer,indices,data,P,D1,D2,transpose);
    return pardiso_return;
}

template<> int matrix_sparse<Complex>::pardiso_maximal_weight(int* P, double* D1, double* D2) const {
    Real* absdata = 0;
    int neqns = read_pointer_size()-1 ;
    int transpose=0;
    int partial = 0;
    int fast = 0;
    int pardiso_return;
    int nonzeroes = read_pointer(neqns);
    if(nonzeroes>0) absdata = new Real[nonzeroes];
    for(Integer i=0;i<nonzeroes;i++) absdata[i] = fabs(data[i]);
    // old: return mps_pardiso(partial,neqns,pointer,indices,data,P,D1,D2,transpose);
    pardiso_return = mps_pardiso(fast,partial,neqns,pointer,indices,absdata,P,D1,D2,transpose);
    delete [] absdata;
    absdata = 0;
    return pardiso_return;
}

template<class T> int matrix_sparse<T>::pardiso_maximal_weight(index_list& P, vector_dense<T>& D1, vector_dense<T>& D2) const {
    int *Perm = 0;
    double *scale1 = 0;
    double *scale2 = 0;
    int neqns = read_pointer_size()-1;
    int i,error;
    P.resize_without_initialization(neqns);
    D1.resize_without_initialization(neqns);
    D2.resize_without_initialization(neqns);
    if (neqns>0){
        Perm = new int[neqns];
        scale1 = new double[neqns];
        scale2 = new double[neqns];
    }
    error = pardiso_maximal_weight(Perm, scale1, scale2);
    for(i=0;i<neqns;i++) P[i]  = Perm[i];
    for(i=0;i<neqns;i++) D1[i] = std::exp(scale1[i]);
    for(i=0;i<neqns;i++) D2[i] = std::exp(scale2[i]);
    if (Perm !=0) delete [] Perm;
    if (scale1 !=0) delete [] scale1;
    if (scale1 !=0) delete [] scale2;
    return error;
}


template<class T> int matrix_sparse<T>::pardiso_maximal_weight_inverse_scales(index_list& P, vector_dense<T>& D1, vector_dense<T>& D2) const {
    int *Perm = 0;
    double *scale1 = 0;
    double *scale2 = 0;
    int neqns = read_pointer_size()-1;
    int i,error;
    P.resize_without_initialization(neqns);
    D1.resize_without_initialization(neqns);
    D2.resize_without_initialization(neqns);
    if (neqns>0){
        Perm = new int[neqns];
        scale1 = new double[neqns];
        scale2 = new double[neqns];
    }
    error = pardiso_maximal_weight(Perm, scale1, scale2);
    for(i=0;i<neqns;i++) P[i]  = Perm[i];
    for(i=0;i<neqns;i++) D1[i] = std::exp(-scale1[i]);
    for(i=0;i<neqns;i++) D2[i] = std::exp(-scale2[i]);
    if (Perm !=0) delete [] Perm;
    if (scale1 !=0) delete [] scale1;
    if (scale1 !=0) delete [] scale2;
    return error;
}

#endif // using pardiso


template<class T> bool matrix_sparse<T>::test_I_matrix() const {
    if(!square_check()){
        std::cout<<"Matrix is not square, hence no I-matrix."<<std::endl;
        return false;
    }
    bool Imatrix = true;
    bool found_diagonal_element;
    Real error = 0.0;
    for(Integer i=0;i<read_pointer_size()-1;i++){
        found_diagonal_element = false;
        for(Integer j=pointer[i]; j<pointer[i+1];j++){
            if ( (i !=read_index(j))&& ((fabs(read_data(j))-1.0)>COMPARE_EPS) ) {
                #ifdef VERBOSE
                    if(orient()==ROW) std::cout<<"("<<i<<","<<read_index(j)<<") has value "<<read_data(j)<<std::endl;
                    else  std::cout<<"("<<read_index(j)<<","<<i<<") has value "<<read_data(j)<<std::endl;
                #endif
                error += fabs(read_data(j))-1.0;
                Imatrix = false;
            }
            if (i==read_index(j)){
                found_diagonal_element = true;
                if (fabs(fabs(read_data(j))-1.0)>COMPARE_EPS){
                #ifdef VERBOSE
                    if(orient()==ROW) std::cout<<"("<<i<<","<<read_index(j)<<") has value "<<read_data(j)<<std::endl;
                    else  std::cout<<"("<<read_index(j)<<","<<i<<") has value "<<read_data(j)<<std::endl;
                #endif
                error += fabs(fabs(read_data(j))-1.0);
                Imatrix = false;
                }
            }
        }
        if(!found_diagonal_element){
            #ifdef VERBOSE
                std::cout<<"Zero diagonal element at position ("<<i<<","<<i<<")."<<std::endl;
            #endif
            Imatrix = false;
            error += 1.0;
        }
    }
    if(!Imatrix) std::cout<<"Not an I-matrix. Total deviation is: "<<error<<std::endl;
    return Imatrix;
}


template<class T> bool matrix_sparse<T>::test_normalized_I_matrix() const {
    if(!square_check()){
        std::cout<<"Matrix is not square, hence no I-matrix."<<std::endl;
        return false;
    }
    bool Imatrix = true;
    bool found_diagonal_element;
    for(Integer i=0;i<read_pointer_size()-1;i++){
        found_diagonal_element = false;
        for(Integer j=pointer[i]; j<pointer[i+1];j++){
            if ( (i !=read_index(j))&& ((fabs(read_data(j))-1.0)>COMPARE_EPS) ) {
                if(orient()==ROW) std::cout<<"("<<i<<","<<read_index(j)<<") has value "<<read_data(j)<<std::endl;
                else  std::cout<<"("<<read_index(j)<<","<<i<<") has value "<<read_data(j)<<std::endl;
                Imatrix = false;
            }
            if (i==read_index(j)){
                found_diagonal_element = true;
                if (fabs(read_data(j)-1.0)>COMPARE_EPS){
                    Imatrix = false;
                    if(orient()==ROW) std::cout<<"("<<i<<","<<read_index(j)<<") has value "<<read_data(j)<<std::endl;
                    else  std::cout<<"("<<read_index(j)<<","<<i<<") has value "<<read_data(j)<<std::endl;
                }
            }
        }
        if(!found_diagonal_element){
            std::cout<<"Zero diagonal element at position ("<<i<<","<<i<<")."<<std::endl;
            Imatrix = false;
        }
    }
    return Imatrix;
}


template<class T> Real matrix_sparse<T>::test_diag_dominance(Integer i) const {
    #ifdef DEBUG
        if(non_fatal_error(i<0||i>=pointer_size,"matrix_sparse::test_diag_dominance:: no such row or column")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    #endif
    Real offdiag = 0.0;
    Real diag = 0.0;
    for(Integer k=pointer[i];k<pointer[i+1];k++){
        if(indices[k] != i) offdiag += fabs(data[k]);
        else diag = fabs(data[k]);
    }
    return offdiag/diag;
}


template<class T> Real matrix_sparse<T>::test_diag_dominance() const {
     Real sum = 0.0;
     Real maximum = 0.0;
     Real minimum = 1e300;
     Real current;
     for(Integer i=0;i<pointer_size-1;i++){
         current = test_diag_dominance(i);
         sum += current;
         maximum = max(maximum,current);
         minimum = min(minimum,current);
     }
     std::cout<<"test_diag_dominance: "<<std::endl;
     std::cout<<"    worst-case: "<<maximum<<std::endl;
     std::cout<<"    average:    "<<sum/(pointer_size-1)<<std::endl;
     std::cout<<"    best-case: "<<minimum<<std::endl<<std::endl;
     return maximum;
  }

template<class T>  Real matrix_sparse<T>::memory() const{
    return (Real) ((sizeof(T)+sizeof(Integer))* (Real) nnz) + (sizeof(Integer)*(Real) pointer_size) +  4*sizeof(Integer);
}

//***********************************************************************************************************************
//                                                                                                                      *
//           The implementation of the class matrix_oriented                                                            *
//                                                                                                                      *
//***********************************************************************************************************************

//***********************************************************************************************************************
// Class matrix_oriented: private functions                                                                             *
//***********************************************************************************************************************

template<class T> void matrix_oriented<T>::insert_data_along_orientation(const vector_dense<T>& data_vector,Integer k){
    Integer dim_al_or = (Integer)( matrix_oriented<T>::dim_along_orientation());
    Integer offset=k*dim_al_or;
    if(k+1 >= dim_al_or){
        std::cerr<<"matrix_oriented<T>::insert_data_along_orientation: the dimension "<<k<<" is too large for this matrix."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    for(Integer i = 0; i < dim_al_or; i++){
        data[offset+i]=data_vector[i];
    }
  }


//***********************************************************************************************************************
// Class matrix_oriented: constructors, destructor                                                                      *
//***********************************************************************************************************************

template<class T> matrix_oriented<T>::matrix_oriented(orientation_type o){
     number_rows    = 0;
     number_columns = 0;
     size           = 0;
     orientation    = o;
     data           = 0;
}

template<class T> matrix_oriented<T>::matrix_oriented(orientation_type o, Integer m, Integer n){
    number_rows    = 0;
    number_columns = 0;
    size           = 0;
    orientation    = o;
    data           = 0;
    resize(o,m,n);
}

template<class T> matrix_oriented<T>::matrix_oriented(const matrix_oriented& X){
    number_rows    = 0;
    number_columns = 0;
    size           = 0;
    data           = 0;
    resize(X.orientation,X.number_rows,X.number_columns);
    Integer i;
    for (i=0;i<size;i++) data[i] = X.data[i];
}

template<class T> matrix_oriented<T>::~matrix_oriented() {
     if (data != 0) delete [] data; data=0;
  }

//***********************************************************************************************************************
// Class matrix_oriented: basic operators                                                                               *
//***********************************************************************************************************************

template<class T> matrix_oriented<T> matrix_oriented<T>::operator = (const matrix_oriented<T>& X){
    if(this==&X)
        return *this;
    resize(X.orientation,X.number_rows,X.number_columns);
    Integer i;
    for (i=0;i<matrix_oriented<T>::nnz;i++) data[i] = X.data[i];
    return *this;
}

//***********************************************************************************************************************
// Class matrix_oriented: basic information                                                                             *
//***********************************************************************************************************************

template<class T> Integer matrix_oriented<T>::rows() const {
     return number_rows;
  }

template<class T> Integer matrix_oriented<T>::columns() const{
     return number_columns;
  }

template<class T> T matrix_oriented<T>::get_data(Integer i) const{
     #ifdef DEBUG
         if(i<0 || i> size){
             std::cerr<<"matrix_oriented::get_data: index out of range."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     return data[i];
  }

template<class T> bool matrix_oriented<T>::square_check() const {
    return (number_rows == number_columns);
  }


template<class T> Integer matrix_oriented<T>::dim_along_orientation() const {
     if (orientation == ROW) return number_rows;
     else return number_columns;
  }

template<class T> Integer matrix_oriented<T>::dim_against_orientation() const {
     if (orientation == ROW) return number_columns;
     else return number_rows;
  }

template<class T> void matrix_oriented<T>::print_all() const {
    std::cout<<"A ("<<number_rows<<"x"<<number_columns<<") matrix containing "<<size<< " elements and having ";
    if (orientation == ROW) std::cout<<"ROW"; else std::cout<<"COLUMN";
    std::cout<<" orientation."<<std::endl<<"The elements are: "<<std::endl;
    for (Integer k=0; k<size; k++) std::cout<<data[k]<<" ";
    std::cout<<std::endl;
  }

//***********************************************************************************************************************
// Class matrix_oriented: basic functions                                                                               *
//***********************************************************************************************************************

template<class T> void matrix_oriented<T>::set_all(T d){
    for(Integer i = 0; i < size; i++) data[i]=d;
  }

template<class T> void matrix_oriented<T>::resize(orientation_type o, Integer m, Integer n){
    if(m<0) m=0;
    if(n<0) n=0;
    Integer newsize = ((Integer)(m))*((Integer)(n));
    if (size != newsize) {
        size = newsize;
        if (data    != 0) delete [] data;
        if (newsize == 0){
            data = 0;
        } else {
            data = new T[newsize];
        }
    }
    number_rows = m;
    number_columns = n;
    orientation = o;
}

//***********************************************************************************************************************
// Class matrix_oriented: accessing data                                                                                *
//***********************************************************************************************************************

template<class T> void matrix_oriented<T>::extract(const matrix_sparse<T> &A, Integer m, Integer n){
    if(non_fatal_error(n == 0,"matrix_oriented::extract: a positive number must be specified to be extracted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(m+n > A.dim_against_orientation(),"matrix_oriented::extract: the rows/columns to be extracted do not exist." )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if (A.orient() == ROW)
        resize(ROW, n, A.columns());
    else
        resize(COLUMN, A.rows(), n);
    Integer k,i;
    Integer offset;
    set_all(0.0);
    for(k = 0; k < n; k++){
        offset = ((Integer)(k))*((Integer)(dim_against_orientation()));
        for(i=A.get_pointer(k+m); i< A.get_pointer(k+m+1); i++){
            data[ offset + ((Integer)(A.get_index(i))) ]  = A.get_data(i);
        }
    }
}

//***********************************************************************************************************************
// Class matrix_oriented: input/output                                                                                  *
//***********************************************************************************************************************

template<class T> std::ostream& operator << (std::ostream& os, const matrix_oriented<T> & x){
     Integer i_data;
     os<<"The matrix has "<<x.rows()<<" rows and "<<x.columns()<<" columns."<<std::endl;
     if(x.orient() == ROW){
         for(Integer i_row=0;i_row<x.rows();i_row++){
             os<<"*** row: "<<i_row<<" ***"<<std::endl;
             for(i_data = i_row*x.columns(); i_data<(i_row+1)*x.columns();i_data++)
                  os<<" "<<x.get_data(i_data)<<" ";
             std::cout<<std::endl;
         }
     }
     else
         for(Integer i_column=0;i_column<x.columns();i_column++){
             os<<"*** column: "<<i_column<<" ***"<<std::endl;
             for(i_data=i_column*x.rows();i_data<(i_column+1)*x.rows();i_data++)
                 os<<x.get_data(i_data)<<" ";
             std::cout<<std::endl;
         }
     return os;
  }

//***********************************************************************************************************************
// Class matrix_oriented: other functions                                                                               *
//***********************************************************************************************************************



template<class T> Real matrix_oriented<T>::norm(Integer k) const{
    Real norm_squared = 0.0;
    Integer offset = ((Integer)(k)) * ( (Integer)(dim_against_orientation()) );
    for(Integer j = 0; j < dim_against_orientation(); j++){
        norm_squared += absvalue_squared(data[offset+j]);
    }
    return sqrt(norm_squared);
  }


template<class T>  Real matrix_oriented<T>::memory() const{
    return (Real) ((sizeof(T))* (Real) number_rows *(Real) number_columns) +  4*sizeof(Integer);
}


//***********************************************************************************************************************
//                                                                                                                      *
//           The implementation of the class matrix_dense                                                               *
//                                                                                                                      *
//***********************************************************************************************************************

//***********************************************************************************************************************
// Class matrix_dense: private functions                                                                                *
//***********************************************************************************************************************

template<class T> void matrix_dense<T>::generic_matrix_vector_multiplication_addition(const vector_dense<T>& x, vector_dense<T>& v) const {
     for(Integer i=0;i<number_columns;i++)
         for(Integer j=0;j<number_rows;j++) v.set_data(i) += data[i][j] * x.get_data(j);
  }

template<class T> void matrix_dense<T>::generic_matrix_transpose_vector_multiplication_addition(const vector_dense<T>& x, vector_dense<T>& v) const {
     for(Integer i=0;i<number_columns;i++)
         for(Integer j=0;j<number_rows;j++) v._set(j) += data[i][j] * x[i];
  }

template<class T> void matrix_dense<T>::generic_matrix_matrix_multiplication_addition(const matrix_dense<T>& A, const matrix_dense<T>& B) {
     for(Integer i=0;i<number_columns;i++)
         for(Integer j=0;j<number_rows;j++)
             for(Integer k=0; k<A.number_columns; k++) data[i][j] += A.data[i][k] * B.data[k][j];
  }

//***********************************************************************************************************************
// Class matrix_dense: constructors, destructors, etc.                                                                  *
//***********************************************************************************************************************


template<class T> matrix_dense<T>::matrix_dense(){
    number_columns = 0; number_rows = 0; data = 0;
  }

template<class T> matrix_dense<T>::matrix_dense(Integer m, Integer n){
    number_columns = 0; number_rows = 0; data = 0;
    resize(m,n);
}

template<class T> matrix_dense<T>::matrix_dense(Integer m, Integer n, T d){
    number_columns = 0; number_rows = 0; data = 0;
    Integer i,j;
    resize(m,n);
    for(i=0;i<m;i++)
        for(j=0; j<n; j++) {
            data[i][j]=0;
        }
    for(i=0; i<min(m,n); i++) data[i][i]=d;
}

template<class T> matrix_dense<T>::matrix_dense(const matrix_dense& X){
    number_columns = 0; number_rows = 0; data = 0;
    Integer i,j;
    resize(X.number_rows,X.number_columns);
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++) data[i][j]=X.data[i][j];;
}

template<class T> matrix_dense<T>::matrix_dense(const matrix_sparse<T> &A) {
    number_columns = 0; number_rows = 0; data = 0;
    resize(A.rows(),A.columns());
    Integer i,j;
    for(i=0;i<A.rows();i++)
        for(j=0; j<A.columns(); j++)
            data[i][j]=0;
    if (A.orient() == ROW){
        for(i=0;i<A.read_pointer_size()-1;i++){
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++){
                data[i][A.read_index(j)]+=A.read_data(j);
            }
        }
    } else {
        for(i=0;i<A.read_pointer_size()-1;i++)
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++)
                data[A.read_index(j)][i]+=A.read_data(j);
    }
}



template<class T> matrix_dense<T>::~matrix_dense() {
       if (data != 0){
           for(Integer i=0;i<number_rows;i++)
               if (data[i]!=0) delete[] data[i];
           delete[] data;
           data = 0;
       }
  }


template<class T>  bool matrix_dense<T>::square_check() const {
    return (columns() == rows());
  }

//***************************************************************************************************************************************
//  Class matrix_dense: basic functions                                                                                                 *
//***************************************************************************************************************************************

template<class T> void matrix_dense<T>::matrix_vector_multiplication_add(const vector_dense<T>& x, vector_dense<T>& v) const {
    if ((number_columns != x.dimension())||(number_rows != v.dimension())){
        std::cerr<<"matrix_dense:matrix_vector_multiplication_add(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else generic_matrix_vector_multiplication_addition(x,v);
}

template<class T> void matrix_dense<T>::matrix_vector_multiplication(const vector_dense<T>& x, vector_dense<T>& v) const {
    if (number_columns != x.dimension()){
        std::cerr << "matrix_dense:matrix_vector_multiplication(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else {
        v.resize(number_rows,0.0);
        generic_matrix_vector_multiplication_addition(x,v);
    }
}


template<class T> void matrix_dense<T>::matrix_matrix_multiplication_add(const matrix_dense<T>& A, const matrix_dense<T>& B) {
    if ((number_columns != B.number_columns)||(number_rows != A.number_rows) || (A.number_columns != B.number_rows)){
        std::cerr<<"matrix_dense:matrix_matrix_multiplication_add: Dimension error in matrix-matrix-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else generic_matrix_matrix_multiplication_addition(A,B);
}

template<class T> void matrix_dense<T>::matrix_matrix_multiplication(const matrix_dense<T>& A, const matrix_dense<T>& B){
    if (A.number_columns != B.number_rows){
        std::cerr<<"matrix_dense:matrix_matrix_multiplication: Dimension error in matrix-matrix-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else {
        resize(A.number_rows,B.number_columns);
        set_all(0.0);
        generic_matrix_matrix_multiplication_addition(A,B);
    }
}

template<class T> void matrix_dense<T>::matrix_transpose_vector_multiplication_add(const vector_dense<T>& x, vector_dense<T>& v) const {
     if ((number_columns != v.dimension())||(number_rows != x.dimension())){
         std::cerr << "matrix_dense:matrix_transpose_vector_multiplication_add(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     } else {
         generic_matrix__transpose_vector_multiplication_addition(x,v);
     }
  }

template<class T> void matrix_dense<T>::matrix_transpose_vector_multiplication(const vector_dense<T>& x, vector_dense<T>& v) const {
    if (number_rows != x.dimension() ){
        std::cerr << "matrix_dense:matrix_transpose_vector_multiplication(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else {
        v.resize(number_columns,0.0);
        generic_matrix_transpose_vector_multiplication_addition(x,v);
    }
}

//***************************************************************************************************************************************
//  Class matrix: matrix_dense valued operations                                                                                        *
//***************************************************************************************************************************************

template<class T> matrix_dense<T> matrix_dense<T>::operator*(T k) const {
    matrix_dense<T> Y(number_rows, number_columns,0.0);
    Integer i,j;
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++) Y.data[i][j]=k*data[i][j];
    return Y;
}

template<class T> matrix_dense<T> matrix_dense<T>::operator+ (const matrix_dense& X) const {
    if ((number_rows==X.number_rows)&&(number_columns==X.number_columns)){
        matrix_dense<T> Y(number_rows,number_columns,0.0);
        Integer i,j;
        for(i=0;i<number_rows;i++)
            for(j=0;j<number_columns;j++) Y.data[i][j]=data[i][j]+X.data[i][j];
        return Y;
    } else {
        std::cerr << "matrix_dense<T>::operator +: Dimensions error adding matrices."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> matrix_dense<T> matrix_dense<T>::operator- (const matrix_dense& X) const {
    if ((number_rows==X.number_rows)&&(number_columns==X.number_columns)){
        matrix_dense<T> Y(number_rows, number_columns,0.0);
        Integer i,j;
        for(i=0;i<number_rows;i++)
            for(j=0;j<number_columns;j++) Y.data[i][j]=data[i][j]-X.data[i][j];
        return Y;
    } else {
        std::cerr << "matrix_dense<T>::operator -: Dimensions error subtracting matrices."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> matrix_dense<T> matrix_dense<T>::operator*(const matrix_dense& X) const {
    matrix_dense<T> Y(number_rows, X.number_columns,0.0);
    Integer i,j,k;
    T summe;
    if (number_columns==X.number_rows){
        for(i=0;i<number_rows;i++)
            for(j=0;j<X.number_columns;j++){
                summe=0;
                for(k=0;k<number_columns;k++) summe+=data[i][k]*X.data[k][j];
                Y.data[i][j]=summe;
            }
        return Y;
    } else {
        std::cerr<<"matrix_dense<T>::operator *: Dimensions error multiplying matrices. The dimensions are: "<<std::endl<<"("<<
            number_rows<<"x"<<number_columns<<") und ("<<X.number_rows<<"x"<<
            X.number_columns<<")"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> matrix_dense<T>& matrix_dense<T>::operator= (const matrix_dense<T>& X){
    Integer i,j;
    if(this==&X) return *this;
    resize(X.number_rows,X.number_columns);
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++) data[i][j]=X.data[i][j];
    return *this;
}

template<class T> void matrix_dense<T>::resize(Integer m, Integer n){
    if(m<0) m = 0;
    if(n<0) n = 0;
    Integer i;
    if(m != number_rows || n != number_columns){
        if(data != 0){
            for(i=0;i<number_rows;i++)
                if (data[i] != 0){
                    delete[] data[i];
                    data[i] = 0;
                }
            delete[] data; data = 0;
        }
        number_rows=m;
        number_columns=n;
        if(number_rows == 0 || number_columns == 0){
            data = 0;
        } else {
            data = new T*[number_rows];
            for(i=0;i<number_rows;i++){
                data[i] = new T[number_columns];
            }
        }
    }
}


//***************************************************************************************************************************************
//  Class matrix_dense: Matrix-Vector-Multiplication                                                                                    *
//***************************************************************************************************************************************

template<class T> vector_dense<T> matrix_dense<T>::operator * (vector_dense<T> const & x) const {
    if (number_rows==x.dimension()){
        vector_dense<T> res(number_columns);
        generic_matrix_vector_multiplication_addition(x,res);
        return res;
    } else {
        std::cerr << "Dimension error in matrix_dense*vector_dense"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}


//***************************************************************************************************************************************
//  Class matrix_dense: Matrix-valued functions                                                                                         *
//***************************************************************************************************************************************

template<class T> matrix_dense<T> matrix_dense<T>::transp() const {
    matrix_dense<T> y(number_columns, number_rows);
    Integer i,j;
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++)
            y.data[j][i]=data[i][j];
    return y;
}


//***************************************************************************************************************************************
//   Class matrix: Generation of special matrices                                                                                       *
//***************************************************************************************************************************************

template<class T> void matrix_dense<T>::set_all(T d){
    for(Integer i=0;i<number_rows;i++)
        for(Integer j=0;j<number_columns;j++)
            data[i][j]=d;
}

template<class T> void matrix_dense<T>::diag(T d){
    set_all(0.0);
    for(Integer i=0;i<number_rows;i++)
        data[i][i]=d;
}

template<class T> void matrix_dense<T>::diag(const vector_dense<T>& d){
    resize(d.dimension(),d.dimension());
    set_all(0.0);
    for(Integer i=0;i<number_rows;i++)
        data[i][i]=d.read(i);
}


template<class T> void matrix_dense<T>::random_multiplicatively_perturbed_projection_matrix(Integer n, Integer rank, Integer min_nnz, Integer max_nnz, orientation_type O, Real eps_EV, Real eps_similarity) {
    Integer k;
    if(eps_EV <= 0.0) eps_EV = 0.0;
    if(eps_similarity <= 0.0) eps_similarity = 0.0;
    matrix_sparse<T> perturbed_ID_sparse;
    matrix_dense<T> perturbed_ID_dense, inv_perturbed_ID_dense,H;
    perturbed_ID_sparse.random_perturbed_projection_matrix(n,n,min_nnz,max_nnz,O,eps_similarity);
    perturbed_ID_dense.expand(perturbed_ID_sparse);
    inv_perturbed_ID_dense.invert(perturbed_ID_dense);
    resize(n,n);
    set_all(0.0);
    for(k=0;k<rank;k++) data[k][k] = 1.0 +  eps_EV *(2.0* ((T) rand()) / ((T)RAND_MAX) - 1.0);
    H.matrix_matrix_multiplication(*this,inv_perturbed_ID_dense);
    matrix_matrix_multiplication(perturbed_ID_dense,H);
}

template<class T> matrix_dense<T>& matrix_dense<T>::scale_rows(const vector_dense<T>& d){
     if (number_rows==d.dimension()){
         for(Integer i=0;i<number_rows;i++)
             for(Integer j=0;j<number_columns;j++)
                 data[i][j]*=d.read(i);
         return *this;
     } else {
         std::cerr << "Dimension error in matrix_dense::scale_rows"<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     }
  }
template<class T> matrix_dense<T>& matrix_dense<T>::scale_columns(const vector_dense<T>& d){
     if (number_columns==d.dimension()){
         for(Integer i=0;i<number_rows;i++)
             for(Integer j=0;j<number_columns;j++)
                 data[i][j]*=d.read(j);
         return *this;
     } else {
         std::cerr << "Dimension error in matrix_dense*::scale_columns"<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     }
  }

template<class T> matrix_dense<T>& matrix_dense<T>::inverse_scale_rows(const vector_dense<T>& d){
     if (number_rows==d.dimension()){
         for(Integer i=0;i<number_rows;i++)
             for(Integer j=0;j<number_columns;j++)
                 data[i][j]/=d.read(i);
         return *this;
     } else {
         std::cerr << "Dimension error in matrix_dense::scale_rows"<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     }
  }
template<class T> matrix_dense<T>& matrix_dense<T>::inverse_scale_columns(const vector_dense<T>& d){
     if (number_columns==d.dimension()){
         for(Integer i=0;i<number_rows;i++)
             for(Integer j=0;j<number_columns;j++)
                 data[i][j]/=d.read(j);
         return *this;
     } else {
         std::cerr << "Dimension error in matrix_dense*::scale_columns"<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     }
  }


template<class T> void matrix_dense<T>::permute_columns(const matrix_dense<T>& A, const index_list& perm){
    if (A.number_columns==perm.dimension()){
        resize(A.number_rows,A.number_columns);
        for(Integer i=0;i<A.number_rows;i++)
            for(Integer j=0;j<A.number_columns;j++)
                data[i][j] = A.data[i][perm[j]];
    } else {
        std::cerr << "Dimension error in matrix_dense::permute_columns"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}


template<class T> matrix_dense<T> matrix_dense<T>::permute_columns(const index_list& perm) const {
    matrix_dense<T> B;
    B.permute_columns(*this,perm);
    return B;
}


template<class T> void matrix_dense<T>::permute_rows(const matrix_dense<T>& A, const index_list& perm){
    if (A.number_rows==perm.dimension()){
        resize(A.number_rows,A.number_columns);
        for(Integer i=0;i<A.number_rows;i++)
            for(Integer j=0;j<A.number_columns;j++)
                data[i][j] = A.data[perm[i]][j];
    } else {
        std::cerr << "Dimension error in matrix_dense::permute_rows"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}


template<class T> matrix_dense<T> matrix_dense<T>::permute_rows(const index_list& perm) const {
    matrix_dense<T> B;
    B.permute_rows(*this,perm);
    return B;
}

template<class T> void matrix_dense<T>::elementwise_addition(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_addition: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] += A.data[i][j];
}

template<class T> void matrix_dense<T>::elementwise_subtraction(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_subtraction: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] -= A.data[i][j];
}

template<class T> void matrix_dense<T>::elementwise_multiplication(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_multiplication: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] *= A.data[i][j];
}

template<class T> void matrix_dense<T>::elementwise_division(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_division: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] /= A.data[i][j];
}


template<class T> void matrix_dense<T>::bandmatrixfull(T a,T b){
    for(Integer i=0;i<number_rows;i++)
        for(Integer j=0;j<number_columns;j++)
            data[i][j]=a-b*fabs((long double)(i-j));
  }

template<class T> void matrix_dense<T>::tridiag(T a, T b,T c){
    Integer i,j;
    for(i=0;i<number_rows;i++)
         for(j=0;j<number_columns;j++)
              data[i][j]=0;
    for(i=1;i<min(number_rows,number_columns);i++)
          data[i][i-1]=a;
    for(i=0;i<min(number_rows,number_columns);i++)
          data[i][i]=b;
    for(i=1;i<min(number_rows,number_columns);i++)
          data[i-1][i]=c;
  }

template<class T> void matrix_dense<T>::interpolation_matrix(){
    for(Integer i=0;i<number_rows;i++)
        for(Integer j=0;j<number_columns;j++){
             data[i][j]=pow((long double)i,(long double)j);
        }
  }

template<class T> void matrix_dense<T>::overwrite(const matrix_dense& A, Integer m, Integer n){
    Integer i,j;
    if(m+A.number_rows>number_rows || n+A.number_columns>number_columns||m<0||n<0){
        std::cerr<<"matrix_dense::overwrite: This matrix does not fit as desired."<<std::endl;
        return;
    }
    for(i=0;i<A.number_rows;i++)
        for(j=0;j<A.number_columns;j++)
            data[m+i][n+j] = A.data[i][j];
}



//***********************************************************************************************************************
// Class matrix_dense: functions, information                                                                           *
//***********************************************************************************************************************

template<class T> Integer matrix_dense<T>::rows() const {
     return number_rows;
  }

template<class T> Integer matrix_dense<T>::columns() const {
     return number_columns;
  }

template<class T> Real matrix_dense<T>::normF() const {
    Real normsq=0.0;
    for(Integer i=0;i<number_rows;i++)
    for(Integer j=0;j<number_columns;j++)
        normsq=normsq+pow(data[i][j],2);
    return sqrt(normsq);
  }

template<class T> Real matrix_dense<T>::norm1() const {
     Real norm=0.0;
     Real column_sum;
     for(Integer j=0;j<number_columns;j++){
         column_sum=0.0;
         for(Integer i=0;i<number_rows;i++) column_sum+=fabs(data[i][j]);
         if(column_sum>norm) norm=column_sum;
     }
     return norm;
  }

template<class T> T& matrix_dense<T>::operator()(Integer i, Integer j){
     #ifdef DEBUG
         if(0<=i&&i<number_rows&&0<=j&&j<number_columns)return data[i][j];
         else {std::cerr<<"matrix_dense(*,*): this matrix entry does not exist. Accessing Element ("<<i<<","<<j<<") of a ("<<number_rows<<","<<number_columns<<") matrix."<<std::endl; throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);}
     #endif
     return data[i][j];
  }

template<class T> const T& matrix_dense<T>::operator()(Integer i, Integer j) const {
    return const_cast<matrix_dense<T>&>(*this)(i, j);
}

//***********************************************************************************************************************
// Class matrix_dense: Conversion                                                                                       *
//***********************************************************************************************************************



template<class T> matrix_sparse<T> matrix_dense<T>::compress(orientation_type o, double threshold){
std::cerr<<"The use of this function is deprecated."<<std::endl;
std::cerr<<"Use member function matrix_sparse<T>::compress"<<std::endl;
std::cerr<<"Returning NULL matrix."<<std::endl;
matrix_sparse<T> M;
return M;
/*
     Integer counter=0;
     Integer i,j;
     for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++)
            if (fabs(data[i][j]) > threshold) counter++;
             M.reformat(number_rows, number_columns,counter,o);
     counter = 0;
     if(o == ROW){
         for(i=0;i<number_rows;i++){
             M.pointer[i]=counter;
             for(j=0;j<number_columns;j++)
                 if(fabs(data[i][j]) > threshold) {
                     M.indices[counter] = j;
                     M.data[counter] = data[i][j];
                     counter++;
             }
         }
         M.pointer[number_rows]=counter;
     } else {
         for(j=0;j<number_columns;j++){
             M.pointer[j]=counter;
             for(i=0;i<number_rows;i++)
                 if (fabs(data[i][j]) > threshold) {
                     M.indices[counter] = i;
                     M.data[counter] = data[i][j];
                     counter++;
                 }
         }
         M.pointer[number_columns]=counter;
     }
     return M;
*/
  }

template<class T> void matrix_dense<T>::expand(const matrix_sparse<T>& B) {
#ifdef DEBUG
    if(non_fatal_error(!B.check_consistency(),"matrix_dense::expand: matrix is inconsistent.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    Integer i;
    Integer j;
    resize(B.rows(),B.columns());
    set_all(0.0);
    if (B.orient() == ROW){
        for(i=0;i<B.get_pointer_size()-1;i++){ // std::cout<<"i schleife i="<<i<<"pointer[i]= "<<pointer[i]<< "pointer[i+1]= "<<pointer[i+1]<<std::endl;
            for(j=B.get_pointer(i);j<B.get_pointer(i+1);j++){
                data[i][B.get_index(j)]+=B.get_data(j);
            }
        }
    } else {
        for(i=0;i<B.get_pointer_size()-1;i++){
            for(j=B.get_pointer(i);j<B.get_pointer(i+1);j++){
                data[B.get_index(j)][i]+=B.get_data(j);
            }
        }
    }
  }

template<class T>  void matrix_dense<T>::compress(Real threshold){
     for(Integer i=0;i<number_rows;i++)
         for(Integer j=0;j<number_columns;j++)
              if (fabs(data[i][j])<threshold) data[i][j] = (T) 0;
}
//***************************************************************************************************************************************
//   Class matrix_dense: Input, Output                                                                                                  *
//***************************************************************************************************************************************

template<class T> std::istream& operator >> (std::istream& is, matrix_dense<T>& X){
     std::cout<<"Matrix elements for the ("<<X.number_rows<<"x"<<X.number_colums<<")-Matrix:"<<std::endl;
     for(Integer i=0;i<X.number_rows;i++)
       for(Integer j=0;j<X.number_columns;j++)
        is >> X.data[i][j];
       std::cout<<"End >>"<<std::endl;
     return is;
  }

template<class T>std::ostream& operator << (std::ostream& os, const matrix_dense<T>& x){
     os<<std::endl;
     for(Integer i=0;i<x.rows();i++){
         os <<"(";
         for(Integer j=0;j<x.columns();j++) os << std::setw(14) << x(i,j)<< "  ";
         os << " )" << std::endl;
     }
     os<<std::endl;
     if(x.rows() == 0) os<<"( )"<<std::endl;
     return os;
  }

//***************************************************************************************************************************************
//   Class matrix_dense: Gauss-Jordan Elimination                                                                                       *
//***************************************************************************************************************************************


template<class T> void matrix_dense<T>::pivotGJ(T **r, Integer k) const {
     Integer size=number_rows;
     T help;
     Integer p, i;
     p=k;
     for (i=k+1;i<size;i++)
         if (fabs(r[p][k])<fabs(r[i][k])) p=i;
     if (p!=k)
        for (i=0;i<=size;i++){
            help=r[p][i];
            r[p][i]=r[k][i];
            r[k][i]=help;
        }
    }


template<class T> Integer matrix_dense<T>::minusGJ(T **r, Integer k) const {
     Integer size=number_rows;
     Integer i, j;
     for (i=0;i<size;i++)
         if (i!=k)
             for (j=k+1;j<=size;j++){
                 if (r[k][k]==0) return 1;
                 r[i][j]=r[i][j]-r[i][k]/r[k][k]*r[k][j];
             }
         for (j=size;j>=k;j--)
         r[k][j]=r[k][j]/r[k][k];
         return 0;
    }


template<class T> void matrix_dense<T>::GaussJordan(const vector_dense<T> &b, vector_dense<T> &x) const {
    if(non_fatal_error(number_rows != number_columns,"Gauss-Jordan requires a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer k, j, i;
    Integer errorcode=0;
    Integer size=number_rows;
    x.resize(size,0.0);
    if(size == 0) return;
    T **r = new  T* [size];
    for (i=0;i<size;i++){
        r[i]=new  T [size+1];
    }
    for (k=0;k<size;k++){
        for (j=0;j<size;j++)
            r[k][j]=data[k][j];
        r[k][size]=b[k];
    }
    for (k=0;k<size;k++){
        pivotGJ(r,k);
        errorcode=minusGJ(r,k);
        if (errorcode==1) {std::cerr<<"System is not solvable;"<<std::endl;};
    }
    for (k=0;k<size;k++)
        x[k]=r[k][size];
    for (i=0;i<size;i++)
        delete []r[i];
    delete []r;
}



template<class T> bool matrix_dense<T>::solve(const vector_dense<T> &b, vector_dense<T> &x) const {
    Gauss(b,x);
    return true;
}



template<class T> Integer matrix_dense<T>::minus_invert(matrix_dense<T> &r, Integer k) const {
     Integer size=r.number_rows;
     Integer i, j;
     for (i=0;i<size;i++)
         if (i!=k)
             for (j=k+1;j<2*size;j++){
                 if (r.read(k,k)== (T) 0) return 1;
                 r(i,j)-= r.read(i,k)/r.read(k,k)*r.read(k,j);
             }
         for (j=2*size-1;j>=k;j--)
         r(k,j)/=r.read(k,k);
         return 0;
    }

template<class T> void matrix_dense<T>::pivot_invert(matrix_dense<T> &r, Integer k) const {
     Integer size=r.number_rows;
     T help;
     Integer p, i;
     p=k;
     for (i=k+1;i<size;i++)
         if (fabs(r.read(p,k))<fabs(r.read(i,k))) p=i;
     if (p!=k)
        for (i=0;i<2*size;i++){
            help=r.read(p,i);
            r(p,i)=r.read(k,i);
            r(k,i)=help;
        }
    }

template<class T> void matrix_dense<T>::invert(const matrix_dense<T> &B){
    if(non_fatal_error(number_rows != number_columns,"Gauss-Jordan requires a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer k, j;
    Integer errorcode=0;
    Integer size=B.number_rows;
    if(B.number_rows != B.number_columns){
        std::cerr<<"Matrix is not square;"<<std::endl;
        return;
    }
    resize(size,size);
    matrix_dense<T> r(size,2*size);
    for (k=0;k<size;k++){
        for (j=0;j<size;j++){
            r(k,j)=B.read(k,j);
            r(k,j+size)=0.0;
        }
        r(k,k+size)=1.0;
    }
    for (k=0;k<size;k++){
        pivot_invert(r,k);
        errorcode=minus_invert(r,k);
        if (errorcode==1) {std::cerr<<"Matrix is not invertible;"<<std::endl;};
    }
    for (k=0;k<size;k++)
        for (j=0;j<size;j++)
            (*this)(k,j)=r.read(k,j+size);
}


template<class T> Integer matrix_dense<T>::Gauss(const vector_dense<T> &b, vector_dense<T> &x) const{
    if(non_fatal_error((rows()!=columns()),"matrix_dense::Gauss: matrix must be square")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error((rows()!=b.dimension()),"matrix_dense::Gauss: the dimension of the right hand side is incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n=rows();
    index_list permut(n);
    vector_dense<T> y(n);
    Integer i,j,k;
    Integer maxc;
    T maxv, newmaxv;
    T *swap;
    Integer iswap;
    x.erase_resize_data_field(n);
    // begin factorization
    // searching for pivot.
    for (i=0;i<n;i++) {
        maxc = i;
        maxv = fabs(data[i][i]);
        for (j = i + 1; j < n; j++)
            if ((newmaxv = fabs(data[j][i])) > maxv) {
                maxv = newmaxv;
                maxc = j;
            }
        // matrix is singular, if no pivot can be found
        if (maxv == 0.0) {
            std::cerr << "matrix_dense::Gauss: Matrix is singular." << std::endl
                << "A row of zeroes occurred in the " << i << "th step."<< std::endl;
            return 0;
        }
        // Swap rows
        iswap = permut[maxc];
        permut[maxc] = permut[i];
        permut[i] = iswap;

        swap = data[maxc];
        data[maxc] = data[i];
        data[i] = swap;
        // Factorize
        for (j=i+1;j<n;j++) {
            data[j][i] /= data[i][i];
            for (k = i + 1; k < n; k++)
                data[j][k] -= data[j][i] * data[i][k];
        }
    }
    // Solve system
    // Forward elimination
    y=b;
    for (i=0;i<n;i++)
        for (j=i+1;j<n;j++)
            y[permut[j]] -= data[j][i] * y[permut[i]];
    // Backward elimination
    for (i=n-1;i>=0;i--) {
        x[i] = y[permut[i]];
        for (j=i+1;j<n;j++)
            x[i]-=data[i][j] * x[j];
        x[i] /= data[i][i];
    }
    // return with success
    return 1;
}

template<class T> bool matrix_dense<T>::ILUCP(const matrix_dense<T>& A, matrix_dense<T>& U, index_list& perm, Integer fill_in, Real tau, Integer& zero_pivots){
    if(tau>500.0) tau=0.0;
    else tau=std::exp(-tau*std::log(10.0));
    if(non_fatal_error(!A.square_check(),"matrix_dense::ILUCP: A must be a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(!U.square_check(),"matrix_dense::ILUCP: U must be a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(!square_check(),"matrix_dense::ILUCP: *this must be a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(A.columns() != columns() || U.columns() != columns(),"matrix_dense::ILUCP: Dimensions are incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n=A.rows();
    Integer k,i,j,p;
    zero_pivots=0;
    Real val_larg_el;
    Integer pos_larg_el;
    //Real norm_L, norm_U;
    vector_dense<T> w(n), z(n);
    vector_dense<bool> non_pivot(n);
    index_list inverse_perm(n);
    if(fill_in<1) fill_in=1;
    if(fill_in>n) fill_in=n;
    perm.resize(n);
    non_pivot.set_all(true);
    for(k=0;k<n;k++) for(i=0;i<n;i++) U.data[k][i]=0.0;
    for(k=0;k<n;k++) for(i=0;i<n;i++) data[k][i]=0.0;
    for(k=0;k<n;k++){
        z.set_all(0.0);
        w.set_all(0.0);
        for(i=0;i<n;i++) if(non_pivot[i]) z[i]=A.data[k][i];
        for(i=0;i<k;i++)
            for(j=0;j<n;j++)
                if(non_pivot[j]) z[j]-=data[k][i]*U.data[i][j];
        val_larg_el=abs(z[0]);
        pos_larg_el=0;
        for(i=1;i<n;i++)
            if(non_pivot[i])
                if(abs(z[i])>val_larg_el){
                    pos_larg_el=i;
                    val_larg_el=abs(z[i]);
                }
        if(val_larg_el==0.0){
            zero_pivots++;
            return false;
        }
        for(i=0;i<n;i++) if(non_pivot[i]) U.data[k][i]=z[i];
        p=inverse_perm[pos_larg_el];
        inverse_perm.switch_index(perm[k],pos_larg_el);
        perm.switch_index(k,p);
        non_pivot[pos_larg_el]=false;
        for(i=k+1;i<n;i++) w[i]=A.data[i][pos_larg_el];
        for(i=0;i<k;i++)
            for(j=k+1;j<n;j++)
                w[j]-=U.data[i][pos_larg_el]*data[j][i];
        for(i=k+1;i<n;i++) data[i][k] = w[i]/U.data[k][pos_larg_el];
        data[k][k]=1.0;
    }   // end for k.
    return true;
}

template<class T>  Real matrix_dense<T>::memory() const{
    return (Real) ((sizeof(T))* (Real) number_rows *(Real) number_columns) +  2*sizeof(Integer);
}

//***********************************************************************************************************************
//                                                                                                                      *
//           The implementation of the class index_list                                                                 *
//                                                                                                                      *
//***********************************************************************************************************************

index_list::index_list(){
}

index_list::index_list(Integer _size){
    resize(_size, _size);
}

index_list::index_list(Integer _size, Integer _reserved){
     resize(_size, _reserved);
}

void index_list::init(){
    fill_identity(vec());
}

void index_list::init(Integer n) {
#ifdef DEBUG
    non_fatal_error(dimension()<n,"index_list::init(Integer): index list cannot be initialized for integers larger than its size.");
#endif
    for(Integer i=0;i<std::min(n,dimension());i++)
        indices[i]=i;
}

void index_list::init(Integer n, Integer begin){
#ifdef DEBUG
    non_fatal_error(n>dimension(),"index_list::init(Integer, Integer): index list cannot be initialized for integers larger than its size.");
#endif
    for(Integer i=0;i<n;i++)
        indices[i]=i+begin;
}

void index_list::resize_without_initialization(Integer newsize){
    indices.resize(newsize);
}

void index_list::resize_without_initialization(Integer newsize, Integer new_memory){
    indices.resize(newsize);
    indices.reserve(new_memory);
}

void index_list::resize(Integer newsize){
    make_identity(vec(), newsize);
}

void index_list::resize(Integer newsize, Integer new_memory){
    resize_without_initialization(newsize, new_memory);
    init();
}

void index_list::resize_with_constant_value(Integer newsize, Integer d){
    indices.resize(newsize, d);
}

void index_list::switch_index(Integer i, Integer j){
#ifdef DEBUG
    if((i>=dimension())||(j>=dimension())|| (i<0) || (j<0)){
        std::cerr << "index_list::switch_index: out of domain error: size of list "<<dimension()<<" indices to be switched: "<<i<<" "<<j<<std::endl<<std::flush;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    std::swap(indices[i], indices[j]);
}

Integer& index_list::operator[](Integer j){
#ifdef DEBUG
    if (j >= dimension() || j<0 ){
        std::cerr<<"index_list::operator[]: given index is out of domain"<<std::endl;
        std::cerr<<"index_list has size "<<dimension()<<" and requested index is "<<j<<std::endl;
        std::cerr<<"complete list"<<std::endl;
        std::cerr<<*this;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    return indices[j];
}

const Integer& index_list::operator[](Integer j) const {
#ifdef DEBUG
    if (j >= dimension() || j<0 ){
        std::cerr<<"index_list::operator[]: given index is out of domain"<<std::endl;
        std::cerr<<"index_list has size "<<dimension()<<" and requested index is "<<j<<std::endl;
        std::cerr<<"complete list"<<std::endl;
        std::cerr<<*this;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    return indices[j];
}

void index_list::print_info() const {
    std::cout<<"An index list of dimension "<<dimension()<<std::endl;
  }

Integer index_list::find(Integer k) const {
    for(Integer i=0; i<dimension(); i++) if (indices[i]==k) return i;
    return -1;
  }

Integer index_list::memory_used() const {
    return indices.capacity();
  }

void index_list::quicksort(Integer left, Integer right){
     #ifdef DEBUG
         if(left<0 || right >= dimension()){
             std::cerr<<"index_list::quicksort: arguments out of range. Not sorting."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer i,j;
     Integer m;
     if(left<right){
         m=indices[left];
         i=left;
         j=right;
         while(i<=j){
             while(indices[i]<m) i++;
             while(indices[j]>m) j--;
             if(i<=j){
                 switch_index(i,j);
                 i++;
                 j--;
             }
         }
         quicksort(left,j);
         quicksort(i,right);
     }
  }

void index_list::quicksort_with_inverse(index_list& invperm, Integer left, Integer right){
     #ifdef DEBUG
         if(left<0 || right >= dimension()){
             std::cerr<<"index_list::quicksort_with_inverse: arguments out of range. Not sorting."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer i,j;
     Integer m;
     if(left<right){
         m=indices[left];
         i=left;
         j=right;
         while(i<=j){
             while(indices[i]<m) i++;
             while(indices[j]>m) j--;
             if(i<=j){
                 invperm.switch_index((*this)[i], (*this)[j]);
                 switch_index(i,j);
                 i++;
                 j--;
             }
         }
         quicksort_with_inverse(invperm,left,j);
         quicksort_with_inverse(invperm,i,right);
     }
  }

void index_list::quicksort(index_list& list, Integer left, Integer right){
     #ifdef DEBUG
         if(left<0 || right >= dimension()){
             std::cerr<<"index_list::quicksort: arguments out of range. Not sorting."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer i,j;
     Integer m;
     if(left<right){
         m=indices[left];
         i=left;
         j=right;
         while(i<=j){
             while(indices[i]<m) i++;
             while(indices[j]>m) j--;
             if(i<=j){
                 switch_index(i,j);
                 list.switch_index(i,j);
                 i++;
                 j--;
             }
         }
         quicksort(list,left,j);
         quicksort(list,i,right);
     }
  }

void index_list::permute(const index_list& x, const index_list& perm){
    indices = permute_vec(x.vec(), perm);
}

 index_list index_list::permute(const index_list& perm){
     index_list x;
     x.permute(*this, perm);
     return x;
 }

void index_list::invert(const index_list& perm){
    indices.resize(perm.dimension());
    for(Integer i=0; i<dimension(); i++)
        indices[perm[i]] = i;
}


void index_list::reflect(const index_list& perm){
    resize_without_initialization(perm.dimension());
    for(Integer i=0; i<dimension(); i++) (*this)[i] = dimension() - 1 - perm[i];
}


void index_list::interchange(index_list& A){
    std::swap(indices, A.indices);
}

std::ostream& operator << (std::ostream& os, const index_list& x){
    for(Integer i=0;i<x.dimension();i++) os << x[i] << std::endl; std::cout<<std::endl;
    return os;
}


void index_list::compose(const index_list& P, const index_list& Q){
    if(non_fatal_error(P.dimension() != Q.dimension(), "index_list::compose: Arguments must have same dimension.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    resize_without_initialization(P.dimension());
    Integer i;
    for (i=0; i<P.dimension(); i++) (*this)[i] = P[Q[i]];
}



void index_list::compose(const index_list& P){
    compose_right(P);
}


void index_list::compose_right(const index_list& P){
    index_list H;
    H.compose(*this,P);
    interchange(H);
}


void index_list::compose_left(const index_list& P){
    index_list H;
    H.compose(P,*this);
    interchange(H);
}


bool index_list::ID_check() const {
    for(Integer i=0; i<dimension(); i++)
        if ((*this)[i] != i)
            return false;
    return true;
}

bool index_list::check_if_permutation() const {
    Integer i;
    std::vector<Integer> list(dimension(), -1);
    for(i=0;i<dimension();i++){
        if (list[(*this)[i]]==-1) list[(*this)[i]] = i;
        else return false;
    }
    for(i=0;i<dimension();i++) if (list[i] == -1) return false;
    return true;
}

Real index_list::memory() const{
    return (Real)((indices.capacity()+1)*sizeof(Integer));
}

Integer index_list::equality(const index_list& v) const {
     #ifdef DEBUG
         if(dimension() != v.dimension()){
             std::cerr<<"index_list::relative_equality: the lists have incompatible dimensions."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     Integer counter = 0;
     Integer i;
     for(i=0;i<dimension();i++) if(indices[i] == v.indices[i]) counter++;
     return counter;
}

Integer index_list::equality(const index_list& v, Integer from, Integer to) const {
     #ifdef DEBUG
         if(dimension() != v.dimension()){
             std::cerr<<"index_list::equality: the lists have incompatible dimensions."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
         if(from < 0 || to > dimension()){
             std::cerr<<"index_list::equality: range exceeds indices. Returning -1."<<std::endl;
             return (Integer) -1;
         }
     #endif
     Integer counter = 0;
     Integer i;
     for(i=from;i<to;i++) if(indices[i] == v.indices[i]) counter++;
     return counter;
}


Real index_list::relative_equality(const index_list& v) const {
    return (Real) equality(v) / (Real) dimension();
 }

Real index_list::relative_equality(const index_list& v, Integer from, Integer to) const {
#ifdef DEBUG
    if(to <= from){
        std::cerr<<"index_list::relative_equality: range is empty. Returning -1."<<std::endl;
        return (Real) -1.0;
    }
#endif
    return (Real) equality(v,from,to) / (Real) (to-from);
}


//***********************************************************************************************************************
//                                                                                                                      *
//           The implementation of other functions                                                                      *
//                                                                                                                      *
//***********************************************************************************************************************

template<class T> T scalar_prod(const matrix_sparse<T> &A, Integer m, const matrix_oriented<T> &B, Integer n){
     #ifdef DEBUG
         if(non_fatal_error((m>=A.dim_along_orientation())||(n>=B.dim_along_orientation()),"scalar_prod: these rows/columns do not exist.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         if(non_fatal_error((A.orientation != B.orientation),"scalar_prod: the arguments must have the same orientation.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         if(non_fatal_error((A.dim_against_orientation() != B.dim_against_orientation()),"scalar_prod: the rows/columns of the arguments have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     T prod=0.0;
     Integer offset = ((Integer)(n))*((Integer)(B.dim_against_orientation()));
     for(Integer k=A.read_pointer(m); k<A.read_pointer(m+1);k++)
         prod += A.read_data(k)*B.read_data(offset+A.read_indices(k));
     return prod;
   }

template<class T> T scalar_prod(const matrix_sparse<T> &A, Integer m, const matrix_sparse<T> &B, Integer n){
     #ifdef DEBUG
         if(non_fatal_error((m>=A.dim_along_orientation())||(n>=B.dim_along_orientation()),"scalar_prod: these rows/columns do not exist.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         if(non_fatal_error((A.orientation != B.orientation),"scalar_prod: the arguments must have the same orientation.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         if(non_fatal_error((A.dim_against_orientation() != B.dim_against_orientation()),"scalar_prod: the rows/columns of the arguments have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     T scalar_product=0.0;
     Integer ind_m = A.read_pointer(m);
     Integer ind_n = B.read_pointer(n);
     Integer ind_mmax = A.read_pointer(m+1);
     Integer ind_nmax = B.read_pointer(n+1);
     while ((ind_m < ind_mmax)&&(ind_n < ind_nmax)){
         if(A.read_indices(ind_m) == B.read_indices(ind_n)){
             scalar_product += A.read_data(ind_m) * B.read_data(ind_n);
             ind_m++;
             ind_n++;
         } else {
             if (A.read_indices(ind_m) < B.read_indices(ind_n)){
                 ind_m++;
             } else {
                 ind_n++;
             }
         }
     } // end while
     return scalar_product;
   }

} // end namespace iluplusplus

#endif

