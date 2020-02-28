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
#include "ILUCDP.hpp"
#include "IChol.hpp"
#include "ILU0.hpp"

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
    std::swap(data[i], data[j]);
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
                 switch_entry(left,right);
                 list.switch_index(left,right);
             }  // end if
             break;
         } else {
             mid = (left+right)/2;
             switch_entry(mid,left+1);
             list.switch_index(mid,left+1);
             if(data[left]>data[right]){
                 switch_entry(left,right);
                 list.switch_index(left,right);
             }
             if(data[left+1]>data[right]){
                 switch_entry(left+1,right);
                 list.switch_index(left+1,right);
             }
             if(data[left]>data[left+1]){
                 switch_entry(left,left+1);
                 list.switch_index(left,left+1);
             }
             i=left+1;
             j=right;
             const T a=data[left+1];
             a_list=list[left+1];
             while(true){
                 do i++; while(data[i]<a);
                 do j--; while(data[j]>a);
                 if(j<i) break;
                 switch_entry(i,j);
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
    for(i=from;i<to;i++) norm_input += absvalue_squared(weights[i]*data[perm[i]]);
    norm_input=sqrt(norm_input);
    for(i=from;i<to;i++){
        product = weights[i]*fabs(data[perm[i]]);
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
    for(i=from;i<to;i++) norm += absvalue_squared(weight[i]*data[i]);
    norm=sqrt(norm);
    for(i=from;i<to;i++){
        product=fabs(weight[i]*data[i]);
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
    if(non_fatal_error((begin+n>v.size),"vector_dense::absvalue: dimensions are incompatible"))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    erase_resize_data_field(n);
    for(Integer i=0;i<n;i++) data[i]=fabs(v.data[i+begin]);
  }

template<class T> void vector_dense<T>::absvalue(const T* values, Integer begin, Integer n){   // (*this) contains n absolute values of the field data from begin.
    erase_resize_data_field(n);
    for(Integer i=begin;i<begin+n;i++) data[i]=fabs(values[i]);
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

template<class T> bool vector_dense<T>::zero_check(Integer k){
#ifdef DEBUG
    if(k<0||k>=size){
        std::cerr<<"vector_dense::zero_check: index out of range."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    return (data[k]==0);
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
    return occupancy[j] < 0;
}

template<class T> bool vector_sparse_dynamic<T>::non_zero_check(Integer j) const {
    return !zero_check(j);
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
            if(input_abs[i]>val_larg_el){
                pos_larg_el=i;
                val_larg_el=input_abs[i];
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
                if(input_abs[i]>val_larg_el){
                    pos_larg_el=i;
                    val_larg_el=input_abs[i];
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
            if(input_abs[i]>val_larg_el){
                pos_larg_el=i;
                val_larg_el=input_abs[i];
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
        norm += absvalue_squared(weights[pointer[i]]*data[i]);
    }
    if(val_larg_el*perm_tol>std::abs(read(pivot_position))){ // do pivoting
        norm=sqrt(norm);
        for(i=0;i<nnz;i++){
            product=std::abs(data[i])*weights[pointer[i]];
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
                if(input_abs[i]>val_larg_el){
                    pos_larg_el=i;
                    val_larg_el=input_abs[i];
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
            if(std::abs(data[i])*weights[pointer[i]]> norm*tau && pointer[i] != pivot_position){
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
            norm += absvalue_squared(weights[pointer[i]]*data[i]);
    }
    norm=sqrt(norm);
    for(i=0;i<nnz;i++){
        product = weights[pointer[i]]* std::abs(data[i]);
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
            sum += input_abs[i];
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
            sum += input_abs[i];
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
            //product = (weight + weights[pointer[i]]) * std::abs(data[i]);
            product = max(weight,weights[pointer[i]]) * std::abs(data[i]);
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs[i];
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
            //product = (weight + weights[pointer[i]]) * std::abs(data[i]);
            product = max(weight,weights[pointer[i]]) * std::abs(data[i]);
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
            //product = (weight + weights[pointer[i]]) * std::abs(data[i]);
            product = max(weight,weights[pointer[i]]) * std::abs(data[i]);
            input_abs[i]=product;
            complete_list[i]=pointer[i];
        }
        input_abs.quicksort(complete_list,0,nnz-1);
        for(i=0;i<nnz;i++){
            sum += input_abs[i];
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
            //product = (weight + weights[pointer[i]]) * std::abs(data[i]);
            product = max(weight,weights[pointer[i]]) * std::abs(data[i]);
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
            sum += input_abs[i];
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
            sum += input_abs[i];
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
            sum += input_abs[i];
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
            sum += input_abs[i];
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
                if(input_abs_U[i]>input_abs_U[pos_larg_element])
                    pos_larg_element=i;
            complete_list_U.switch_index(pos_larg_element,number_elements_larger_tau_U-1);
            list_U.resize_without_initialization(n_U);
            for (i=0;i<list_U.dimension();i++) list_U[i]=complete_list_U[offset+i];
        } else {
            pos_larg_element=0;
            //if(number_elements_larger_tau_U>0)  //
            for(i=1;i<number_elements_larger_tau_U;i++)
                if(input_abs_U[i]>input_abs_U[pos_larg_element])
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
                if(input_abs_U[i]>input_abs_U[pos_larg_element])
                    pos_larg_element=i;
            complete_list_U.switch_index(pos_larg_element,number_elements_larger_tau_U-1);
            list_U.resize_without_initialization(n_U);
            for (i=0;i<list_U.dimension();i++) list_U[i]=complete_list_U[offset+i];
        } else {
            pos_larg_element=0;
            for(i=1;i<number_elements_larger_tau_U;i++)
                if(input_abs_U[i]>input_abs_U[pos_larg_element])
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
        if (invperm[pointer[i]]<mid) norm_input_L += absvalue_squared(weights_L[invperm[pointer[i]]]*data[i]);
        else norm_input_U += absvalue_squared(data[i]);
    norm_input_L=sqrt(norm_input_L);
    norm_input_U=sqrt(norm_input_U);
    for(i=0;i<nnz;i++){
        if(invperm[pointer[i]]<mid){
            product = std::abs(weights_L[invperm[pointer[i]]]*data[i]);
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

template<class T> void matrix_sparse<T>::insert_data(const vector_dense<T>& data_vector, const index_list& list, Integer begin_index)
{
    Integer j,k;
    for(Integer i=0; i<list.dimension(); i++){
        k=begin_index+i;
        j=list[i];
        indices[k]=j;
        data[k]=data_vector[j];
    }
}

template<class T> void matrix_sparse<T>::insert_data(
        const vector_dense<T>& data_vector, const index_list& list, Integer begin_index_matrix,
        Integer begin_index_list, Integer n, Integer offset)
{
    Integer index_matrix, index_list, index_data;
    if (non_fatal_error((n+begin_index_list>list.dimension()),
                "matrix_sparse<T>::insert_data: trying to insert too many elements."))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
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

template<class T>
matrix_sparse<T> matrix_sparse<T>::change_orientation() const
{
    matrix_sparse<T> A(other_orientation(orientation), number_rows, number_columns, actual_non_zeroes());

    Integer i,j,k,l;

    for (i=0; i<pointer[pointer_size-1]; i++)
        A.pointer[1+indices[i]]++;

    for (i=1; i<A.pointer_size; i++) A.pointer[i] += A.pointer[i-1];

    std::vector<Integer> counter(A.pointer_size, 0);

    for (i=0; i<pointer_size-1; i++) {
        for(j=pointer[i]; j< pointer[i+1]; j++) {
            l = indices[j];
            k = A.pointer[l] + counter[l];
            A.data[k] = data[j];
            A.indices[k] = i;
            counter[l]++;
        }
    }
    return A;
}

template<class T>
matrix_sparse<T> matrix_sparse<T>::natural_triangular_part() const
{
    if (!square_check())
        throw std::logic_error("can only compute triangular part of square matrix");

    matrix_sparse<T> A(orientation, number_rows, number_columns, actual_non_zeroes());

    Integer* cur_ind = &A.indices[0];
    T* cur_dat = &A.data[0];

    for (Integer i = 0; i < dim_major(); ++i) {
        for (Integer k = pointer[i]; k < pointer[i+1]; ++k) {
            const Integer j = indices[k];
            if (j <= i) {
                *cur_ind++ = j;
                *cur_dat++ = data[k];
            }
        }
        // finished i-th row - set pointer to start of next row
        A.pointer[i + 1] = static_cast<Integer>(cur_ind - &A.indices[0]);
    }
    return A;
}

template<class T> void matrix_sparse<T>::insert(const matrix_sparse<T> &A, const vector_dense<T>& row, const vector_dense<T>& column, T center, Integer pos_row, Integer pos_col, Real threshold){
    if(A.orient()==ROW) insert_orient(A,row,column,center,pos_row,pos_col,threshold);
    else insert_orient(A,column,row,center,pos_col,pos_row,threshold);
}


template<class T> void matrix_sparse<T>::insert_orient(const matrix_sparse<T> &A, const vector_dense<T>& along_orient, const vector_dense<T>& against_orient, T center, Integer pos_along_orient, Integer pos_against_orient, Real threshold){
    if(non_fatal_error(A.dim_along_orientation() != against_orient.dimension() || A.dim_against_orientation() != along_orient.dimension()   ,"matrix_sparse<T>::insert_orient: dimensions of vectors to be inserted are not compatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(pos_along_orient < 0 || pos_against_orient < 0 || pos_along_orient > A.dim_along_orientation() ||  pos_against_orient > A.dim_against_orientation(),"matrix_sparse<T>::insert_orient: insert positions are not available.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    std::vector<bool> selected_along_orient(along_orient.dimension(),false);
    std::vector<bool> selected_against_orient(against_orient.dimension(),false);
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
    reformat(m, n, nz, o);
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



template<class T> matrix_sparse<T>::matrix_sparse(const matrix_sparse& X)
{
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

// move constructor
template <class T>
matrix_sparse<T>::matrix_sparse(matrix_sparse&& X)
{
    data            = X.data;
    pointer         = X.pointer;
    indices         = X.indices;
    orientation     = X.orientation;
    number_rows     = X.number_rows;
    number_columns  = X.number_columns;
    nnz             = X.nnz;
    pointer_size    = X.pointer_size;

    X.null_matrix_keep_data();
}

template<class T> matrix_sparse<T>::~matrix_sparse() { // std::cout<<"matrixdestruktor"<<std::endl;
    if (!non_owning) {
        if (data    != 0) delete [] data; data=0;
        if (indices != 0) delete [] indices; indices=0;
        if (pointer != 0) delete [] pointer; pointer=0;
    }
}

template<class T> matrix_sparse<T>& matrix_sparse<T>::operator= (matrix_sparse<T> X){
    // copy-and-swap idiom (copy is made by passing X by value)
    interchange(X);
    return *this;
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


template<class T> void matrix_sparse<T>::scale_orientation_based(const vector_dense<T>& D1, const vector_dense<T>& D2){
    if(non_fatal_error( ((D1.dimension() != this->dim_along_orientation())||(D2.dimension() != this->dim_against_orientation())), "matrix_sparse::scale: matrix and vector have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer i;
    Integer j;
    for(i=0;i<pointer_size-1;i++)
        for(j=pointer[i];j<pointer[i+1];j++)
            data[j] *= D1[i]*D2[read_index(j)];
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

template<class T> void matrix_sparse<T>::reorder(const index_list& invperm){
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

template<class T> void matrix_sparse<T>::compress(double threshold){
    // need a few variables:
    Integer i,j;
    Integer k;
    Integer counter=0;
    // make new fields to temporarily store the new data.
    std::vector<Integer> new_pointer(pointer_size);
    std::vector<Integer> new_indices(nnz);
    std::vector<T>       new_data(nnz);
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
    std::vector<Integer> new_pointer(pointer_size);
    std::vector<Integer> new_indices(nnz);
    std::vector<T>       new_data(nnz);
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

template<class T> void matrix_sparse<T>::triangular_solve_perm(special_matrix_type form, matrix_usage_type use, const index_list& perm, const vector_dense<T>& b, vector_dense<T>& x) const{
    if(non_fatal_error(!(square_check()),"matrix_sparse::triangular_solve_perm: matrix needs to be square."))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(b.dimension() != number_rows, "matrix_sparse::triangular_solve_perm: size of rhs is incompatible."))
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);

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
         ((form==PERMUTED_UPPER_TRIANGULAR) && (orientation==COLUMN) && (use==TRANSPOSE)) )
    {
        std::cerr<<"matrix_sparse<T>::triangular_solve_perm: this particular form of solving should not be needed!!"<<std::endl;
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
            if (use==ID )std::cout<<"      triangular_solve_perm: using: PERMUTED_LOWER_TRIANGULAR,COLUMN,ID"<<std::endl;
            else         std::cout<<"      triangular_solve_perm: using: PERMUTED_UPPER_TRIANGULAR,ROW,TRANSPOSE"<<std::endl;
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
            if (use==ID )std::cout<<"      triangular_solve_perm: using: PERMUTED_UPPER_TRIANGULAR,ROW,ID"<<std::endl;
            else         std::cout<<"      triangular_solve_perm: using: PERMUTED_LOWER_TRIANGULAR,COLUMN,TRANSPOSE"<<std::endl;
        #endif
        vector_dense<T> y;
        y=b;
        for(k=number_rows-1;k>=0;k--){
            for(j=pointer[k]+1;j<pointer[k+1];j++) y[k]-= data[j]*x[indices[j]];
            non_fatal_error(data[pointer[k]]==0,"matrix_sparse::triangular_solve_perm: pivot must be non-zero.");
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
        std::cerr<<"matrix_sparse<T>::triangular_solve_perm: this particular form of solving should not be needed!!"<<std::endl;
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
    throw std::runtime_error("matrix_sparse::triangular_solve_perm: unknown matrix usage");
}

template<class T> void matrix_sparse<T>::triangular_solve_perm(special_matrix_type form, matrix_usage_type use, const index_list& perm, vector_dense<T>& x) const {
    vector_dense<T> b = x;
    triangular_solve_perm(form, use, perm, b, x);
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
                w[i]=M.data[M.pointer[k]+i]*weights[M.indices[M.pointer[k]+i]];
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
                w[i]=M.data[M.pointer[k]+1+i]*weights[M.indices[pointer[k]+1+i]];
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
    matrix_sparse<T> A = this->change_orientation();

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
    matrix_sparse<T> A = this->change_orientation();

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
    matrix_sparse<T> A = this->change_orientation();

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
    matrix_sparse<T> A = this->change_orientation();

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
    matrix_sparse<T> A = this->change_orientation();

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
    matrix_sparse<T> A = this->change_orientation();

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
    std::vector<Integer> firstA(n), listA(n), headA(n);
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
        const iluplusplus_precond_parameter& IP, index_list& P, index_list& Q,
        index_list& invP, index_list& invQ, vector_dense<T>& Drow, vector_dense<T>& Dcol){
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
    for (size_t k = 0; k < IP.get_PREPROCESSING().size(); ++k) {
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
        if(perm_along.check_if_permutation()) std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: first index_list is a permutation of size "<<perm_along.dimension()<<std::endl;
        else std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: first index_list is NOT a permutation of size "<<perm_along.dimension()<<std::endl;
        if(invperm_against.check_if_permutation()) std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: second index_list is a permutation of size "<<invperm_against.dimension()<<std::endl;
        else std::cerr<<"matrix_sparse::permute_along_with_perm_and_against_orientation_with_invperm: second index_list is NOT a permutation of size "<<invperm_against.dimension()<<std::endl;
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
    index_list permP, permQ, invpermP, invpermQ;
    if(PQorient == A.orientation){
        pos = A.ddPQ(invpermP,invpermQ,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        if (A.orientation == COLUMN) permute(A,permQ,permP);
        else permute(A,permP,permQ);
    } else {
        matrix_sparse<T> B = A.change_orientation();
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
    index_list permP, permQ, invpermP, invpermQ;
    if(PQorient == A.orientation){
        pos = A.ddPQ(invpermP,invpermQ,from,to,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        if (A.orientation == COLUMN) permute(A,permQ,permP);
        else permute(A,permP,permQ);
    } else {
        matrix_sparse<T> B = A.change_orientation();
        pos = B.ddPQ(invpermP,invpermQ,from,to,tau);
        permP.invert(invpermP);
        permQ.invert(invpermQ);
        permute(B,permP,permQ);
    }
    return pos;
}



template<class T> Integer matrix_sparse<T>::ddPQ(matrix_sparse<T>& A, const vector_dense<T>& bold, vector_dense<T>& bnew, orientation_type PQorient, Real tau){
    Integer pos;
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
        matrix_sparse<T> B = A.change_orientation();
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
        matrix_sparse<T> B = A.change_orientation();
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

