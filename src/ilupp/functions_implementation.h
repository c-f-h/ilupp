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


#ifndef FUNCTIONS_IMPLEMENTATION_H
#define FUNCTIONS_IMPLEMENTATION_H

#include <iostream>
#include <string>

#include "declarations.h"
#include "functions.h"

namespace iluplusplus {


std::string integertostring(Integer k){
    std::string part;
    Integer trunc;
    if(k == 0) return "0";
    if(k < 0) trunc = -k;
    else trunc = k;
    while(trunc != 0){
        switch(trunc % 10){
            case 0: part = "0"+part; break;
            case 1: part = "1"+part; break;
            case 2: part = "2"+part; break;
            case 3: part = "3"+part; break;
            case 4: part = "4"+part; break;
            case 5: part = "5"+part; break;
            case 6: part = "6"+part; break;
            case 7: part = "7"+part; break;
            case 8: part = "8"+part; break;
            case 9: part = "9"+part; break;
        }
        trunc = trunc/10;
    }
    if(k<0) part = "-"+part;
    return part;
  }

std::string integertostring_with_spaces(Integer k){
    std::string part;
    Integer trunc;
    Integer counter = 0;
    if(k == 0) return "0";
    if(k < 0) trunc = -k;
    else trunc = k;
    while(trunc != 0){
        counter++;
        switch(trunc % 10){
            case 0: part = "0"+part; break;
            case 1: part = "1"+part; break;
            case 2: part = "2"+part; break;
            case 3: part = "3"+part; break;
            case 4: part = "4"+part; break;
            case 5: part = "5"+part; break;
            case 6: part = "6"+part; break;
            case 7: part = "7"+part; break;
            case 8: part = "8"+part; break;
            case 9: part = "9"+part; break;
        }
        trunc = trunc/10;
        if(counter % 3 == 0 && trunc != 0) part = " "+part;
    }
    if(k<0) part = "-"+part;
    return part;
  }

std::string booltostring(bool b){
    std::string out;
    if(b) out = "TRUE"; else out = "FALSE";
    return out;
}

Integer RoundRealToInteger(Real d){
  return (Integer) (d<0?d-.5:d+.5);
}       


orientation_type other_orientation(orientation_type o)  // returns the other orientation
  {
     if (o == ROW) return COLUMN;
     else return ROW;
  }


matrix_usage_type other_usage(matrix_usage_type u)
  {
     if (u == ID) return TRANSPOSE;
     else return ID;
  }


std::string string(preprocessing_type pt){
    std::string output;
    switch(pt){
        case  TEST_ORDERING:                           output = "T";     break;
        case  NORMALIZE_COLUMNS:                       output = "NC";    break;
        case  NORMALIZE_ROWS:                          output = "NR";    break;
#ifdef ILUPLUSPLUS_USES_SPARSPAK
        case  REVERSE_CUTHILL_MCKEE_ORDERING:          output = "RCM";   break;
#endif
        case  PQ_ORDERING:                             output = "PQ";    break;
        case  DYN_AV_PQ_ORDERING:                      output = "dPQ";   break;
        case  SYMM_PQ:                                 output = "sPQ";   break;
        case  MAX_WEIGHTED_MATCHING_ORDERING:          output = "IM";    break;  // I-Matrix
#ifdef ILUPLUSPLUS_USES_METIS
        case  METIS_NODE_ND_ORDERING:                  output = "Met";   break;
#endif
        case  UNIT_OR_ZERO_DIAGONAL_SCALING:           output = "uD";    break;
#ifdef ILUPLUSPLUS_USES_PARDISO
        case  PARDISO_MAX_WEIGHTED_MATCHING_ORDERING:  output = "PIM";   break;  // Pardiso I-Matrix
#endif
        case  SPARSE_FIRST_ORDERING:                   output = "sf";    break;
        case  SYMM_MOVE_CORNER_ORDERING:               output = "s";     break;
        case  SYMM_MOVE_CORNER_ORDERING_IM:            output = "si";    break;
        case  SYMB_SYMM_MOVE_CORNER_ORDERING:          output = "ss";    break;
        case  SYMB_SYMM_MOVE_CORNER_ORDERING_IM:       output = "ssi";   break;
        case  SP_SYMM_MOVE_CORNER_ORDERING:            output = "sp";    break;
        case  SP_SYMM_MOVE_CORNER_ORDERING_IM:         output = "spi";   break;
        case  WGT_SYMM_MOVE_CORNER_ORDERING:           output = "ws";    break;
        case  WGT_SYMM_MOVE_CORNER_ORDERING_IM:        output = "wsi";   break;
        case  WGT2_SYMM_MOVE_CORNER_ORDERING:          output = "w2s";   break;
        case  WGT2_SYMM_MOVE_CORNER_ORDERING_IM:       output = "w2si";  break;
        case  DD_SYMM_MOVE_CORNER_ORDERING_IM:         output = "dds";   break;
        default:                                       output = "???";   break;
    }
    return output;
}

std::string cap_string(data_type t){
    std::string output;
    switch(t){
        case SUCC_SOLVE:    output = "SUCC_SOLVE";       break;
        case THRESHOLD:     output = "THRESHOLD";        break; 
        case FILLIN:        output = "FILLIN";           break; 
        case MEM_STORAGE:   output = "MEM_STORAGE";      break; 
        case MEM_USED:      output = "MEM_USED";         break; 
        case MEM_ALLOCATED: output = "MEM_ALLOCATED";    break; 
        case ITERATIONS:    output = "ITERATIONS";       break; 
        case ABS_ERROR:     output = "ABS_ERROR";        break; 
        case REL_RESIDUAL:  output = "REL_RESIDUAL";     break; 
        case ABS_RESIDUAL:  output = "ABS_RESIDUAL";     break; 
        case SETUP_TIME:    output = "SETUP_TIME";       break; 
        case ITER_TIME:     output = "ITER_TIME";        break; 
        case TOTAL_TIME:    output = "TOTAL_TIME";       break; 
        case MATRIX_DIM:    output = "MATRIX_DIM";       break; 
        case MATRIX_NNZ:    output = "MATRIX_NNZ";       break; 
        case LEVELS:        output = "LEVELS";           break;
        default:            output = "???";              break;
    }
    return output;
}

std::string string(data_type t){
    std::string output;
    switch(t){
        case SUCC_SOLVE:    output = "successful solve";   break;
        case THRESHOLD:     output = "threshold";          break; 
        case FILLIN:        output = "fill-in";            break; 
        case MEM_STORAGE:   output = "memory for storage"; break; 
        case MEM_USED:      output = "memory used";        break; 
        case MEM_ALLOCATED: output = "memory allocated";   break; 
        case ITERATIONS:    output = "iterations";         break; 
        case ABS_ERROR:     output = "absolute error";     break; 
        case REL_RESIDUAL:  output = "relative residual";  break; 
        case ABS_RESIDUAL:  output = "absolute residual";  break; 
        case SETUP_TIME:    output = "setup time";         break; 
        case ITER_TIME:     output = "iteration time";     break; 
        case TOTAL_TIME:    output = "total time";         break; 
        case MATRIX_DIM:    output = "dimension";          break; 
        case MATRIX_NNZ:    output = "non-zeroes";         break; 
        case LEVELS:        output = "levels";             break;
        default:            output = "???";                break;
    }
    return output;
}

std::string long_string(preprocessing_type pt){
    std::string output;
    switch(pt){
        case NORMALIZE_COLUMNS:                      output = "NORMALIZE_COLUMNS";                      break;
        case NORMALIZE_ROWS:                         output = "NORMALIZE_ROWS";                         break;
#ifdef ILUPLUSPLUS_USES_SPARSPAK
        case REVERSE_CUTHILL_MCKEE_ORDERING:         output = "REVERSE_CUTHILL_MCKEE_ORDERING";         break;
#endif
        case PQ_ORDERING:                            output = "PQ_ORDERING";                            break;
        case DYN_AV_PQ_ORDERING:                     output = "DYN_AV_PQ_ORDERING";                     break;
        case SYMM_PQ:                                output = "SYMM_PQ";                                break;
        case  MAX_WEIGHTED_MATCHING_ORDERING:        output = "MAX_WEIGHTED_MATCHING_ORDERING";         break;  // I-Matrix
#ifdef ILUPLUSPLUS_USES_METIS
        case METIS_NODE_ND_ORDERING:                 output = "METIS_NODE_ND_ORDERING";                 break;
#endif
        case UNIT_OR_ZERO_DIAGONAL_SCALING:          output = "UNIT_OR_ZERO_DIAGONAL_SCALING";          break;
#ifdef ILUPLUSPLUS_USES_PARDISO
        case PARDISO_MAX_WEIGHTED_MATCHING_ORDERING: output = "PARDISO_MAX_WEIGHTED_MATCHING_ORDERING"; break;
#endif
        case SPARSE_FIRST_ORDERING:                  output = "SPARSE_FIRST_ORDERING";                  break;
        case SYMM_MOVE_CORNER_ORDERING:              output = "SYMM_MOVE_CORNER_ORDERING";              break;
        case SYMM_MOVE_CORNER_ORDERING_IM:           output = "SYMM_MOVE_CORNER_ORDERING_IM";           break;
        case SYMB_SYMM_MOVE_CORNER_ORDERING:         output = "SYMB_SYMM_MOVE_CORNER_ORDERING";         break;
        case SYMB_SYMM_MOVE_CORNER_ORDERING_IM:      output = "SYMB_SYMM_MOVE_CORNER_ORDERING_IM";      break;
        case SP_SYMM_MOVE_CORNER_ORDERING:           output = "SP_SYMM_MOVE_CORNER_ORDERING";           break;
        case SP_SYMM_MOVE_CORNER_ORDERING_IM:        output = "SP_SYMM_MOVE_CORNER_ORDERING_IM";        break;
        case WGT_SYMM_MOVE_CORNER_ORDERING:          output = "WGT_SYMM_MOVE_CORNER_ORDERING";          break;
        case WGT_SYMM_MOVE_CORNER_ORDERING_IM:       output = "WGT_SYMM_MOVE_CORNER_ORDERING_IM";       break;
        case WGT2_SYMM_MOVE_CORNER_ORDERING:         output = "WGT2_SYMM_MOVE_CORNER_ORDERING";         break;
        case WGT2_SYMM_MOVE_CORNER_ORDERING_IM:      output = "WGT2_SYMM_MOVE_CORNER_ORDERING_IM ";     break;
        case DD_SYMM_MOVE_CORNER_ORDERING_IM:        output = "DD_SYMM_MOVE_CORNER_ORDERING_IM";        break;
        default:                                     output = "???"; break;
    }
    return output;
}

//************************************************************************************************************************
//                                                                                                                       *
//         Needed global functions                                                                                       *
//                                                                                                                       *
//************************************************************************************************************************



template<class T> inline void switchnumbers(T& x, T& y)
     {T z=x; x=y; y=z;}

template<class T> inline void interchange(T& x, T& y)
     {T z=x; x=y; y=z;}

/*
template<class T> inline T min(T x, T y){
       if (x<y) return x; else return y;
  }
*/
/*
template<class T> inline T max(T x, T y){
       if (x<y) return y; else return x;
  }
*/
template<class T> inline T conj(T x){
       #ifdef DEBUG
           std::cout<<"template<class T> inline T conj(T x): using non-specialized template. This is not a good idea. Please write a specialized template for your data type."<<std::endl;
       #endif
       return x;
  }


template<> inline float conj(float x){
       return x;
  }

template<> inline double conj(double x){
       return x;
  }

template<> inline long double conj(long double x){
       return x;
  }

template<> inline std::complex<float>  conj(std::complex<float> x){
       return std::conj(x);
  }


template<> inline std::complex<double> conj(std::complex<double> x){
       return std::conj(x);
  }

template<> inline std::complex<long double> conj(std::complex<long double> x){
       return std::conj(x);
  }



template<class T> inline Real real(T x){
    return (Real) std::real(x);
}

template<> inline Real real(float x){
    return (Real) x;
}

template<> inline Real real(double x){
    return (Real) x;
}

template<> inline Real real(long double x){
    return (Real) x;
}

template<> inline Real real(std::complex<float> x){
    return (Real) std::real(x);
}

template<> inline Real real(std::complex<double> x){
    return (Real) std::real(x);
}

template<> inline Real real(std::complex<long double> x){
    return (Real) std::real(x);
}

inline float fabs(std::complex<float> x){
       return std::abs(x);
}

inline double fabs(std::complex<double> x){
       return std::abs(x);
}

inline long double fabs(std::complex<long double> x){
       return std::abs(x);
}

inline float fabs(float x){
       return std::fabs(x);
}

inline double fabs(double x){
       return std::fabs(x);
}

inline long double fabs(long double x){
       return std::fabs(x);
}

template<class T> inline Real absvalue_squared(T x){
       return fabs(x*conj(x));
  }

template<> inline Real absvalue_squared(float x){
       return (Real) x*x;
  }

template<> inline Real absvalue_squared(double x){
       return (Real) x*x;
  }

template<> inline Real absvalue_squared(long double x){
       return (Real) x*x;
  }

template<> inline Real absvalue_squared(std::complex<float> x){
       return (Real) norm(x);
  }

template<> inline Real absvalue_squared(std::complex<double> x){
       return (Real) norm(x);
  }

template<> inline Real absvalue_squared(std::complex<long double> x){
       return (Real) norm(x);
  }


template<class T> inline T sqr(T x){
       return x*x;
  }

inline void fatal_error(bool exp, const std::string message){
    if(exp){
        std::cerr << message <<std::endl;
        exit(1);
    }
  }

inline bool non_fatal_error(bool exp, const std::string message){
    if(exp){
        std::cerr << message <<std::endl;
        return true;
    } else return false;
  }

Integer bin(Integer n, Integer k){
      Integer b=1;
      Integer i;
      for(i=0;i<k;i++)  b *= n-i;
      for(i=1;i<=k;i++) b /= i;
      return b;
  }

std::string replace_underscore_with_backslash_underscore(std::string oldstring){
    std::string newstring="";
    unsigned int pos1=0;
    unsigned int pos2=0;
    while(pos2<oldstring.size()){
        pos2=oldstring.find("_",pos1);
        newstring += oldstring.substr(pos1, pos2-pos1);
        if(pos2<oldstring.size()) newstring += "\\_";
        pos1=pos2+1;
    }
    return newstring;
  }

std::string replace_underscore_with_double_backslash_underscore(std::string oldstring){
    std::string newstring="";
    unsigned int pos1=0;
    unsigned int pos2=0;
    while(pos2<oldstring.size()){
        pos2=oldstring.find("_",pos1);
        newstring += oldstring.substr(pos1, pos2-pos1);
        if(pos2<oldstring.size()) newstring += "\\\\_";
        pos1=pos2+1;
    }
    return newstring;
  }



template<class T> bool equal_to_zero(T t){
    return (fabs((Real) t) < COMPARE_EPS);
}

bool equal_to_zero(Real t){
    return (fabs(t) < COMPARE_EPS);
}


template<class T> bool equal(T x, T y){
    return (fabs((Real)(x-y)) < COMPARE_EPS);
}


bool equal(Real x, Real y){
    return (fabs(x-y)<COMPARE_EPS);
}

#ifndef ILUPLUSPLUS_USES_SPARSPAK

template<class T> T max(T x, T y){
    return ((x >= y) ? x: y);
}

Integer max(Integer x, Integer y){
    return ((x >= y) ? x: y);
}

Real max(Real x, Real y){
    return ((x >= y) ? x: y);
}

template<class T> T min(T x, T y){
    return ((x <= y) ? x: y);
}

Integer min(Integer x, Integer y){
    return ((x <= y) ? x: y);
}

Real min(Real x, Real y){
    return ((x <= y) ? x: y);
}


#endif


//************************************************************************************************************************
//                                                                                                                       *
//         The implementation of the class iluplusplus_error                                                             *
//                                                                                                                       *
//************************************************************************************************************************




iluplusplus_error::iluplusplus_error(){
    error=UNKNOWN_ERROR;
  }

iluplusplus_error::iluplusplus_error(error_type E){
    error = E;
  }

iluplusplus_error::~iluplusplus_error(){}

iluplusplus_error::iluplusplus_error(const iluplusplus_error& E){
      error = E.error;
  }

iluplusplus_error& iluplusplus_error::operator = (const iluplusplus_error& E){
      error = E.error;
      return *this;
  }

void iluplusplus_error::print() const {
    std::cerr<<"iluplusplus_error: "<<error_message()<<std::endl<<std::flush;
  }

std::string iluplusplus_error::error_message() const {
      std::string message;
      switch(error){
          case UNKNOWN_ERROR: 
              message = "unknown error";
          break;
          case INSUFFICIENT_MEMORY:
              message = "error allocating memory: insufficient memory";
          break;
          case INCOMPATIBLE_DIMENSIONS:
              message = "incompatible dimensions";
          break;
          case ARGUMENT_NOT_ALLOWED:
              message = "argument not allowed for this function";
          break;
          case FILE_ERROR:
              message = "error reading or writing a file";
          break;
          case OTHER_ERROR:
              message = "other error";
          break;
      }
      return message;
  }

error_type& iluplusplus_error::set(){
      return error;
  }

error_type iluplusplus_error::get() const{
      return error;
  }

error_type iluplusplus_error::read() const {
      return error;
  }


}

#endif
