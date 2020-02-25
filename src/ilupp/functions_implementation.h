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


std::string booltostring(bool b){
    return b ? "TRUE" : "FALSE";
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
    switch(t){
        case SUCC_SOLVE:    return "SUCC_SOLVE";
        case THRESHOLD:     return "THRESHOLD";
        case FILLIN:        return "FILLIN";
        case MEM_STORAGE:   return "MEM_STORAGE";
        case MEM_USED:      return "MEM_USED";
        case MEM_ALLOCATED: return "MEM_ALLOCATED";
        case ITERATIONS:    return "ITERATIONS";
        case ABS_ERROR:     return "ABS_ERROR";
        case REL_RESIDUAL:  return "REL_RESIDUAL";
        case ABS_RESIDUAL:  return "ABS_RESIDUAL";
        case SETUP_TIME:    return "SETUP_TIME";
        case ITER_TIME:     return "ITER_TIME";
        case TOTAL_TIME:    return "TOTAL_TIME";
        case MATRIX_DIM:    return "MATRIX_DIM";
        case MATRIX_NNZ:    return "MATRIX_NNZ";
        case LEVELS:        return "LEVELS";
        default:            return "???";
    }
}

std::string string(data_type t){
    switch(t){
        case SUCC_SOLVE:    return "successful solve";
        case THRESHOLD:     return "threshold";
        case FILLIN:        return "fill-in";
        case MEM_STORAGE:   return "memory for storage";
        case MEM_USED:      return "memory used";
        case MEM_ALLOCATED: return "memory allocated";
        case ITERATIONS:    return "iterations";
        case ABS_ERROR:     return "absolute error";
        case REL_RESIDUAL:  return "relative residual";
        case ABS_RESIDUAL:  return "absolute residual";
        case SETUP_TIME:    return "setup time";
        case ITER_TIME:     return "iteration time";
        case TOTAL_TIME:    return "total time";
        case MATRIX_DIM:    return "dimension";
        case MATRIX_NNZ:    return "non-zeroes";
        case LEVELS:        return "levels";
        default:            return "???";
    }
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
