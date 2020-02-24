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



#ifndef PARAMETERS_IMPLEMENATION_H
#define PARAMETERS_IMPLEMENATION_H

#include <string>
#include "parameters.h"

namespace iluplusplus {



//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: iluplusplus_precond_parameter                                                                      //
//                                                                                                                       //
//***********************************************************************************************************************//


iluplusplus_precond_parameter::iluplusplus_precond_parameter(){default_parameters();}

void iluplusplus_precond_parameter::set(Integer fi, Real th, Real pt){fill_in=fi; threshold=th; perm_tol=pt;}

std::string iluplusplus_precond_parameter::filename() const {
    std::string name;
    if(get_PRECON_PARAMETER()>=0){
        name = ("ML_"+std::to_string(get_PRECON_PARAMETER())+"_"+get_PREPROCESSING().string_with_hyphen()+".out");
    } else {
        #ifdef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1) name = "PARDISO.out";
        #endif
        #ifndef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1){
                std::cout<<"iluplusplus_precond_parameter::filename(): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
        #ifdef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2) name = "ILUPACK.out";
        #endif
        #ifndef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2){
                std::cout<<"iluplusplus_precond_parameter::filename(): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
    }
    return name;
}

std::string iluplusplus_precond_parameter::precondname() const {
    std::string name;
    if(get_PRECON_PARAMETER()>=0){
        name = ("ML "+std::to_string(get_PRECON_PARAMETER())+"-"+get_PREPROCESSING().string()+": "+convert_to_string());
    } else {
        #ifdef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1){
                name = "PARDISO";
            }
        #endif
        #ifndef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1){
                std::cout<<"iluplusplus_precond_parameter::precondname(): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
        #ifdef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2) name = "ILUPACK";
        #endif
        #ifndef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2){
                std::cout<<"iluplusplus_precond_parameter::precondname(): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
    }
    return name;
}

std::string iluplusplus_precond_parameter::filename(std::string matrix_name) const {
    std::string name;
    if(get_PRECON_PARAMETER()>=0){
         name = ("ML_"+std::to_string(get_PRECON_PARAMETER())+"_"+get_PREPROCESSING().string_with_hyphen()+"_"+matrix_name+".out");
    } else {
        #ifdef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1){
                name = ("PARDISO_"+matrix_name+".out");
            }
        #endif
        #ifndef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1){
                std::cout<<"iluplusplus_precond_parameter::filename(std::string matrix_name): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
        #ifdef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2){
                name = ("IP_"+matrix_name+".out");
            }
        #endif
        #ifndef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2){
                std::cout<<"iluplusplus_precond_parameter::filename(std::string matrix_name): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
    }
    return name;
}

std::string iluplusplus_precond_parameter::short_string() const {
    std::string name;
    if(get_PRECON_PARAMETER()>=0){
        name = (get_PREPROCESSING().string()+"+"+std::to_string(get_PRECON_PARAMETER()));
    } else {
        #ifdef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1){
                name = "PARDISO";
            }
        #endif
        #ifndef ILUPLUSPLUS_USES_PARDISO
            if(get_PRECON_PARAMETER()==-1){
                std::cout<<"iluplusplus_precond_parameter::short_string(): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
        #ifdef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2){
                name = "ILUPACK";
            }
        #endif
        #ifndef ILUPLUSPLUS_USES_ILUPACK
            if(get_PRECON_PARAMETER()==-2){
                std::cout<<"iluplusplus_precond_parameter::short_string(): PRECON_PARAMETER is negative. This is reserved for special uses. Returning empty string."<<std::endl;
                return name;
            }
        #endif
    }
    return name;
}

//## new parameter demands change here
void                     iluplusplus_precond_parameter::set_fill_in(Integer x){fill_in=x;}
void                     iluplusplus_precond_parameter::set_threshold(Real x){threshold=x;}
void                     iluplusplus_precond_parameter::set_perm_tol(Real x){perm_tol=x;}
void                     iluplusplus_precond_parameter::set_GLOBAL_COMMENT(std::string x){GLOBAL_COMMENT=x;}
void                     iluplusplus_precond_parameter::set_PRECON_PARAMETER(Integer x){PRECON_PARAMETER=x;}
void                     iluplusplus_precond_parameter::set_PREPROCESSING(preprocessing_sequence x){PREPRPOCESSING=x;}
void                     iluplusplus_precond_parameter::set_PQ_THRESHOLD(Real x){PQ_THRESHOLD=x;}
void                     iluplusplus_precond_parameter::set_PQ_ALGORITHM(Integer x){PQ_ALGORITHM=x;}
void                     iluplusplus_precond_parameter::set_MAX_LEVELS(Integer x){MAX_LEVELS=x;}
void                     iluplusplus_precond_parameter::set_MAX_FILLIN_IS_INF(bool x){MAX_FILLIN_IS_INF=x;}
void                     iluplusplus_precond_parameter::set_MEMORY_MAX_LEVELS(Integer x){MEMORY_MAX_LEVELS=x;}
void                     iluplusplus_precond_parameter::set_BEGIN_TOTAL_PIV(bool x){BEGIN_TOTAL_PIV=x;}
void                     iluplusplus_precond_parameter::set_TOTAL_PIV(Integer x){TOTAL_PIV=x;}
void                     iluplusplus_precond_parameter::set_MIN_ML_SIZE(Integer x){MIN_ML_SIZE=x;}
void                     iluplusplus_precond_parameter::set_USE_FINAL_THRESHOLD(bool x){ USE_FINAL_THRESHOLD=x;}
void                     iluplusplus_precond_parameter::set_FINAL_THRESHOLD(Real x){ FINAL_THRESHOLD=x;}
void                     iluplusplus_precond_parameter::set_VARY_THRESHOLD_FACTOR(Real x){VARY_THRESHOLD_FACTOR=x;}
void                     iluplusplus_precond_parameter::set_THRESHOLD_SHIFT_SCHUR(Real x){THRESHOLD_SHIFT_SCHUR=x;}
void                     iluplusplus_precond_parameter::set_PERMUTE_ROWS(Integer x){PERMUTE_ROWS=x;}
void                     iluplusplus_precond_parameter::set_EXTERNAL_FINAL_ROW(bool x){EXTERNAL_FINAL_ROW=x;}
void                     iluplusplus_precond_parameter::set_MIN_ELIM_FACTOR(Real x){MIN_ELIM_FACTOR=x;}
void                     iluplusplus_precond_parameter::set_REQUIRE_ZERO_SCHUR(bool x){REQUIRE_ZERO_SCHUR=x;}
void                     iluplusplus_precond_parameter::set_REQ_ZERO_SCHUR_SIZE(Integer x){REQ_ZERO_SCHUR_SIZE=x;}
void                     iluplusplus_precond_parameter::set_EXT_MIN_ELIM_FACTOR(Real x){EXT_MIN_ELIM_FACTOR=x;}
void                     iluplusplus_precond_parameter::set_FINAL_ROW_CRIT(Integer x){FINAL_ROW_CRIT=x;}
void                     iluplusplus_precond_parameter::set_SMALL_PIVOT_TERMINATES(bool x){SMALL_PIVOT_TERMINATES=x;}
void                     iluplusplus_precond_parameter::set_MIN_PIVOT(Real x){MIN_PIVOT=x;}
void                     iluplusplus_precond_parameter::set_USE_THRES_ZERO_SCHUR(bool x){USE_THRES_ZERO_SCHUR=x;}
void                     iluplusplus_precond_parameter::set_THRESHOLD_ZERO_SCHUR(Real x){THRESHOLD_ZERO_SCHUR=x;}
void                     iluplusplus_precond_parameter::set_MIN_SIZE_ZERO_SCHUR(Integer x){MIN_SIZE_ZERO_SCHUR=x;}
void                     iluplusplus_precond_parameter::set_ROW_U_MAX(Real x){ROW_U_MAX=x;}
void                     iluplusplus_precond_parameter::set_MOVE_LEVEL_FACTOR(Real x){MOVE_LEVEL_FACTOR=x;}
void                     iluplusplus_precond_parameter::set_MOVE_LEVEL_THRESHOLD(Real x){MOVE_LEVEL_THRESHOLD=x;}
void                     iluplusplus_precond_parameter::set_USE_MAX_AS_MOVE(bool x){USE_MAX_AS_MOVE=x;}
void                     iluplusplus_precond_parameter::set_MEM_FACTOR(Real x){MEM_FACTOR=x;}
void                     iluplusplus_precond_parameter::set_VARIABLE_MEM(Integer x){VARIABLE_MEM=x;}
void                     iluplusplus_precond_parameter::set_USE_STANDARD_DROPPING (bool x){USE_STANDARD_DROPPING=x;}
void                     iluplusplus_precond_parameter::set_USE_STANDARD_DROPPING2 (bool x){USE_STANDARD_DROPPING2=x;}
void                     iluplusplus_precond_parameter::set_USE_INVERSE_DROPPING (bool x){USE_INVERSE_DROPPING=x;}
void                     iluplusplus_precond_parameter::set_USE_WEIGHTED_DROPPING (bool x){USE_WEIGHTED_DROPPING=x;}
void                     iluplusplus_precond_parameter::set_USE_WEIGHTED_DROPPING2 (bool x){USE_WEIGHTED_DROPPING2=x;}
void                     iluplusplus_precond_parameter::set_USE_ERR_PROP_DROPPING (bool x){USE_ERR_PROP_DROPPING=x;}
void                     iluplusplus_precond_parameter::set_USE_ERR_PROP_DROPPING2 (bool x){USE_ERR_PROP_DROPPING2=x;}
void                     iluplusplus_precond_parameter::set_USE_PIVOT_DROPPING(bool x){USE_PIVOT_DROPPING=x;}
void                     iluplusplus_precond_parameter::set_INIT_WEIGHTS_LU (Real x){INIT_WEIGHTS_LU=x;}
void                     iluplusplus_precond_parameter::set_DROP_TYPE_L (Integer x){DROP_TYPE_L=x;}
void                     iluplusplus_precond_parameter::set_DROP_TYPE_U (Integer x){DROP_TYPE_U=x;}
void                     iluplusplus_precond_parameter::set_BANDWIDTH_MULTIPLIER (Real x){BANDWIDTH_MULTIPLIER=x;}
void                     iluplusplus_precond_parameter::set_BANDWIDTH_OFFSET (Integer x){BANDWIDTH_OFFSET=x;}
void                     iluplusplus_precond_parameter::set_SIZE_TABLE_POS_WEIGHTS (Integer x){SIZE_TABLE_POS_WEIGHTS=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_TABLE_TYPE (Integer x){WEIGHT_TABLE_TYPE=x;}
void                     iluplusplus_precond_parameter::set_SCALE_WEIGHT_INVDIAG (bool x){SCALE_WEIGHT_INVDIAG=x;}
void                     iluplusplus_precond_parameter::set_SCALE_WGT_MAXINVDIAG (bool x){SCALE_WGT_MAXINVDIAG=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_STANDARD_DROP (Real x){WEIGHT_STANDARD_DROP=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_STANDARD_DROP2 (Real x){WEIGHT_STANDARD_DROP2=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_INVERSE_DROP (Real x){WEIGHT_INVERSE_DROP=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_WEIGHTED_DROP (Real x){WEIGHT_WEIGHTED_DROP=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_ERR_PROP_DROP (Real x){WEIGHT_ERR_PROP_DROP=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_ERR_PROP_DROP2 (Real x){WEIGHT_ERR_PROP_DROP2=x;}
void                     iluplusplus_precond_parameter::set_WEIGHT_PIVOT_DROP (Real x){WEIGHT_PIVOT_DROP=x;}
void                     iluplusplus_precond_parameter::set_COMBINE_FACTOR (Integer x){COMBINE_FACTOR=x;}
void                     iluplusplus_precond_parameter::set_NEUTRAL_ELEMENT (Real x){NEUTRAL_ELEMENT=x;}
void                     iluplusplus_precond_parameter::set_MIN_WEIGHT (Real x){MIN_WEIGHT=x;}
void                     iluplusplus_precond_parameter::set_WEIGHTED_DROPPING (bool x){WEIGHTED_DROPPING=x;}
void                     iluplusplus_precond_parameter::set_SUM_DROPPING (bool x){SUM_DROPPING=x;}
void                     iluplusplus_precond_parameter::set_USE_POS_COMPRESS (bool x){USE_POS_COMPRESS=x;}
void                     iluplusplus_precond_parameter::set_POST_FACT_THRESHOLD (Real x){POST_FACT_THRESHOLD=x;} 
void                     iluplusplus_precond_parameter::set_SCHUR_COMPLEMENT (Integer x){SCHUR_COMPLEMENT=x;} 

void iluplusplus_precond_parameter::print() const {
        std::cout<<"Note: although all parameters are set by initialization, not all may actually be used or influence the calculation of the preconditioner. Particularly, some numerical values may have been voided by a flag."<<std::endl;
        std::cout<<"PRECON_PARAMETER:         "<<PRECON_PARAMETER<<std::endl;
        std::cout<<"GLOBAL_COMMENT:           "<<GLOBAL_COMMENT<<std::endl;
        std::cout<<"Preprocessing:"<<std::endl;
        get_PREPROCESSING().print();
        std::cout<<"Main Parameters:"<<std::endl;
        std::cout<<"   max fill-in:             "<<fill_in<<std::endl;
        std::cout<<"   threshold:               "<<threshold<<std::endl;
        std::cout<<"   perm_tol:                "<<perm_tol<<std::endl;
        std::cout<<"Restrictions on Memory and Calculation Times:"<<std::endl;
        std::cout<<"   MEMORY_MAX_LEVELS:       "<<MEMORY_MAX_LEVELS<<std::endl;
        std::cout<<"   MAX_LEVELS:              "<<MAX_LEVELS<<std::endl;
        std::cout<<"   MIN_ML_SIZE:             "<<MIN_ML_SIZE<<std::endl;
        std::cout<<"   MEM_FACTOR:              "<<MEM_FACTOR<<std::endl;
        std::cout<<"   VARIABLE_MEM:            "<<VARIABLE_MEM<<std::endl;
        std::cout<<"   MAX_FILLIN_IS_INF:       "<<booltostring(MAX_FILLIN_IS_INF)<<std::endl;
        std::cout<<"Row Permutation and Column Pivoting"<<std::endl;
        std::cout<<"   PERMUTE_ROWS:            "<<PERMUTE_ROWS<<std::endl;
        std::cout<<"   TOTAL_PIV:               "<<TOTAL_PIV<<std::endl;
        std::cout<<"   BEGIN_TOTAL_PIV:         "<<booltostring(BEGIN_TOTAL_PIV)<<std::endl;
        std::cout<<"Threshold Shifts"<<std::endl;
        std::cout<<"   USE_FINAL_THRESHOLD:     "<<booltostring(USE_FINAL_THRESHOLD)<<std::endl;
        std::cout<<"   VARY_THRESHOLD_FACTOR:   "<<VARY_THRESHOLD_FACTOR<<std::endl;
        std::cout<<"   THRESHOLD_SHIFT_SCHUR:   "<<THRESHOLD_SHIFT_SCHUR<<std::endl;
        std::cout<<"Post-Factorization Processing"<<std::endl;
        std::cout<<"   POST_FACT_THRESHOLD:     "<<POST_FACT_THRESHOLD<<std::endl;
        std::cout<<"Level Termination"<<std::endl;
        std::cout<<"   EXTERNAL_FINAL_ROW:      "<<booltostring(EXTERNAL_FINAL_ROW)<<std::endl;
        std::cout<<"   MIN_ELIM_FACTOR:         "<<MIN_ELIM_FACTOR<<std::endl;
        std::cout<<"   REQUIRE_ZERO_SCHUR:      "<<booltostring(REQUIRE_ZERO_SCHUR)<<std::endl;
        std::cout<<"   REQ_ZERO_SCHUR_SIZE:     "<<REQ_ZERO_SCHUR_SIZE<<std::endl;
        std::cout<<"   FINAL_ROW_CRIT:          "<<FINAL_ROW_CRIT<<std::endl;
        std::cout<<"   MIN_PIVOT;:              "<<MIN_PIVOT<<std::endl;
        std::cout<<"   ROW_U_MAX:               "<<ROW_U_MAX<<std::endl;
        std::cout<<"   MOVE_LEVEL_FACTOR:       "<<MOVE_LEVEL_FACTOR<<std::endl;
        std::cout<<"   MOVE_LEVEL_THRESHOLD:    "<<MOVE_LEVEL_THRESHOLD<<std::endl;
        std::cout<<"   USE_MAX_AS_MOVE:         "<<booltostring(USE_MAX_AS_MOVE)<<std::endl;
        std::cout<<"   SMALL_PIVOT_TERMINATES:  "<<booltostring(SMALL_PIVOT_TERMINATES)<<std::endl;
        std::cout<<"   MIN_PIVOT:               "<<MIN_PIVOT<<std::endl;
        std::cout<<"Dropping Rules:"<<std::endl;
        std::cout<<"   USE_STANDARD_DROPPING:   "<<booltostring(USE_STANDARD_DROPPING)<<std::endl;
        std::cout<<"   USE_STANDARD_DROPPING2:  "<<booltostring(USE_STANDARD_DROPPING2)<<std::endl;
        std::cout<<"   USE_INVERSE_DROPPING:    "<<booltostring(USE_INVERSE_DROPPING)<<std::endl;
        std::cout<<"   USE_WEIGHTED_DROPPING:   "<<booltostring(USE_WEIGHTED_DROPPING)<<std::endl;
        std::cout<<"   USE_WEIGHTED_DROPPING2:  "<<booltostring(USE_WEIGHTED_DROPPING2)<<std::endl;
        std::cout<<"   USE_ERR_PROP_DROPPING:   "<<booltostring(USE_ERR_PROP_DROPPING)<<std::endl;
        std::cout<<"   USE_ERR_PROP_DROPPING2:  "<<booltostring(USE_ERR_PROP_DROPPING2)<<std::endl;
        std::cout<<"   USE_PIVOT_DROPPING:      "<<booltostring(USE_PIVOT_DROPPING)<<std::endl;
        std::cout<<"   INIT_WEIGHTS_LU:         "<<INIT_WEIGHTS_LU<<std::endl;
        std::cout<<"   DROP_TYPE_L:             "<<DROP_TYPE_L<<std::endl;
        std::cout<<"   DROP_TYPE_U:             "<<DROP_TYPE_U<<std::endl;
        std::cout<<"   SCALE_WEIGHT_INVDIAG:    "<<booltostring(SCALE_WEIGHT_INVDIAG)<<std::endl;
        std::cout<<"   SCALE_WGT_MAXINVDIAG:    "<<booltostring(SCALE_WGT_MAXINVDIAG)<<std::endl;
        std::cout<<"Details on Dropping Rules"<<std::endl;
        std::cout<<"   WEIGHT_TABLE_TYPE:       "<<WEIGHT_TABLE_TYPE<<std::endl;
        std::cout<<"   WEIGHT_STANDARD_DROP:    "<<WEIGHT_STANDARD_DROP<<std::endl;
        std::cout<<"   WEIGHT_STANDARD_DROP2:   "<<WEIGHT_STANDARD_DROP2<<std::endl;
        std::cout<<"   WEIGHT_INVERSE_DROP:     "<<WEIGHT_INVERSE_DROP<<std::endl;
        std::cout<<"   WEIGHT_WEIGHTED_DROP:    "<<WEIGHT_WEIGHTED_DROP<<std::endl;
        std::cout<<"   WEIGHT_ERR_PROP_DROP:    "<<WEIGHT_ERR_PROP_DROP<<std::endl;
        std::cout<<"   WEIGHT_ERR_PROP_DROP2:   "<<WEIGHT_ERR_PROP_DROP2<<std::endl;
        std::cout<<"   WEIGHT_PIVOT_DROP:       "<<WEIGHT_PIVOT_DROP<<std::endl;
        std::cout<<"   COMBINE_FACTOR:          "<<COMBINE_FACTOR<<std::endl;
        std::cout<<"   NEUTRAL_ELEMENT:         "<<NEUTRAL_ELEMENT<<std::endl;
        std::cout<<"   MIN_WEIGHT:              "<<MIN_WEIGHT<<std::endl;
        std::cout<<"   WEIGHTED_DROPPING:       "<<booltostring(WEIGHTED_DROPPING)<<std::endl;
        std::cout<<"   SUM_DROPPING:            "<<booltostring(SUM_DROPPING)<<std::endl;
        std::cout<<"   USE_POS_COMPRESS:        "<<booltostring(USE_POS_COMPRESS)<<std::endl;
        std::cout<<"Bandwidth Approaches:"<<std::endl;
        std::cout<<"   BANDWIDTH_MULTIPLIER:    "<<BANDWIDTH_MULTIPLIER<<std::endl;
        std::cout<<"   BANDWIDTH_OFFSET:        "<<BANDWIDTH_OFFSET<<std::endl;
        std::cout<<"   SIZE_TABLE_POS_WEIGHTS:  "<<SIZE_TABLE_POS_WEIGHTS<<std::endl;
        std::cout<<"   SIZE_TABLE_POS_WEIGHTS:  "<<SIZE_TABLE_POS_WEIGHTS<<std::endl;
        std::cout<<"   TABLE_POSITIONAL_WEIGHTS:"; TABLE_POSITIONAL_WEIGHTS.print_info();
        std::cout<<"Details on Algorithms"<<std::endl;
        std::cout<<"   PQ_ALGORITHM:            "<<PQ_ALGORITHM<<std::endl;
        std::cout<<"   PQ_THRESHOLD:            "<<PQ_THRESHOLD<<std::endl;
        std::cout<<"Details on Schur Complement "<<std::endl;
        std::cout<<"   SCHUR_COMPLEMENT:        "<<SCHUR_COMPLEMENT<<std::endl;
        std::cout<<"   USE_THRES_ZERO_SCHUR:    "<<booltostring(USE_THRES_ZERO_SCHUR)<<std::endl;
        std::cout<<"   THRESHOLD_ZERO_SCHUR:    "<<THRESHOLD_ZERO_SCHUR<<std::endl;
        std::cout<<"   MIN_SIZE_ZERO_SCHUR:     "<<MIN_SIZE_ZERO_SCHUR<<std::endl;
//## new parameter demands change here
   }


void iluplusplus_precond_parameter::make_table(){
  try {
    TABLE_POSITIONAL_WEIGHTS.erase_resize_data_field(SIZE_TABLE_POS_WEIGHTS+1);
    Integer k;
    switch(WEIGHT_TABLE_TYPE){
        case 1:  for(k=0;k<=SIZE_TABLE_POS_WEIGHTS;k++) TABLE_POSITIONAL_WEIGHTS.set(k)=exp(5.0-(10.0/(Real) SIZE_TABLE_POS_WEIGHTS)*(Real) k); break;
        case 2:  for(k=0;k<=SIZE_TABLE_POS_WEIGHTS;k++) TABLE_POSITIONAL_WEIGHTS.set(k)=0.01*(1000.0-(1000.0/(Real) SIZE_TABLE_POS_WEIGHTS)*(Real) k); break;
        case 3:  for(k=0;k<=SIZE_TABLE_POS_WEIGHTS;k++) TABLE_POSITIONAL_WEIGHTS.set(k)=1.0; break;
        case 4:  for(k=0;k<=SIZE_TABLE_POS_WEIGHTS;k++) TABLE_POSITIONAL_WEIGHTS.set(k)=exp(2.0-(6.0/(Real) SIZE_TABLE_POS_WEIGHTS)*(Real) k); break;
        case 5:  for(k=0;k<=SIZE_TABLE_POS_WEIGHTS;k++) TABLE_POSITIONAL_WEIGHTS.set(k)=exp(4.0-(6.0/(Real) SIZE_TABLE_POS_WEIGHTS)*(Real) k); break;
        default: std::cerr<<"make_table: please use acceptable value for WEIGHT_TABLE_TYPE"<<std::endl; exit(1);
    }
  }
  catch(iluplusplus_error){
     std::cerr<<"iluplusplus_precond_parameter::make_table: Error allocating memory."<<std::endl;
     throw;
  }
}

void iluplusplus_precond_parameter::use_only_standard_dropping1(){
             USE_STANDARD_DROPPING = true;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = false;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = false;
         }

void iluplusplus_precond_parameter::use_only_standard_dropping2(){
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= true;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = false;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = false;
         }

void iluplusplus_precond_parameter::use_only_inverse_dropping(){
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = true;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = false;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = false;
         }

void iluplusplus_precond_parameter::use_only_weighted_dropping1(){
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = true;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = false;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = false;
         }

void iluplusplus_precond_parameter::use_only_weighted_dropping2(){
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= true;
             USE_ERR_PROP_DROPPING = false;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = false;
         }

void iluplusplus_precond_parameter::use_only_error_propagation_dropping1(){
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = true;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = false;
         }

void iluplusplus_precond_parameter::use_only_error_propagation_dropping2(){
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = false;
             USE_ERR_PROP_DROPPING2= true;
             USE_PIVOT_DROPPING    = false;
         }

void iluplusplus_precond_parameter::use_only_pivot_dropping(){
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = false;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = true;
         }

 void iluplusplus_precond_parameter::default_parameters(){
             fill_in               = 10000;
             threshold             = 1000.0;
             perm_tol              = 0.0;
             GLOBAL_COMMENT        = "default parameters";
             PRECON_PARAMETER      = 0;
             PQ_ALGORITHM          = 0;
             PQ_THRESHOLD          = 0.0;
             MAX_LEVELS            = 100;
             MEMORY_MAX_LEVELS     = 100;
             MAX_FILLIN_IS_INF     = true;
             BEGIN_TOTAL_PIV       = true;
             TOTAL_PIV             = 1;
             MIN_ML_SIZE           = 0;
             USE_FINAL_THRESHOLD   = false;
             FINAL_THRESHOLD       = 1000.0;
             VARY_THRESHOLD_FACTOR = 0.0;
             THRESHOLD_SHIFT_SCHUR = 1000.0;
             PERMUTE_ROWS          = 3;
             EXTERNAL_FINAL_ROW    = false;
             MIN_ELIM_FACTOR       = 0.5;
             REQUIRE_ZERO_SCHUR    = false;
             REQ_ZERO_SCHUR_SIZE   = 0;
             EXT_MIN_ELIM_FACTOR   = 0.0;
             FINAL_ROW_CRIT        = -1;
             SMALL_PIVOT_TERMINATES= false;
             MIN_PIVOT             = 1e-2;
             USE_THRES_ZERO_SCHUR  = false;
             THRESHOLD_ZERO_SCHUR  = 1e-6;
             MIN_SIZE_ZERO_SCHUR   = 100;
             ROW_U_MAX             = 1.5;
             MOVE_LEVEL_FACTOR     = 2.0;
             MOVE_LEVEL_THRESHOLD  = 10.0;
             USE_MAX_AS_MOVE       = true;
             MEM_FACTOR            = 3.0;
             VARIABLE_MEM          = 0;
             USE_STANDARD_DROPPING = false;
             USE_STANDARD_DROPPING2= false;
             USE_INVERSE_DROPPING  = false;
             USE_WEIGHTED_DROPPING = false;
             USE_WEIGHTED_DROPPING2= false;
             USE_ERR_PROP_DROPPING = true;
             USE_ERR_PROP_DROPPING2= false;
             USE_PIVOT_DROPPING    = false;
             INIT_WEIGHTS_LU       = 1.0;
             DROP_TYPE_L           = 0;
             DROP_TYPE_U           = 0;
             BANDWIDTH_MULTIPLIER  = 0.5;
             BANDWIDTH_OFFSET      = 0;
             SIZE_TABLE_POS_WEIGHTS= 100;
             WEIGHT_TABLE_TYPE     = 1;
             SCALE_WEIGHT_INVDIAG  = false;
             SCALE_WGT_MAXINVDIAG  = false;
             WEIGHT_STANDARD_DROP  = 1.0;
             WEIGHT_STANDARD_DROP2 = 1.0;
             WEIGHT_INVERSE_DROP   = 1.0;
             WEIGHT_WEIGHTED_DROP  = 1.0;
             WEIGHT_ERR_PROP_DROP  = 1.0;
             WEIGHT_ERR_PROP_DROP2 = 1.0;
             WEIGHT_PIVOT_DROP     = 1.0;
             COMBINE_FACTOR        = 0;
             NEUTRAL_ELEMENT       = 0.0;
             MIN_WEIGHT            = 1.0;
             WEIGHTED_DROPPING     = true;
             SUM_DROPPING          = false;
             USE_POS_COMPRESS      = false;
             POST_FACT_THRESHOLD   = 1000.0;
             SCHUR_COMPLEMENT      = 0;
             make_table();
             PREPRPOCESSING.set_PQ();
//## new parameter demands change here
}


std::string iluplusplus_precond_parameter::convert_to_string() const {
           std::ostringstream _fill_in;
           std::ostringstream _threshold;
           std::ostringstream _perm_tol;
           _threshold.setf(std::ios::right|std::ios::fixed);
           _perm_tol.setf(std::ios::right|std::ios::fixed);
           _threshold.precision(1);
           _perm_tol.precision(1);
           if(threshold<500.0)
               _threshold<<threshold;
           else
               _threshold<<"Inf";
           if(perm_tol<500.0)
               _perm_tol<<perm_tol;
           else
             _perm_tol<<"Inf";
           if(MAX_FILLIN_IS_INF)
             _fill_in<<"Inf";
           else
             _fill_in<<fill_in;
           return _fill_in.str()+"/"+_threshold.str()+"/"+_perm_tol.str();
   }



Real iluplusplus_precond_parameter::combine(Real x, Real y) const {
    switch(COMBINE_FACTOR){
        case 0: return max(x,y);
        case 1: return x+y;
        case 2: return x*y;
        case 3: return max(MIN_WEIGHT, max(x,y));
        default: return max(x,y);
    }
}



void iluplusplus_precond_parameter::default_configuration(Integer configuration = 0){
    preprocessing_sequence L;
    Integer precon_parameter;
    switch (configuration){
        case 0:
            L.set_PQ();
            precon_parameter = 0;
        break;
        case 1:
            L.set_PQ();
            precon_parameter = 10;
        break;
        case 10:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 0;
        break;
        case 11:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 10;
        break;
        case 12:
            L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 10;
        break;
        case 13:
            L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 10;
        break;
        #ifdef ILUPLUSPLUS_USES_PARDISO
        case 100:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 0;
        break;
        case 101:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 10;
        break;
        #endif
        case 1000:
            L.set_PQ();
            precon_parameter = 1000;
        break;
        case 1001:
            L.set_PQ();
            precon_parameter = 1010;
        break;
        case 1010:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1000;
        break;
        case 1011:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1010;
        break;
        case 1012:
            L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1010;
        break;
        case 1013:
            L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1010;
        break;
        #ifdef ILUPLUSPLUS_USES_PARDISO
        case 1100:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1000;
        break;
        case 1101:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1010;
        break;
        #endif


        // these are testing configurations. Many do not work well.
        case -1:  // testing configuration
            L.set_test_new();
            precon_parameter = 10;
        break;
        case -2:
            L.set_PQ();
            precon_parameter = 40;
        break;
        case -3:
            L.set_PQ();
            precon_parameter = 60;
        break;
        case -4:
            L.set_PQ();
            precon_parameter = 90;
        break;
        case -5:
            L.set_PQ();
            precon_parameter = 5;
        break;
        case -6:
            L.set_PQ();
            precon_parameter = 15;
        break;
        case -7:
            L.set_PQ();
            precon_parameter = 45;
        break;
        case -8:
            L.set_PQ();
            precon_parameter = 65;
        break;
        case -9:
            L.set_PQ();
            precon_parameter = 95;
        break;
        case -12:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 40;
        break;
        case -13:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 60;
        break;
        case -14:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 90;
        break;
        case -15:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 5;
        break;
        case -16:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 15;
        break;
        case -17:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 45;
        break;
        case -18:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 65;
        break;
        case -19:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 95;
        break;
        #ifdef ILUPLUSPLUS_USES_PARDISO
        case -102:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 40;
        break;
        case -103:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 60;
        break;
        case -104:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 90;
        break;
        case -105:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 5;
        break;
        case -106:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 15;
        break;
        case -107:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 45;
        break;
        case -108:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 65;
        break;
        case -109:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 95;
        break;
        #endif
        case -1002:
            L.set_PQ();
            precon_parameter = 1040;
        break;
        case -1003:
            L.set_PQ();
            precon_parameter = 1060;
        break;
        case -1004:
            L.set_PQ();
            precon_parameter = 1090;
        break;
        case -1005:
            L.set_PQ();
            precon_parameter = 105;
        break;
        case -1006:
            L.set_PQ();
            precon_parameter = 1015;
        break;
        case -1007:
            L.set_PQ();
            precon_parameter = 1045;
        break;
        case -1008:
            L.set_PQ();
            precon_parameter = 1065;
        break;
        case -1009:
            L.set_PQ();
            precon_parameter = 1095;
        break;
        case -1012:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1040;
        break;
        case -1013:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1060;
        break;
        case -1014:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1090;
        break;
        case -1015:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1005;
        break;
        case -1016:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1015;
        break;
        case -1017:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1045;
        break;
        case -1018:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1065;
        break;
        case -1019:
            L.set_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1095;
        break;
        #ifdef ILUPLUSPLUS_USES_PARDISO
        case -1100:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1000;
        break;
        case -1101:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1010;
        break;
        case -1102:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1040;
        break;
        case -1103:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1060;
        break;
        case -1104:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1090;
        break;
        case -1105:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1005;
        break;
        case -1106:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1015;
        break;
        case -1107:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1045;
        break;
        case -1108:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
            precon_parameter = 1065;
        break;
        case -1109:
            L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING();
            precon_parameter = 1095;
        break;
        #endif

        default:
            L.set_normalize();
            precon_parameter = 0;
        break;
    }
    init(L,precon_parameter,"");
}



void iluplusplus_precond_parameter::init(const preprocessing_sequence& L, Integer precon_parameter = 0, std::string global_comment = ""){
    default_parameters();
    PRECON_PARAMETER = precon_parameter;
    GLOBAL_COMMENT   = global_comment;
    PREPRPOCESSING = L;
    switch (PRECON_PARAMETER){
        #ifdef ILUPLUSPLUS_USES_ILUPACK
        case -2:    // set to ILUPACK
        break;
        #endif
        #ifndef ILUPLUSPLUS_USES_ILUPACK
        case -2:
            std::cout<<"iluplusplus_precond_parameter::init: precon_parameter = -2 is reserved for use with ILUPACK. Please install ILUPACK and adjust Makefile.in to use this option. Setting precon_parameter = 0."<<std::endl;
            PRECON_PARAMETER = 0;
        break;
        #endif
        #ifdef ILUPLUSPLUS_USES_PARDISO
        case -1:    // set to direct solver PARDISO
        break;
        #endif
        #ifndef ILUPLUSPLUS_USES_PARDISO
        case -1:
            std::cout<<"iluplusplus_precond_parameter::init: precon_parameter = -1 is reserved for use with PARDISO. Please install PARDISO and adjust Makefile.in to use this option. Setting precon_parameter = 0."<<std::endl;
            PRECON_PARAMETER = 0;
        break;
        #endif
        //  ******************************************************
        //  Multilevel Preconditioners
        //  ******************************************************
        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels based on preconditioner by fill-in. Preprocessing not essencial, I-matrix preprocessing beneficial
        case    0:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
        break;
        case    1:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case    2:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case    3:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case    5:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             SCHUR_COMPLEMENT      = 1;
        break;
        case    6:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;
        case    7:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;
        case    8:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;


        // multilevel ILUC, no pivoting or row permutation at all, level termination by pivot size -- needs I-matrix preprocessing and some moving to corner
        case   10:    //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   11:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   12:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   13:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   15:    //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   16:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   17:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   18:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;


        //   multilevel ILUC, no pivoting or row interchanges ever. No internal level termination. Requires high quality prior preprocessing and level termination provided externally, e.g. by PQ-preprocessing.
        case   20:  //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
        break;
        case   21:  //  multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case   22:  //  multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case   23:  //  multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case   25:  //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   26:  //  multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   27:  //  multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   28:  //  multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;

        //   multilevel ILUCP, pivoting by columns but no row interchanges ever. No internal level termination. Requires high quality prior preprocessing and level termination provided externally, e.g. by PQ-preprocessing.
        //   performance is generally not satisfactory, only for testing purposes.
        case   30:  //  ILUCP, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
        break;
        case   31:  //  ILUCP, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case   32:  //  ILUCP, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case   33:  //  ILUCP, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case   35:  //  ILUCP, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   36:  //  ILUCP, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   37:  //  ILUCP, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   38:  //  ILUCP, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
        break;

        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels whenever pivot becomes too small or fill-in too large. Preprocessing not essential, I-matrix preprocessing beneficial
        case   40:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   41:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   42:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   43:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
        break;
        case   45:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   46:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   47:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;
        case   48:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
        break;
        //   multilevel ILUCDP with row permutations and corresponding column as pivot, switching levels whenever fill-in too large.  I-matrix preprocessing almost essential
        
        case    50:  // multilevel ILUCDP, error-based dropping 
             perm_tol              = 1.0;
        break;
        case    51:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1.0;
        break;
        case    52:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1.0;
        break;
        case    53:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1.0;
        break;
        case    55:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
        break;
        case    56:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
        break;
        case    57:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
        break;
        case    58:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
        break;


        // +50 is same as before, except that USE_THRES_ZERO_SCHUR = true (requires SMALL_PIVOT_TERMINATES == true to make sense)
        // multilevel ILUC, no pivoting or row permutation at all, level termination by pivot size -- needs I-matrix preprocessing and some moving to corner
        case   60:    //  multilevel ILUC, error-based dropping, with Schur == 0 check, see 10 -- 18
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   61:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   62:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   63:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   65:    //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   66:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   67:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   68:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;

        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels whenever pivot becomes too small, with Schur == 0 check, see 40 -- 48
        // Preprocessing not essencial, I-matrix preprocessing beneficial; enforces a zero Schur Complement of dimension 1
        case   80:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        case   81:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        case   82:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        case   83:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        case   85:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        case   86:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        case   87:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        case   88:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             REQUIRE_ZERO_SCHUR    = true;
             REQ_ZERO_SCHUR_SIZE   = 1;
        break;
        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels whenever pivot becomes too small, with Schur == 0 check, see 40 -- 48
        // Preprocessing not essencial, I-matrix preprocessing beneficial
        case   90:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   91:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   92:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   93:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   95:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   96:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   97:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;
        case   98:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
        break;


        //  ******************************************************
        //  Single Level Preconditioners
        //  ******************************************************

        // single-level ILUCDP
        case 100:  // ILUCDP, error-based dropping
             MAX_LEVELS            = 1;
        break;
        case 101:  //  ILUCDP, inverse-based dropping
             MAX_LEVELS            = 1;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case  102:  //  ILUCDP, weighted-dropping
             MAX_LEVELS            = 1;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case  103:  //  ILUCDP, dual threshold dropping
             MAX_LEVELS            = 1;
             USE_ERR_PROP_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;

        // single-level ILUC
        case  110:  //  ILUC error-based dropping
             MAX_LEVELS            = 1;
             PERMUTE_ROWS          = 0;
             perm_tol              = 1000.0;
        break;
        case  111:  //  ILUC inverse-based dropping
             MAX_LEVELS            = 1;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             PERMUTE_ROWS          = 0;
             perm_tol              = 1000.0;
        break;
        case  112:  //  ILUC  weighted-dropping
             MAX_LEVELS            = 1;
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1000.0;
        break;
        case  113:  //  ILUC dual threshold dropping
             MAX_LEVELS            = 1;
             PERMUTE_ROWS          = 0;
             USE_ERR_PROP_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1000.0;
        break;

        // single-level ILUCP
        case  130:  //  ILUCP, error-based dropping
             MAX_LEVELS            = 1;
             PERMUTE_ROWS          = 0;
        break;
        case  131:  //  ILUCP, inverse-based dropping
             MAX_LEVELS            = 1;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             PERMUTE_ROWS          = 0;
        break;
        case  132:  //  ILUCP, weighted-dropping
             MAX_LEVELS            = 1;
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;
        case  133:  //  ILUCP, dual threshold dropping
             MAX_LEVELS            = 1;
             PERMUTE_ROWS          = 0;
             USE_ERR_PROP_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
        break;


        //  ******************************************************
        //  Multilevel Preconditioners
        //  ******************************************************
        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels based on preconditioner by fill-in. Preprocessing not essencial, I-matrix preprocessing beneficial
        //   different level termination criterion

        case  200:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             FINAL_ROW_CRIT        = -2;
        break;
        case  201:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             FINAL_ROW_CRIT        = -2;
        break;
        case  202:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             FINAL_ROW_CRIT        = -2;
        break;
        case  203:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             FINAL_ROW_CRIT        = -2;
        break;
        case  205:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
        break;
        case  206:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
        break;
        case  207:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
        break;
        case  208:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
        break;

        // ******************************************************
        // ******************************************************
        // ******************************************************
        // ****                                              ****
        // ****         same as above, but less memory       ****
        // ****                                              ****
        // ******************************************************
        // ******************************************************
        // ******************************************************


        case 1000:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1001:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1002:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1003:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1005:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1006:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1007:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1008:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;


        // multilevel ILUC, no pivoting or row permutation at all, level termination by pivot size -- needs I-matrix preprocessing and some moving to corner
        case 1010:    //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1011:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1012:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1013:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1015:    //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1016:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1017:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1018:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;


        //   multilevel ILUC, no pivoting or row interchanges ever. No internal level termination. Requires high quality prior preprocessing and level termination provided externally, e.g. by PQ-preprocessing.
        case 1020:  //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1021:  //  multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1022:  //  multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1023:  //  multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1025:  //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1026:  //  multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1027:  //  multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1028:  //  multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             perm_tol              = 1000.0;
             BEGIN_TOTAL_PIV       = false;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;

        //   multilevel ILUCP, pivoting by columns but no row interchanges ever. No internal level termination. Requires high quality prior preprocessing and level termination provided externally, e.g. by PQ-preprocessing.
        //   performance is generally not satisfactory, only for testing purposes.
        case 1030:  //  ILUCP, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1031:  //  ILUCP, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1032:  //  ILUCP, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1033:  //  ILUCP, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1035:  //  ILUCP, error-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1036:  //  ILUCP, inverse-based dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1037:  //  ILUCP, weighted-dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1038:  //  ILUCP, dual threshold dropping
             PERMUTE_ROWS          = 0;
             EXTERNAL_FINAL_ROW    = true;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;

        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels whenever pivot becomes too small or fill-in too large. Preprocessing not essential, I-matrix preprocessing beneficial
        case 1040:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1041:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1042:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1043:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1045:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1046:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1047:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1048:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1050:  // multilevel ILUCDP, error-based dropping 
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1051:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1052:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1053:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1055:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1056:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1057:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case  1058:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             perm_tol              = 1.0;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        // +50 is same as before, except that USE_THRES_ZERO_SCHUR = true (requires SMALL_PIVOT_TERMINATES == true to make sense)
        // multilevel ILUC, no pivoting or row permutation at all, level termination by pivot size -- needs I-matrix preprocessing and some moving to corner
        case 1060:    //  multilevel ILUC, error-based dropping, with Schur == 0 check, see 10 -- 18
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1061:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1062:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1063:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1065:    //  multilevel ILUC, error-based dropping
             PERMUTE_ROWS          = 0;
             TOTAL_PIV             = 0;
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1066:  // multilevel ILUC, inverse-based dropping
             PERMUTE_ROWS          = 0;
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1067:  // multilevel ILUC, weighted-dropping
             PERMUTE_ROWS          = 0;
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1068:  // multilevel ILUC, dual threshold dropping
             PERMUTE_ROWS          = 0;
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             TOTAL_PIV             = 0; 
             perm_tol              = 1000.0;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;

        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels whenever pivot becomes too small, with Schur == 0 check, see 40 -- 48
        // Preprocessing not essencial, I-matrix preprocessing beneficial
        case 1090:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1091:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1092:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1093:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1095:  // multilevel ILUCDP, error-based dropping
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1096:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1097:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1098:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SMALL_PIVOT_TERMINATES= true;
             MIN_ELIM_FACTOR       = 0.0;
             SCHUR_COMPLEMENT      = 1;
             USE_THRES_ZERO_SCHUR  = true;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;

        //  ******************************************************
        //  Multilevel Preconditioners
        //  ******************************************************
        //   multilevel ILUCDP with pivoting and row permutations (dual pivoting), switching levels based on preconditioner by fill-in. Preprocessing not essencial, I-matrix preprocessing beneficial
        //   different level termination criterion

        case 1200:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1201:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1202:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1203:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1205:  // multilevel ILUCDP, error-based dropping (DEFAULT preconditioner)
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1206:  // multilevel ILUCDP, inverse-based dropping
             USE_INVERSE_DROPPING  = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1207:  // multilevel ILUCDP, weighted-dropping
             USE_WEIGHTED_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
        case 1208:  // multilevel ILUCDP, dual threshold dropping
             USE_STANDARD_DROPPING = true;
             USE_ERR_PROP_DROPPING = false;
             SCHUR_COMPLEMENT      = 1;
             FINAL_ROW_CRIT        = -2;
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
        break;
  /*
             THRESHOLD_SHIFT_SCHUR = 3.0;
             MAX_FILLIN_IS_INF     = false;
             fill_in               = 500;
  */
        default:
            std::cerr<<"init: Please use a permissible global parameter."<<std::endl;
            exit(1);
        break;
    }
    switch(COMBINE_FACTOR){
        case 0: NEUTRAL_ELEMENT = 0.0; break;
        case 1: NEUTRAL_ELEMENT = 0.0; break;
        case 2: NEUTRAL_ELEMENT = 1.0; break;
        case 3: NEUTRAL_ELEMENT = 1.0; break;
    }
   if(MAX_LEVELS > MEMORY_MAX_LEVELS){
         std::cerr<<"MAX_LEVELS is "<<MAX_LEVELS<<" but must be less than or equal to "<<MEMORY_MAX_LEVELS<<std::endl;
         exit(1);
    }
    make_table();
    if(USE_WEIGHTED_DROPPING2){
        DROP_TYPE_L = 2;
        DROP_TYPE_U = 2;
    }
}



//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUCP Preconditioner:                                                                              //
//                                                                                                                       //
//***********************************************************************************************************************//

         ILUCP_precond_parameter::ILUCP_precond_parameter() : fill_in(0), threshold(1000.0), perm_tol(0.0), row_pos(-1){}
         ILUCP_precond_parameter::ILUCP_precond_parameter(Integer fi, Real th, Real pt, Integer rp = -1) : fill_in(fi), threshold(th), perm_tol(pt), row_pos(rp) {}
         ILUCP_precond_parameter::ILUCP_precond_parameter(const ILUCP_precond_parameter& p) {fill_in=p.fill_in; threshold=p.threshold; perm_tol=p.perm_tol;row_pos=p.row_pos;}
         ILUCP_precond_parameter& ILUCP_precond_parameter::operator =(const ILUCP_precond_parameter& p){fill_in=p.fill_in; threshold=p.threshold; perm_tol=p.perm_tol;row_pos=p.row_pos; return *this;}
         Integer ILUCP_precond_parameter::get_fill_in() const {return fill_in;}
         Real ILUCP_precond_parameter::get_threshold() const {return threshold;}
         Real ILUCP_precond_parameter::get_perm_tol() const {return perm_tol;}
         Integer ILUCP_precond_parameter::get_row_pos() const {return row_pos;}
         std::string ILUCP_precond_parameter::convert_to_string() const {
             std::ostringstream _fill_in;
             std::ostringstream _threshold;
             std::ostringstream _perm_tol;
             _threshold.precision(2);
             _perm_tol.precision(2);
             if(threshold<500.0)
                 _threshold<<threshold;
             else
                 _threshold<<"N";
             if(perm_tol<500.0)
                 _perm_tol<<perm_tol;
             else
                 _perm_tol<<"N";
             _fill_in<<fill_in;
             return _fill_in.str()+"-"+_threshold.str()+"-"+_perm_tol.str();
         }
         void ILUCP_precond_parameter::set(Integer fi, Real th, Real pt, Integer rp){fill_in=fi; threshold=th; perm_tol=pt;row_pos=rp;}
         void ILUCP_precond_parameter::set_row_pos(Integer rp){row_pos=rp;}
         void ILUCP_precond_parameter::set_threshold(Real th){threshold=th;}
         void ILUCP_precond_parameter::set_perm_tol(Real pt){perm_tol=pt;}


//***********************************************************************************************************************//
//                                                                                                                       //
//         The class: ILUCDP Preconditioner:                                                                             //
//                                                                                                                       //
//***********************************************************************************************************************//

         ILUCDP_precond_parameter::ILUCDP_precond_parameter() : fill_in(0), threshold(1000.0), perm_tol(0.0), begin_perm_row(-1) {}
         ILUCDP_precond_parameter::ILUCDP_precond_parameter(Integer fi, Real th, Real pt, Integer bpr =-1) : fill_in(fi), threshold(th), perm_tol(pt), begin_perm_row(bpr) {}
         ILUCDP_precond_parameter::ILUCDP_precond_parameter(const ILUCDP_precond_parameter& p) {fill_in=p.fill_in; threshold=p.threshold; perm_tol=p.perm_tol; begin_perm_row=p.begin_perm_row;}
         ILUCDP_precond_parameter& ILUCDP_precond_parameter::operator =(const ILUCDP_precond_parameter& p){fill_in=p.fill_in; threshold=p.threshold; perm_tol=p.perm_tol; begin_perm_row=p.begin_perm_row; return *this;}
         Integer ILUCDP_precond_parameter::get_fill_in() const {return fill_in;}
         Real ILUCDP_precond_parameter::get_threshold() const {return threshold;}
         Real ILUCDP_precond_parameter::get_perm_tol() const {return perm_tol;}
         Integer ILUCDP_precond_parameter::get_begin_perm_row() const {return begin_perm_row;}
         std::string ILUCDP_precond_parameter::convert_to_string() const {
             std::ostringstream _fill_in;
             std::ostringstream _threshold;
             std::ostringstream _perm_tol;
             _threshold.precision(2);
             _perm_tol.precision(2);
             if(threshold<500.0)
                 _threshold<<threshold;
             else
                 _threshold<<"N";
             if(perm_tol<500.0)
                 _perm_tol<<perm_tol;
             else
                 _perm_tol<<"N";
             _fill_in<<fill_in;
             return _fill_in.str()+"-"+_threshold.str()+"-"+_perm_tol.str();
         }
         void ILUCDP_precond_parameter::set(Integer fi, Real th, Real pt, Integer bpr = -1){fill_in=fi; threshold=th; perm_tol=pt; begin_perm_row = bpr;}
         void ILUCDP_precond_parameter::set_threshold(Real th){threshold=th;}
         void ILUCDP_precond_parameter::set_perm_tol(Real pt){perm_tol=pt;}
         void ILUCDP_precond_parameter::set_begin_perm_row(Integer pbr){begin_perm_row = pbr;}



} // end namespace iluplusplus

#endif
