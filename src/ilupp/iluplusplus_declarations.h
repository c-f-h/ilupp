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


#ifndef ILUPLUSPLUS_DECLARATIONS_H
#define ILUPLUSPLUS_DECLARATIONS_H



// Although these flags can be set here, they should be set in the Makefile. Make changes there.
// These flags indicate which software is available on your computer.


//#define ILUPLUSPLUS_USES_SPARSPAK
//#define ILUPLUSPLUS_USES_METIS
//#define ILUPLUSPLUS_USES_PARDISO



// Further information for the flags above:
//

// ILUPLUSPLUS_USES_SPARSPAK: needed if Reverse-Cuthill McKee reordering is desired.
//                            In this case, you need to obtain sparspak.f and convert it to sparspak.cpp
//                            with the command "f2c -A -C++ sparspak.f", rename the output file as sparspak.cpp,
//                            and include cmath, and wrap source code as
//                            namespace iluplusplus { extern "C" { source code }}
//                            sparspak.cpp does not need to be compiled directly, as it will be included in ILU++ appropriately.
//                            Linking then requires the following flags (for gcc):
//                            g++ -lf2c -lm -Xlinker -defsym -Xlinker MAIN__=main

// ILUPLUSPLUS_USES_METIS:    needed if metis node nd reordering is desired.
//                            In this case, metis-4.0 needs to be installed.
//                            Restriction: the idxtype type of metis needs to be the same as the Long_Integer type of ILUPLUSPLUS:
//                            compiling and linking then requires the following flags (for gcc):
//                            compiling: -I <metis-4.0>
//                            linking:   -L <metis-4.0> -lmetis
// ILUPLUSPLUS_USES_PARDISO:  needed if PARDISO's I-matrix reordering and scaling is desired.
//                            In this case, the apppropriate pardiso_match library needs to be available and licensed.
//                            Compiling and linking requires lapack and blas to be installed and the following flags must be set (for gcc):
//                            -lf2c -lm -Xlinker -defsym -Xlinker MAIN__=main -lpardiso_match_GNU_IA32  -lgfortran -lf2c -lc  -llapack  -lblas




// Additional flags:

// For additional information on the calculations, particularly timing on matrix manipulations set:
// #define VERBOSE

// For timing results for iterative methods set:
// #define IT_TIME

// For even more detailed information set:
//#define VERYVERBOSE

// For information on solves, set:
//#define INFO

// For statistical informations, set:
//#define STATISTICS

// If index range checks are desired for most array accessing set
//#define DEBUG


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

#ifdef ILUPLUSPLUS_USES_SPARSPAK
#include "f2c.h"  // only needed for SPARSPAK
#endif

#include "declarations.h"
#include "arrays.h"
#include "functions.h"
#include "orderings.h"
#include "parameters.h"
#include "pmwm_declarations.h"
#include "sparse.h"
#include "preconditioner.h"
#include "iterative_solvers.h"
#include "solving_routines.h"


#endif
