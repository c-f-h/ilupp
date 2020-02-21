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


#ifndef ILUPLUSPLUS_IMPLEMENTATINON_H
#define ILUPLUSPLUS_IMPLEMENTATINON_H


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

#include "iluplusplus_declarations.h"

#include "functions_implementation.h"
#include "arrays_implementation.h"
#include "function_class_implementation.h"
#include "orderings_implementation.h"
#include "parameters_implementation.h"
#include "pmwm_implementation.h"
#include "sparse_implementation.h"
#include "preconditioner_implementation.h"
#include "iterative_solvers_implementation.h"
#include "solving_routines_implementation.h"

#endif
