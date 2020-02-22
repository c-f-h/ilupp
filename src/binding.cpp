/*
ilupp -- Python bindings for ILU++
Copyright (C) 2020 Clemens Hofreither
ILU++ is Copyright (C) 2006 by Jan Mayer                                       *

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include <typeinfo>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ilupp/iluplusplus_interface.h"

namespace py = pybind11;

using namespace iluplusplus;

template <class T>
void check_is_1D_contiguous_array(const py::buffer_info& I, std::string name)
{
    if (I.ndim != 1)
        throw std::runtime_error("Expected 1D array for " + name + "!");

    if (I.format != py::format_descriptor<T>::format())
        throw std::runtime_error("Expected " + std::string(typeid(T).name()) + " array for " + name + "!");

    if (I.strides[0] != I.itemsize)
        throw std::runtime_error("Expected contiguous array for " + name + "!");
}

std::tuple<py::array_t<Real>, Integer, Real, Real>
solve(py::buffer A_data, py::buffer A_indices, py::buffer A_indptr, bool is_csr,
        py::buffer rhs, double rtol, double atol, int max_iter, iluplusplus_precond_parameter param)
{
    py::buffer_info A_data_info = A_data.request();
    check_is_1D_contiguous_array<Real>(A_data_info, "A_data");
    py::buffer_info A_indices_info = A_indices.request();
    check_is_1D_contiguous_array<Integer>(A_indices_info, "A_indices");
    py::buffer_info A_indptr_info = A_indptr.request();
    check_is_1D_contiguous_array<Integer>(A_indptr_info, "A_indptr");
    py::buffer_info rhs_info = rhs.request();
    check_is_1D_contiguous_array<Real>(rhs_info, "rhs");

    Integer nnz = A_indices_info.shape[0];
    Integer n = A_indptr_info.shape[0] - 1;

    if (A_indptr_info.shape[0] <= 1)
        throw std::runtime_error("matrix has size 0!");
    if (nnz != A_data_info.shape[0])
        throw std::runtime_error("indices and data should have the same size!");
    if (n != rhs_info.shape[0])
        throw std::runtime_error("right-hand side has wrong size!");

    Real rel_tol   = -std::log10(rtol);  // actual tolerance is 10^{-rel_tol}
    Real abs_tol   = -std::log10(atol);  // actual tolerance is 10^{-abs_tol}
    Real abs_error = 0;

    Real* pdata = (Real*)A_data_info.ptr;
    Integer* pindices = (Integer*)A_indices_info.ptr;
    Integer* pindptr = (Integer*)A_indptr_info.ptr;
    Real* prhs = (Real*)rhs_info.ptr;

    py::array_t<Real> result(n);
    py::buffer_info result_info = result.request();

    Integer n_x = n;
    Real* presult = (Real*)result_info.ptr;

    // only if exact solution is known
    Integer n_x_exact = 0;
    Real* p_xexact = 0;

    // call solver; parameters are:
    //          dimension n of matrix,
    //          nnz matrix,
    //          matrix orientation,
    //          matrix data (length nnz)
    //          matrix indices (length nnz)
    //          matrix row index pointers (length n+1)
    //          rhs (length n)
    //          n_x_exact (dimension n of x_exact),
    //          x_exact (input: exact solution, if known; otherwise vector of dimension 0 is exact solution is not known),
    //          n_x (dimension n of x),
    //          x (input: meaningless, output: solution),
    //          exact solution known (boolean),
    //          rel_tol   (input: stopping criterion (rel. reduction of residual required) output: relative reduction obtained),
    //          abs_tol   (input: stopping criterion (norm of residual required. output: residual obtained),
    //          max_iter  (input: max. iterations allowed. output: number of iterations needed)
    //          abs_error (output: error of solution, if exact solution known),
    //          working directory,
    //          matrix name,
    //          parameters for preprocessing and preconditioning
    // Note:  solve_with_multilevel_preconditioner returns -log10(rel_tol), -log10(abs_tol) achieved. As no exact solution is known, abs_error = nan

    const bool success = solve_with_multilevel_preconditioner(
            n, nnz, is_csr ? ROW : COLUMN,
            pdata, pindices, pindptr, prhs,
            n_x_exact, p_xexact,
            n_x, presult, false,
            rel_tol, abs_tol, max_iter, abs_error,
            "", "", param);

    if (success) {
        return std::make_tuple(result, max_iter, std::pow(10.0, -rel_tol), std::pow(10.0, -abs_tol));
    } else {
        throw std::runtime_error("did not converge");
    }
}

PYBIND11_MODULE(_ilupp, m)
{
    // optional module docstring
    m.doc() = "ILU++ library for incomplete LU factorization";

    m.def("solve", &solve, "Solve a linear system using ILU");

    py::class_<multilevel_preconditioner>(m, "multilevel_preconditioner")
        .def(py::init<>())
        .def("setup",
            [](multilevel_preconditioner& pr, py::buffer A_data, py::buffer A_indices, py::buffer A_indptr, bool is_csr, iluplusplus_precond_parameter param)
            {
                py::buffer_info A_data_info = A_data.request();
                check_is_1D_contiguous_array<Real>(A_data_info, "A_data");
                py::buffer_info A_indices_info = A_indices.request();
                check_is_1D_contiguous_array<Integer>(A_indices_info, "A_indices");
                py::buffer_info A_indptr_info = A_indptr.request();
                check_is_1D_contiguous_array<Integer>(A_indptr_info, "A_indptr");

                Integer nnz = A_indices_info.shape[0];
                Integer n = A_indptr_info.shape[0] - 1;

                if (A_indptr_info.shape[0] <= 1)
                    throw std::runtime_error("matrix has size 0!");
                if (nnz != A_data_info.shape[0])
                    throw std::runtime_error("indices and data should have the same size!");

                pr.setup((Real*)A_data_info.ptr, (Integer*)A_indices_info.ptr, (Integer*)A_indptr_info.ptr,
                    n, nnz, is_csr ? ROW : COLUMN, param);
            }
        )
        .def("apply",
            [](const multilevel_preconditioner& pr, py::buffer x)
            {
                py::buffer_info x_info = x.request();
                check_is_1D_contiguous_array<Real>(x_info, "x");
                if (x_info.shape[0] != pr.dim())
                    throw std::runtime_error("vector has wrong size for preconditioner!");
                pr.apply_preconditioner((Real*)x_info.ptr, x_info.shape[0]);
            }
        )
        .def_property_readonly("memory_used_calculations", &multilevel_preconditioner::memory_used_calculations)
        .def_property_readonly("memory_allocated_calculations", &multilevel_preconditioner::memory_allocated_calculations)
        .def_property_readonly("memory", &multilevel_preconditioner::memory)
        .def_property_readonly("exists", &multilevel_preconditioner::exists)
        .def_property_readonly("special_info", &multilevel_preconditioner::special_info)
        .def_property_readonly("total_nnz", &multilevel_preconditioner::total_nnz)
        .def("print_info", &multilevel_preconditioner::print_info)
        .def_property_readonly("dim", &multilevel_preconditioner::dim)
    ;

    py::class_<iluplusplus_precond_parameter>(m, "iluplusplus_precond_parameter")
        .def(py::init<>())
        .def("default_configuration", &iluplusplus_precond_parameter::default_configuration)
        //
        .def_property("fill_in", &iluplusplus_precond_parameter::get_fill_in, &iluplusplus_precond_parameter::set_fill_in)
        .def_property("threshold", &iluplusplus_precond_parameter::get_threshold, &iluplusplus_precond_parameter::set_threshold)
        .def_property("perm_tol", &iluplusplus_precond_parameter::get_perm_tol, &iluplusplus_precond_parameter::set_perm_tol)
        .def_property("GLOBAL_COMMENT", &iluplusplus_precond_parameter::get_GLOBAL_COMMENT, &iluplusplus_precond_parameter::set_GLOBAL_COMMENT)
        .def_property("PRECON_PARAMETER", &iluplusplus_precond_parameter::get_PRECON_PARAMETER, &iluplusplus_precond_parameter::set_PRECON_PARAMETER)
        .def_property("PREPROCESSING", &iluplusplus_precond_parameter::get_PREPROCESSING, &iluplusplus_precond_parameter::set_PREPROCESSING)
        .def_property("PQ_THRESHOLD", &iluplusplus_precond_parameter::get_PQ_THRESHOLD, &iluplusplus_precond_parameter::set_PQ_THRESHOLD)
        .def_property("PQ_ALGORITHM", &iluplusplus_precond_parameter::get_PQ_ALGORITHM, &iluplusplus_precond_parameter::set_PQ_ALGORITHM)
        .def_property("MAX_LEVELS", &iluplusplus_precond_parameter::get_MAX_LEVELS, &iluplusplus_precond_parameter::set_MAX_LEVELS)
        .def_property("MAX_FILLIN_IS_INF", &iluplusplus_precond_parameter::get_MAX_FILLIN_IS_INF, &iluplusplus_precond_parameter::set_MAX_FILLIN_IS_INF)
        .def_property("MEMORY_MAX_LEVELS", &iluplusplus_precond_parameter::get_MEMORY_MAX_LEVELS, &iluplusplus_precond_parameter::set_MEMORY_MAX_LEVELS)
        .def_property("BEGIN_TOTAL_PIV", &iluplusplus_precond_parameter::get_BEGIN_TOTAL_PIV, &iluplusplus_precond_parameter::set_BEGIN_TOTAL_PIV)
        .def_property("TOTAL_PIV", &iluplusplus_precond_parameter::get_TOTAL_PIV, &iluplusplus_precond_parameter::set_TOTAL_PIV)
        .def_property("MIN_ML_SIZE", &iluplusplus_precond_parameter::get_MIN_ML_SIZE, &iluplusplus_precond_parameter::set_MIN_ML_SIZE)
        .def_property("USE_FINAL_THRESHOLD", &iluplusplus_precond_parameter::get_USE_FINAL_THRESHOLD, &iluplusplus_precond_parameter::set_USE_FINAL_THRESHOLD)
        .def_property("FINAL_THRESHOLD", &iluplusplus_precond_parameter::get_FINAL_THRESHOLD, &iluplusplus_precond_parameter::set_FINAL_THRESHOLD)
        .def_property("VARY_THRESHOLD_FACTOR", &iluplusplus_precond_parameter::get_VARY_THRESHOLD_FACTOR, &iluplusplus_precond_parameter::set_VARY_THRESHOLD_FACTOR)
        .def_property("THRESHOLD_SHIFT_SCHUR", &iluplusplus_precond_parameter::get_THRESHOLD_SHIFT_SCHUR, &iluplusplus_precond_parameter::set_THRESHOLD_SHIFT_SCHUR)
        .def_property("PERMUTE_ROWS", &iluplusplus_precond_parameter::get_PERMUTE_ROWS, &iluplusplus_precond_parameter::set_PERMUTE_ROWS)
        .def_property("EXTERNAL_FINAL_ROW", &iluplusplus_precond_parameter::get_EXTERNAL_FINAL_ROW, &iluplusplus_precond_parameter::set_EXTERNAL_FINAL_ROW)
        .def_property("MIN_ELIM_FACTOR", &iluplusplus_precond_parameter::get_MIN_ELIM_FACTOR, &iluplusplus_precond_parameter::set_MIN_ELIM_FACTOR)
        .def_property("EXT_MIN_ELIM_FACTOR", &iluplusplus_precond_parameter::get_EXT_MIN_ELIM_FACTOR, &iluplusplus_precond_parameter::set_EXT_MIN_ELIM_FACTOR)
        .def_property("REQUIRE_ZERO_SCHUR", &iluplusplus_precond_parameter::get_REQUIRE_ZERO_SCHUR, &iluplusplus_precond_parameter::set_REQUIRE_ZERO_SCHUR)
        .def_property("REQ_ZERO_SCHUR_SIZE", &iluplusplus_precond_parameter::get_REQ_ZERO_SCHUR_SIZE, &iluplusplus_precond_parameter::set_REQ_ZERO_SCHUR_SIZE)
        .def_property("FINAL_ROW_CRIT", &iluplusplus_precond_parameter::get_FINAL_ROW_CRIT, &iluplusplus_precond_parameter::set_FINAL_ROW_CRIT)
        .def_property("SMALL_PIVOT_TERMINATES", &iluplusplus_precond_parameter::get_SMALL_PIVOT_TERMINATES, &iluplusplus_precond_parameter::set_SMALL_PIVOT_TERMINATES)
        .def_property("MIN_PIVOT", &iluplusplus_precond_parameter::get_MIN_PIVOT, &iluplusplus_precond_parameter::set_MIN_PIVOT)
        .def_property("USE_THRES_ZERO_SCHUR", &iluplusplus_precond_parameter::get_USE_THRES_ZERO_SCHUR, &iluplusplus_precond_parameter::set_USE_THRES_ZERO_SCHUR)
        .def_property("THRESHOLD_ZERO_SCHUR", &iluplusplus_precond_parameter::get_THRESHOLD_ZERO_SCHUR, &iluplusplus_precond_parameter::set_THRESHOLD_ZERO_SCHUR)
        .def_property("MIN_SIZE_ZERO_SCHUR", &iluplusplus_precond_parameter::get_MIN_SIZE_ZERO_SCHUR, &iluplusplus_precond_parameter::set_MIN_SIZE_ZERO_SCHUR)
        .def_property("ROW_U_MAX", &iluplusplus_precond_parameter::get_ROW_U_MAX, &iluplusplus_precond_parameter::set_ROW_U_MAX)
        .def_property("MOVE_LEVEL_FACTOR", &iluplusplus_precond_parameter::get_MOVE_LEVEL_FACTOR, &iluplusplus_precond_parameter::set_MOVE_LEVEL_FACTOR)
        .def_property("MOVE_LEVEL_THRESHOLD", &iluplusplus_precond_parameter::get_MOVE_LEVEL_THRESHOLD, &iluplusplus_precond_parameter::set_MOVE_LEVEL_THRESHOLD)
        .def_property("USE_MAX_AS_MOVE", &iluplusplus_precond_parameter::get_USE_MAX_AS_MOVE, &iluplusplus_precond_parameter::set_USE_MAX_AS_MOVE)
        .def_property("MEM_FACTOR", &iluplusplus_precond_parameter::get_MEM_FACTOR, &iluplusplus_precond_parameter::set_MEM_FACTOR)
        .def_property("VARIABLE_MEM", &iluplusplus_precond_parameter::get_VARIABLE_MEM, &iluplusplus_precond_parameter::set_VARIABLE_MEM)
        .def_property("USE_STANDARD_DROPPING", &iluplusplus_precond_parameter::get_USE_STANDARD_DROPPING, &iluplusplus_precond_parameter::set_USE_STANDARD_DROPPING)
        .def_property("USE_STANDARD_DROPPING2", &iluplusplus_precond_parameter::get_USE_STANDARD_DROPPING2, &iluplusplus_precond_parameter::set_USE_STANDARD_DROPPING2)
        .def_property("USE_INVERSE_DROPPING", &iluplusplus_precond_parameter::get_USE_INVERSE_DROPPING, &iluplusplus_precond_parameter::set_USE_INVERSE_DROPPING)
        .def_property("USE_WEIGHTED_DROPPING", &iluplusplus_precond_parameter::get_USE_WEIGHTED_DROPPING, &iluplusplus_precond_parameter::set_USE_WEIGHTED_DROPPING)
        .def_property("USE_WEIGHTED_DROPPING2", &iluplusplus_precond_parameter::get_USE_WEIGHTED_DROPPING2, &iluplusplus_precond_parameter::set_USE_WEIGHTED_DROPPING2)
        .def_property("USE_ERR_PROP_DROPPING", &iluplusplus_precond_parameter::get_USE_ERR_PROP_DROPPING, &iluplusplus_precond_parameter::set_USE_ERR_PROP_DROPPING)
        .def_property("USE_ERR_PROP_DROPPING2", &iluplusplus_precond_parameter::get_USE_ERR_PROP_DROPPING2, &iluplusplus_precond_parameter::set_USE_ERR_PROP_DROPPING2)
        .def_property("USE_PIVOT_DROPPING", &iluplusplus_precond_parameter::get_USE_PIVOT_DROPPING, &iluplusplus_precond_parameter::set_USE_PIVOT_DROPPING)
        .def_property("INIT_WEIGHTS_LU", &iluplusplus_precond_parameter::get_INIT_WEIGHTS_LU, &iluplusplus_precond_parameter::set_INIT_WEIGHTS_LU)
        .def_property("DROP_TYPE_L", &iluplusplus_precond_parameter::get_DROP_TYPE_L, &iluplusplus_precond_parameter::set_DROP_TYPE_L)
        .def_property("DROP_TYPE_U", &iluplusplus_precond_parameter::get_DROP_TYPE_U, &iluplusplus_precond_parameter::set_DROP_TYPE_U)
        .def_property("BANDWIDTH_MULTIPLIER", &iluplusplus_precond_parameter::get_BANDWIDTH_MULTIPLIER, &iluplusplus_precond_parameter::set_BANDWIDTH_MULTIPLIER)
        .def_property("BANDWIDTH_OFFSET", &iluplusplus_precond_parameter::get_BANDWIDTH_OFFSET, &iluplusplus_precond_parameter::set_BANDWIDTH_OFFSET)
        .def_property("SIZE_TABLE_POS_WEIGHTS", &iluplusplus_precond_parameter::get_SIZE_TABLE_POS_WEIGHTS, &iluplusplus_precond_parameter::set_SIZE_TABLE_POS_WEIGHTS)
        //.def_property("TABLE_POSITIONAL_WEIGHTS", &iluplusplus_precond_parameter::get_TABLE_POSITIONAL_WEIGHTS, &iluplusplus_precond_parameter::set_TABLE_POSITIONAL_WEIGHTS)
        .def_property("WEIGHT_TABLE_TYPE", &iluplusplus_precond_parameter::get_WEIGHT_TABLE_TYPE, &iluplusplus_precond_parameter::set_WEIGHT_TABLE_TYPE)
        .def_property("SCALE_WEIGHT_INVDIAG", &iluplusplus_precond_parameter::get_SCALE_WEIGHT_INVDIAG, &iluplusplus_precond_parameter::set_SCALE_WEIGHT_INVDIAG)
        .def_property("SCALE_WGT_MAXINVDIAG", &iluplusplus_precond_parameter::get_SCALE_WGT_MAXINVDIAG, &iluplusplus_precond_parameter::set_SCALE_WGT_MAXINVDIAG)
        .def_property("WEIGHT_STANDARD_DROP", &iluplusplus_precond_parameter::get_WEIGHT_STANDARD_DROP, &iluplusplus_precond_parameter::set_WEIGHT_STANDARD_DROP)
        .def_property("WEIGHT_STANDARD_DROP2", &iluplusplus_precond_parameter::get_WEIGHT_STANDARD_DROP2, &iluplusplus_precond_parameter::set_WEIGHT_STANDARD_DROP2)
        .def_property("WEIGHT_INVERSE_DROP", &iluplusplus_precond_parameter::get_WEIGHT_INVERSE_DROP, &iluplusplus_precond_parameter::set_WEIGHT_INVERSE_DROP)
        .def_property("WEIGHT_WEIGHTED_DROP", &iluplusplus_precond_parameter::get_WEIGHT_WEIGHTED_DROP, &iluplusplus_precond_parameter::set_WEIGHT_WEIGHTED_DROP)
        .def_property("WEIGHT_ERR_PROP_DROP", &iluplusplus_precond_parameter::get_WEIGHT_ERR_PROP_DROP, &iluplusplus_precond_parameter::set_WEIGHT_ERR_PROP_DROP)
        .def_property("WEIGHT_ERR_PROP_DROP2", &iluplusplus_precond_parameter::get_WEIGHT_ERR_PROP_DROP2, &iluplusplus_precond_parameter::set_WEIGHT_ERR_PROP_DROP2)
        .def_property("WEIGHT_PIVOT_DROP", &iluplusplus_precond_parameter::get_WEIGHT_PIVOT_DROP, &iluplusplus_precond_parameter::set_WEIGHT_PIVOT_DROP)
        .def_property("COMBINE_FACTOR", &iluplusplus_precond_parameter::get_COMBINE_FACTOR, &iluplusplus_precond_parameter::set_COMBINE_FACTOR)
        .def_property("NEUTRAL_ELEMENT", &iluplusplus_precond_parameter::get_NEUTRAL_ELEMENT, &iluplusplus_precond_parameter::set_NEUTRAL_ELEMENT)
        .def_property("MIN_WEIGHT", &iluplusplus_precond_parameter::get_MIN_WEIGHT, &iluplusplus_precond_parameter::set_MIN_WEIGHT)
        .def_property("WEIGHTED_DROPPING", &iluplusplus_precond_parameter::get_WEIGHTED_DROPPING, &iluplusplus_precond_parameter::set_WEIGHTED_DROPPING)
        .def_property("SUM_DROPPING", &iluplusplus_precond_parameter::get_SUM_DROPPING, &iluplusplus_precond_parameter::set_SUM_DROPPING)
        .def_property("USE_POS_COMPRESS", &iluplusplus_precond_parameter::get_USE_POS_COMPRESS, &iluplusplus_precond_parameter::set_USE_POS_COMPRESS)
        .def_property("POST_FACT_THRESHOLD", &iluplusplus_precond_parameter::get_POST_FACT_THRESHOLD, &iluplusplus_precond_parameter::set_POST_FACT_THRESHOLD)
        .def_property("SCHUR_COMPLEMENT", &iluplusplus_precond_parameter::get_SCHUR_COMPLEMENT, &iluplusplus_precond_parameter::set_SCHUR_COMPLEMENT)
        //
        .def("use_only_standard_dropping1", &iluplusplus_precond_parameter::use_only_standard_dropping1)
        .def("use_only_standard_dropping2", &iluplusplus_precond_parameter::use_only_standard_dropping2)
        .def("use_only_inverse_dropping", &iluplusplus_precond_parameter::use_only_inverse_dropping)
        .def("use_only_weighted_dropping1", &iluplusplus_precond_parameter::use_only_weighted_dropping1)
        .def("use_only_weighted_dropping2", &iluplusplus_precond_parameter::use_only_weighted_dropping2)
        .def("use_only_error_propagation_dropping1", &iluplusplus_precond_parameter::use_only_error_propagation_dropping1)
        .def("use_only_error_propagation_dropping2", &iluplusplus_precond_parameter::use_only_error_propagation_dropping2)
        .def("use_only_pivot_dropping", &iluplusplus_precond_parameter::use_only_pivot_dropping)
    ;
}
