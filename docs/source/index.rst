Documentation for ``ilupp``
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. py:module:: ilupp

The ``ilupp`` package provides various incomplete LU and Cholesky factorization
routines for sparse matrices. It is implemented in C++ based on Jan Mayer's ILU++
package and comes with convenient Python bindings which use Scipy sparse matrices.

.. note::
    This documentation describes only the Python bindings. The C++ interface is
    essentially equivalent and should be easy to figure out from the header
    files.

All preconditioner classes derive from the following base class:

.. autoclass:: ilupp._BaseWrapper


Preconditioners for symmetric and positive definite matrices
============================================================

.. autoclass:: ilupp.IChol0Preconditioner

.. autoclass:: ilupp.ICholTPreconditioner


Preconditioners for general matrices
====================================

.. autoclass:: ilupp.ILU0Preconditioner
.. autoclass:: ilupp.ILUTPreconditioner
.. autoclass:: ilupp.ILUTPPreconditioner
.. autoclass:: ilupp.ILUCPreconditioner
.. autoclass:: ilupp.ILUCPPreconditioner
.. autoclass:: ilupp.ILUppPreconditioner


Stand-alone factorization functions
===================================

If you do not want a preconditioner object that you can apply to a vector, you can instead
use these factorization functions which directly return the matrix factors of the
chosen factorization. The parameters have the same meaning as for the corresponding
preconditioner classes above.

Note that if you already have a preconditioner object, you can obtain the matrix
factors using the :func:`ilupp._BaseWrapper.factors` method.

.. autofunction:: ilupp.ichol0
.. autofunction:: ilupp.icholt
.. autofunction:: ilupp.ilu0
.. autofunction:: ilupp.ilut
.. autofunction:: ilupp.iluc

Solving linear systems
======================

The :func:`solve` function provides a convenient interface for setting up a
preconditioner and then solving the linear system using a Krylov subspace method.

.. autofunction:: ilupp.solve


ILU++ parameters
================

The class :class:`iluplusplus_precond_parameter` provides the ability to tune
advanced parameters for the multilevel ILU++ preconditioner. Some comments on
these options can be found in
`parameters.h <https://github.com/c-f-h/ilupp/blob/master/src/ilupp/parameters.h>`_.
Further details are given in the
`original publication <https://doi.org/10.1002/nla.554>`_.

The most important members are
:attr:`iluplusplus_precond_parameter.threshold`,
:attr:`iluplusplus_precond_parameter.fill_in`, and
:attr:`iluplusplus_precond_parameter.piv_tol`,
which have essentially the same meaning as in the functions above.
Various sets of default parameters can be chosen by the
:func:`iluplusplus_precond_parameter.default_configuration`
function; see
`the readme document <https://github.com/c-f-h/ilupp/>`_
for some comments on these.

Furthermore,
:attr:`iluplusplus_precond_parameter.PREPROCESSING`
is an instance of
:class:`preprocessing_sequence` (see below) and can be used
to choose various methods of reordering the matrix before
factorization.

For example::

    param = ilupp.iluplusplus_precond_parameter()
    param.default_configuration(10)
    param.PREPROCESSING.set_MAX_WEIGHTED_MATCHING_ORDERING_PQ()

will choose default configuration 10 and preprocess the matrix with a max
weighted matching ordering followed by a PQ ordering.

.. autoclass:: iluplusplus_precond_parameter
    :members:
    :undoc-members:

.. autoclass:: preprocessing_sequence
    :members:
    :undoc-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
