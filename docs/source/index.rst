.. ilupp documentation master file, created by
   sphinx-quickstart on Wed Mar  4 16:07:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for ``ilupp``
===========================


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. contents::
    :local:

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
------------------------------------------------------------

.. autoclass:: ilupp.IChol0Preconditioner

.. autoclass:: ilupp.ICholTPreconditioner


Preconditioners for general matrices
------------------------------------

.. autoclass:: ilupp.ILU0Preconditioner
.. autoclass:: ilupp.ILUTPreconditioner
.. autoclass:: ilupp.ILUTPPreconditioner
.. autoclass:: ilupp.ILUCPreconditioner
.. autoclass:: ilupp.ILUCPPreconditioner
.. autoclass:: ilupp.ILUppPreconditioner


Stand-alone factorization functions
-----------------------------------

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
----------------------

The :func:`solve` function provides a convenient interface for setting up a
preconditioner and then solving the linear system using a Krylov subspace method.

.. autofunction:: ilupp.solve


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
