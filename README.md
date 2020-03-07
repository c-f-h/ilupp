# ilupp -- ILU algorithms for C++ and Python   [![Build Status](https://travis-ci.com/c-f-h/ilupp.svg?branch=master)](https://travis-ci.com/c-f-h/ilupp)

    Copyright (C) 2020 Clemens Hofreither
    ILU++ is Copyright (C) 2006 by Jan Mayer

**Install using pip:**

    $ pip install ilupp

[Read the documentation](https://ilupp.readthedocs.io/en/latest/)

This project provides C++ implementations and Python bindings for many incomplete LU
and incomplete Cholesky algorithms. It is based on the original ILU++ package
described in the publication

- Mayer, J. (2007), ILU++: A new software package for solving sparse linear
  systems with iterative methods. *Proc. Appl. Math. Mech., 7: 2020123-2020124.*
  [doi:10.1002/pamm.200700911](https://dx.doi.org/10.1002/pamm.200700911)

Compared to the original ILU++, this package has been significantly improved:

- Code cleaned up and modernized for C++11
- Extensive test suite
- Many critical bugs fixed
- Massive performance improvements (orders of magnitude for large matrices)
- Added several new factorizations, like ILU(0), IChol(0), and incomplete Cholesky
  with choosable fill-in and thresholding

It also contains the multilevel Crout ILU preconditioner described in
[(Mayer 2007)](https://doi.org/10.1002/nla.554).

The original ILU++ homepage is no longer available, although it can still be accessed
via archive.org:

https://web.archive.org/web/20101001133632/http://iamlasun8.mathematik.uni-karlsruhe.de/~ae04/iluplusplus_introduction.html

The original ILU++ had little documentation beyond some comments in the source code.
Nevertheless, the Python bindings provide a simple interface to either solve a
linear system in a black-box way, or to compute a preconditioner and apply it
to a vector.  In both cases, the matrix should be provided as a Scipy CSR or
CSC matrix.

Below there is a reproduction of the most relevant parts of the original homepage.

## Choosing the Right Combination of Preprocessing and Preconditioner

The standard multilevel preconditioner of ILU++ depends on a larger number of
parameters and it can be combined with many different preprocessing techniques.
An overview can be found in the files orderings.h and parameters.h (and the
respective implementation files). ILU++ offers several default configurations
of such combinations. These configurations have been successful for a large
number of problems. Hence, they are good place to start before experimenting
with the parameters yourself. For easy access, the default configurations are
numbered. Note that the different configurations for the preconditioner (i.e.
the multilevel incomplete LU factorization) are also numbered, so care must be
taken to distinguish between the number of the default configuration and the
number of the preconditioner which a particular default configuration uses in
combination with preprocessing.

### Default Configuration Number 0 (and 1000)

This is implementation is not the first choice of a general purpose solver and
is included mostly for compatibility with previous releases. For each level,
the rows and columns of the coefficient matrix are normalized. Subsequently, PQ
preprocessing is performed to improve diagonal dominance. The preconditioner
(factorization) used is number 0. It pivots by both rows and columns. Column
pivoting intends to avoid small pivots, whereas row interchanges are
implemented to promote better sparsity. The levels are terminated whenever it
appears that continued factorization would result in too much fill-in. Default
configuration 1000 uses the factorisation 1000 instead of 0, which calculates a
sparser Schur complement.

Advantages:
 - Good preconditioner, often quite sparse
 - Very fast setup times

Disadvantages:
 - Memory requirements for intermediate calculations are quite high

### Default Configuration Number 10 (and 1010)

This implementation uses the maximal weighted matching and scaling to produce
an I-matrix as preprocessing. The factorisation used is number 0, which
implements pivoting by columns (to avoid small pivots) and row permutation to
reduce fill-in. Levels are terminated whenever fill-in is too high. This is the
best combination overall in terms of total calculation time, but this comes at
fairly high memory costs for intermediate calculations. Default configuration
1010 uses the factorisation 1000 instead of 0, which calculates a sparser Schur
complement.

Advantages:
 - Good preconditioner, often quite sparse
 - Fast setup times

Disadvantages:
 - Memory requirements for intermediate calculations are quite high

### Default Configuration Number 11 (and 1011)

This implementation uses the same preprocessing as default configuration number
10 plus an additional symmetric reordering to produce an initial diagonally
dominant submatrix. Consequently, setup times are slightly longer. It uses a
different preconditioner (number 10) than the other default configurations. No
pivoting is performed at all, reducing the memory requirements for intermediate
calculations significantly. Even though this works quite well for most matrices
because of the preprocessing, the resulting preconditioner generally needs to
be somewhat more dense to be effective. Levels are terminated whenever a pivot
becomes too small (by absolute value). Default configuration 1011 uses the
factorisation 1010 instead of 10, which calculates a sparser Schur complement.

Advantages:
 - Good preconditioner
 - Little additional memory is needed for calculations

Disadvantages:
 - Longer setup times
 - Preconditioner often not as sparse

## Theoretical Background

The preconditioning used in ILU++ consists of the following steps:

1. Preprocessing
2. Partial factorization with dropping to ensure sparsity
3. Level termination and calculation of the Schur complement
4. Preconditioning of (an approximate) Schur complement using Steps 1 to 4 recursively

Because of 4), a preconditioner generally has several "levels", one
corresponding to each matrix (coefficient matrix or Schur complement) being
factored. Specifically, ILU++ does the following:

### Step 1)

Preprocessing consists of reordering and scaling of rows and columns to make
the coefficient matrix more suitable for incomplete factorization. The
preprocessing to make an I-matrix is best single method, but best results are
generally obtained by combining different preprocessing methods. In particular,
using additional methods to improve diagonal dominance of rows and columns
having low indices is generally favorable. Preprocessing methods which aim at
improving the matrix structurally without taking the elements themselves into
account (e.g. Reverse Cuthill-McKee, METIS, etc.) often result in little
further improvement for the preconditioners implemented in ILU++.

### Step 2)

The coefficient matrix A is factored using Crout's implementation of Gaussian
elimination, meaning that in the k-th step the k-th row of U and the k-th
column of L in the (incomplete) factorization A = LDU is calculated. (D is a
diagonal factor containing the pivots; L and U are unit lower and upper
triangular matrices.) Pivoting by columns can be used to avoid small pivots and
pivoting by rows can be used to eliminate those rows first resulting in the
least fill-in. Pivoting does, however, require substantially more memory to
perform the calculations. If little or no preprocessing is done to improve
diagonal dominance, then pivoting is essential for many matrices.

Dropping is performed by default by a rule to (heuristically) reduce the errors
in L and U and to reduce the propagation of errors in factorization. For many
matrices, even very sparse preconditioners result in convergent iterations. For
other dropping rules, such sparse preconditioners generally fail. For more
difficult matrices, ensuring small errors in the inverses of L and U is more
important. Hence, inverse-based dropping (as used for example in ILUPACK) is
also available. These preconditioners generally require more fill-in for
convergence for most matrices. However, for a few difficult matrices, this
dropping rule results in convergence, whereas the default dropping rule does
not.

### Step 3)

After the calculation of the k-th step has been completed, it is possible to
stop calculating L, D and U and to proceed to calculating (a sparse
approximation of) the Schur complement of A instead. Three possibilities for
terminating a level have been implemented:

 - stop whenever the preprocessing indicates
 - stop whenever fill-in becomes too high
 - stop whenever the absolute value of the pivot becomes too small

Some preprocessing techniques provide a natural point to terminate a level,
(e.g. PQ-reordering attempts to improve diagonal dominance and yields a row
index, upto which this was successful). Whenever pivoting by rows is performed,
rows are eliminated (heuristically) in the order "of increasing fill-in".
Hence, it is possible to terminate a level, whenever the expected fill-in
becomes too high. Finally, whenever no pivoting is performed then small pivots
(in absolute value) are a particular concern (even if preprocessing has
resulted in reasonably good diagonal dominance). In this situation, terminating
a level whenever the pivots become too small (in absolute value) works well.


## Further Details and Citing ILU++

Further details can be found in

Mayer, Jan: A Multilevel Crout ILU Preconditioner with Pivoting and Row Permutation. To appear in Numerical Linear Algebra with Applications.

Mayer, Jan: Symmetric Permutations for I-matrices to Delay and Avoid Small Pivots During Factorization. To appear in SIAM J. Sci. Comput.

and in the literature cited in these papers.

Preprints are available upon request from jan.mayer@mathematik.uni-karlsruhe.de

If you are using ILU++ for your scientific work, please cite these papers and the ILU++ website http://iamlasun8.mathematik.uni-karlsruhe.de/~ae04/iluplusplus.html.
