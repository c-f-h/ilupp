#pragma once

////////////////////////////////////////////////////////////////////////////////
//
// NOTE:
//
// This code is currently unused. It's still kept around because it might be
// useful in the future.
//
////////////////////////////////////////////////////////////////////////////////

#include "declarations.h"

namespace iluplusplus {

//***********************************************************************************************************************
//                                                                                                                      *
//           The class matrix_dense                                                                                     *
//                                                                                                                      *
//***********************************************************************************************************************

template<class T>
class matrix_dense
{
private:
    Integer number_rows;
    Integer number_columns;
    T** data;
    void generic_matrix_vector_multiplication_addition(const vector_dense<T>& x, vector_dense<T>& v) const;      // v=v+(*this)*x, no error handling, hence private
    void generic_matrix_transpose_vector_multiplication_addition(const vector_dense<T>& x, vector_dense<T>& v) const;      // v=v+(*this)*x, no error handling, hence private
    void generic_matrix_matrix_multiplication_addition(const matrix_dense<T>& A, const matrix_dense<T>& B);  // *this = *this + A*B
    void pivotGJ(T **r, Integer k) const;
    Integer minusGJ(T **r, Integer k) const;
    Integer minus_invert(matrix_dense<T> &r, Integer k) const;
    void pivot_invert(matrix_dense<T> &r, Integer k) const;
public:
    // constructors, destructors
    matrix_dense();
    matrix_dense(Integer m, Integer n);
    matrix_dense(Integer m, Integer n, T d);
    matrix_dense(const matrix_dense& X); // copy-constructor
    matrix_dense(const matrix_sparse<T> &A);
    ~matrix_dense();
    void resize(Integer m, Integer n);
    // Basic functions
    void matrix_vector_multiplication_add(const vector_dense<T>& x, vector_dense<T>& v) const;             // v=v+(*this)*x
    void matrix_vector_multiplication(const vector_dense<T>& x, vector_dense<T>& v) const;                 // v=(*this)*x
    void matrix_transpose_vector_multiplication_add(const vector_dense<T>& x, vector_dense<T>& v) const;   // v=v+(*this^T)*x
    void matrix_transpose_vector_multiplication(const vector_dense<T>& x, vector_dense<T>& v) const;       // v=(*this^T)*x
    void matrix_matrix_multiplication_add(const matrix_dense<T>& A, const matrix_dense<T>& B);             // *this = *this + A*B
    void matrix_matrix_multiplication(const matrix_dense<T>& A, const matrix_dense<T>& B);                 // *this = A*B
    // matrix_dense valued operators
    matrix_dense  operator+ (const matrix_dense& X) const;
    matrix_dense  operator- (const matrix_dense& X) const;
    matrix_dense  operator* (const matrix_dense& X) const;
    matrix_dense  operator* (T k) const; // matrix-scalar multiplication
    matrix_dense& operator= (const matrix_dense& X);
    // vector_dense - and scalar-valued operators
    vector_dense<T> operator*(vector_dense<T> const & x) const; //matrix-vector-multiplication
    // matrix-valued functions
    matrix_dense<T> transpose() const;
    // Generating special matrices
    void set_all(T d);
    void diag(T d);
    void diag(const vector_dense<T>& d);
    // the diagonal matrix diag(1+eps1, 1+eps2,...,1+eps(rank),0,...0), |epsk|<eps_EV undergoes a similarity transform by I+U, normF(U)<eps_similarity
    matrix_dense<T>& scale_rows(const vector_dense<T>& d);
    matrix_dense<T>& scale_columns(const vector_dense<T>& d);
    matrix_dense<T>& inverse_scale_rows(const vector_dense<T>& d);
    matrix_dense<T>& inverse_scale_columns(const vector_dense<T>& d);
    matrix_dense<T> permute_rows(const index_list& perm) const; // applies perm to *this and returns permuted matrix
    matrix_dense<T> permute_columns(const index_list& perm) const;
    void permute_rows(const matrix_dense<T>& A, const index_list& perm); // applies perm to A and *this is permuted matrix.
    void permute_columns(const matrix_dense<T>& A, const index_list& perm);
    void overwrite(const matrix_dense& A, Integer m, Integer n);  // overwrites *this with A beginning at index (m,n)
    void elementwise_addition(const matrix_dense& A);
    void elementwise_subtraction(const matrix_dense& A);
    void elementwise_multiplication(const matrix_dense& A);
    void elementwise_division(const matrix_dense& A);
    // Functions, Information
    Integer rows() const;
    Integer columns() const;
    Real normF() const;
    Real norm1() const;
    // Accessing elements:
    T& operator()(Integer i, Integer j);
    const T& operator()(Integer i, Integer j) const;
    // Conversion
    matrix_sparse<T> compress(orientation_type o, double threshold = -1.0);
    friend matrix_dense<T> matrix_sparse<T>::expand() const;
    void expand(const matrix_sparse<T>& B); // *this = expanded B
    // solving systems of linear equations with row pivoting using Gauss-Jordan
    // this has been programmed fairly inefficiently and should not be used for "real" problems!
    void compress(Real threshold); // drops small elements whose absolute value is less than threshold
    void GaussJordan(const vector_dense<T> &b, vector_dense<T> &x) const;
    // this is the choice
    Integer Gauss(const vector_dense<T> &b, vector_dense<T> &x) const;
    bool solve(const vector_dense<T> &b, vector_dense<T> &x) const;
    void invert(const matrix_dense<T> &B);
    // for testing:
    bool ILUCP(const matrix_dense<T>& A, matrix_dense<T>& U, index_list& perm, Integer fill_in, Real tau, Integer& zero_pivots);
    bool square_check() const;
    Real memory() const;
}


//***********************************************************************************************************************
// Class matrix_dense: private functions                                                                                *
//***********************************************************************************************************************

template<class T> void matrix_dense<T>::generic_matrix_vector_multiplication_addition(const vector_dense<T>& x, vector_dense<T>& v) const {
     for(Integer i=0;i<number_columns;i++)
         for(Integer j=0;j<number_rows;j++) v.set_data(i) += data[i][j] * x.get_data(j);
  }

template<class T> void matrix_dense<T>::generic_matrix_transpose_vector_multiplication_addition(const vector_dense<T>& x, vector_dense<T>& v) const {
     for(Integer i=0;i<number_columns;i++)
         for(Integer j=0;j<number_rows;j++) v._set(j) += data[i][j] * x[i];
  }

template<class T> void matrix_dense<T>::generic_matrix_matrix_multiplication_addition(const matrix_dense<T>& A, const matrix_dense<T>& B) {
     for(Integer i=0;i<number_columns;i++)
         for(Integer j=0;j<number_rows;j++)
             for(Integer k=0; k<A.number_columns; k++) data[i][j] += A.data[i][k] * B.data[k][j];
  }


//***********************************************************************************************************************
// Class matrix_dense: constructors, destructors, etc.                                                                  *
//***********************************************************************************************************************

template<class T> matrix_dense<T>::matrix_dense(){
    number_columns = 0; number_rows = 0; data = 0;
}

template<class T> matrix_dense<T>::matrix_dense(Integer m, Integer n){
    number_columns = 0; number_rows = 0; data = 0;
    resize(m,n);
}

template<class T> matrix_dense<T>::matrix_dense(Integer m, Integer n, T d){
    number_columns = 0; number_rows = 0; data = 0;
    Integer i,j;
    resize(m,n);
    for(i=0;i<m;i++)
        for(j=0; j<n; j++) {
            data[i][j]=0;
        }
    for(i=0; i<min(m,n); i++) data[i][i]=d;
}

template<class T> matrix_dense<T>::matrix_dense(const matrix_dense& X){
    number_columns = 0; number_rows = 0; data = 0;
    Integer i,j;
    resize(X.number_rows,X.number_columns);
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++) data[i][j]=X.data[i][j];;
}

template<class T> matrix_dense<T>::matrix_dense(const matrix_sparse<T> &A) {
    number_columns = 0; number_rows = 0; data = 0;
    resize(A.rows(),A.columns());
    Integer i,j;
    for(i=0;i<A.rows();i++)
        for(j=0; j<A.columns(); j++)
            data[i][j]=0;
    if (A.orient() == ROW){
        for(i=0;i<A.read_pointer_size()-1;i++){
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++){
                data[i][A.read_index(j)]+=A.read_data(j);
            }
        }
    } else {
        for(i=0;i<A.read_pointer_size()-1;i++)
            for(j=A.read_pointer(i);j<A.read_pointer(i+1);j++)
                data[A.read_index(j)][i]+=A.read_data(j);
    }
}



template<class T> matrix_dense<T>::~matrix_dense() {
    if (data != 0){
        for(Integer i=0;i<number_rows;i++)
            if (data[i]!=0) delete[] data[i];
        delete[] data;
        data = 0;
    }
}


template<class T>  bool matrix_dense<T>::square_check() const {
    return (columns() == rows());
}

//***************************************************************************************************************************************
//  Class matrix_dense: basic functions                                                                                                 *
//***************************************************************************************************************************************

template<class T> void matrix_dense<T>::matrix_vector_multiplication_add(const vector_dense<T>& x, vector_dense<T>& v) const {
    if ((number_columns != x.dimension())||(number_rows != v.dimension())){
        std::cerr<<"matrix_dense:matrix_vector_multiplication_add(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else generic_matrix_vector_multiplication_addition(x,v);
}

template<class T> void matrix_dense<T>::matrix_vector_multiplication(const vector_dense<T>& x, vector_dense<T>& v) const {
    if (number_columns != x.dimension()){
        std::cerr << "matrix_dense:matrix_vector_multiplication(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else {
        v.resize(number_rows,0.0);
        generic_matrix_vector_multiplication_addition(x,v);
    }
}


template<class T> void matrix_dense<T>::matrix_matrix_multiplication_add(const matrix_dense<T>& A, const matrix_dense<T>& B) {
    if ((number_columns != B.number_columns)||(number_rows != A.number_rows) || (A.number_columns != B.number_rows)){
        std::cerr<<"matrix_dense:matrix_matrix_multiplication_add: Dimension error in matrix-matrix-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else generic_matrix_matrix_multiplication_addition(A,B);
}

template<class T> void matrix_dense<T>::matrix_matrix_multiplication(const matrix_dense<T>& A, const matrix_dense<T>& B){
    if (A.number_columns != B.number_rows){
        std::cerr<<"matrix_dense:matrix_matrix_multiplication: Dimension error in matrix-matrix-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else {
        resize(A.number_rows,B.number_columns);
        set_all(0.0);
        generic_matrix_matrix_multiplication_addition(A,B);
    }
}

template<class T> void matrix_dense<T>::matrix_transpose_vector_multiplication_add(const vector_dense<T>& x, vector_dense<T>& v) const {
     if ((number_columns != v.dimension())||(number_rows != x.dimension())){
         std::cerr << "matrix_dense:matrix_transpose_vector_multiplication_add(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
         throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     } else {
         generic_matrix__transpose_vector_multiplication_addition(x,v);
     }
  }

template<class T> void matrix_dense<T>::matrix_transpose_vector_multiplication(const vector_dense<T>& x, vector_dense<T>& v) const {
    if (number_rows != x.dimension() ){
        std::cerr << "matrix_dense:matrix_transpose_vector_multiplication(vector_dense, vector_dense): Dimension error in matrix-vector-multiplication"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    } else {
        v.resize(number_columns,0.0);
        generic_matrix_transpose_vector_multiplication_addition(x,v);
    }
}

//***************************************************************************************************************************************
//  Class matrix: matrix_dense valued operations                                                                                        *
//***************************************************************************************************************************************

template<class T> matrix_dense<T> matrix_dense<T>::operator*(T k) const {
    matrix_dense<T> Y(number_rows, number_columns,0.0);
    Integer i,j;
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++) Y.data[i][j]=k*data[i][j];
    return Y;
}

template<class T> matrix_dense<T> matrix_dense<T>::operator+ (const matrix_dense& X) const {
    if ((number_rows==X.number_rows)&&(number_columns==X.number_columns)){
        matrix_dense<T> Y(number_rows,number_columns,0.0);
        Integer i,j;
        for(i=0;i<number_rows;i++)
            for(j=0;j<number_columns;j++) Y.data[i][j]=data[i][j]+X.data[i][j];
        return Y;
    } else {
        std::cerr << "matrix_dense<T>::operator +: Dimensions error adding matrices."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> matrix_dense<T> matrix_dense<T>::operator- (const matrix_dense& X) const {
    if ((number_rows==X.number_rows)&&(number_columns==X.number_columns)){
        matrix_dense<T> Y(number_rows, number_columns,0.0);
        Integer i,j;
        for(i=0;i<number_rows;i++)
            for(j=0;j<number_columns;j++) Y.data[i][j]=data[i][j]-X.data[i][j];
        return Y;
    } else {
        std::cerr << "matrix_dense<T>::operator -: Dimensions error subtracting matrices."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> matrix_dense<T> matrix_dense<T>::operator*(const matrix_dense& X) const {
    matrix_dense<T> Y(number_rows, X.number_columns,0.0);
    Integer i,j,k;
    T summe;
    if (number_columns==X.number_rows){
        for(i=0;i<number_rows;i++)
            for(j=0;j<X.number_columns;j++){
                summe=0;
                for(k=0;k<number_columns;k++) summe+=data[i][k]*X.data[k][j];
                Y.data[i][j]=summe;
            }
        return Y;
    } else {
        std::cerr<<"matrix_dense<T>::operator *: Dimensions error multiplying matrices. The dimensions are: "<<std::endl<<"("<<
            number_rows<<"x"<<number_columns<<") und ("<<X.number_rows<<"x"<<
            X.number_columns<<")"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> matrix_dense<T>& matrix_dense<T>::operator= (const matrix_dense<T>& X){
    Integer i,j;
    if(this==&X) return *this;
    resize(X.number_rows,X.number_columns);
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++) data[i][j]=X.data[i][j];
    return *this;
}

template<class T> void matrix_dense<T>::resize(Integer m, Integer n){
    if(m<0) m = 0;
    if(n<0) n = 0;
    Integer i;
    if(m != number_rows || n != number_columns){
        if(data != 0){
            for(i=0;i<number_rows;i++)
                if (data[i] != 0){
                    delete[] data[i];
                    data[i] = 0;
                }
            delete[] data; data = 0;
        }
        number_rows=m;
        number_columns=n;
        if(number_rows == 0 || number_columns == 0){
            data = 0;
        } else {
            data = new T*[number_rows];
            for(i=0;i<number_rows;i++){
                data[i] = new T[number_columns];
            }
        }
    }
}


//***************************************************************************************************************************************
//  Class matrix_dense: Matrix-Vector-Multiplication                                                                                    *
//***************************************************************************************************************************************

template<class T> vector_dense<T> matrix_dense<T>::operator * (vector_dense<T> const & x) const {
    if (number_rows==x.dimension()){
        vector_dense<T> res(number_columns);
        generic_matrix_vector_multiplication_addition(x,res);
        return res;
    } else {
        std::cerr << "Dimension error in matrix_dense*vector_dense"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}


//***************************************************************************************************************************************
//  Class matrix_dense: Matrix-valued functions                                                                                         *
//***************************************************************************************************************************************

template<class T> matrix_dense<T> matrix_dense<T>::transpose() const {
    matrix_dense<T> y(number_columns, number_rows);
    Integer i,j;
    for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++)
            y.data[j][i]=data[i][j];
    return y;
}


//***************************************************************************************************************************************
//   Class matrix: Generation of special matrices                                                                                       *
//***************************************************************************************************************************************

template<class T> void matrix_dense<T>::set_all(T d){
    for(Integer i=0;i<number_rows;i++)
        for(Integer j=0;j<number_columns;j++)
            data[i][j]=d;
}

template<class T> void matrix_dense<T>::diag(T d){
    set_all(0.0);
    for(Integer i=0;i<number_rows;i++)
        data[i][i]=d;
}

template<class T> void matrix_dense<T>::diag(const vector_dense<T>& d){
    resize(d.dimension(),d.dimension());
    set_all(0.0);
    for(Integer i=0;i<number_rows;i++)
        data[i][i]=d[i];
}


template<class T> matrix_dense<T>& matrix_dense<T>::scale_rows(const vector_dense<T>& d){
    if (number_rows==d.dimension()){
        for(Integer i=0;i<number_rows;i++)
            for(Integer j=0;j<number_columns;j++)
                data[i][j]*=d[i];
        return *this;
    } else {
        std::cerr << "Dimension error in matrix_dense::scale_rows"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}
template<class T> matrix_dense<T>& matrix_dense<T>::scale_columns(const vector_dense<T>& d){
    if (number_columns==d.dimension()){
        for(Integer i=0;i<number_rows;i++)
            for(Integer j=0;j<number_columns;j++)
                data[i][j]*=d[j];
        return *this;
    } else {
        std::cerr << "Dimension error in matrix_dense*::scale_columns"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}

template<class T> matrix_dense<T>& matrix_dense<T>::inverse_scale_rows(const vector_dense<T>& d){
    if (number_rows==d.dimension()){
        for(Integer i=0;i<number_rows;i++)
            for(Integer j=0;j<number_columns;j++)
                data[i][j]/=d[i];
        return *this;
    } else {
        std::cerr << "Dimension error in matrix_dense::scale_rows"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}
template<class T> matrix_dense<T>& matrix_dense<T>::inverse_scale_columns(const vector_dense<T>& d){
    if (number_columns==d.dimension()){
        for(Integer i=0;i<number_rows;i++)
            for(Integer j=0;j<number_columns;j++)
                data[i][j]/=d[j];
        return *this;
    } else {
        std::cerr << "Dimension error in matrix_dense*::scale_columns"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}


template<class T> void matrix_dense<T>::permute_columns(const matrix_dense<T>& A, const index_list& perm){
    if (A.number_columns==perm.dimension()){
        resize(A.number_rows,A.number_columns);
        for(Integer i=0;i<A.number_rows;i++)
            for(Integer j=0;j<A.number_columns;j++)
                data[i][j] = A.data[i][perm[j]];
    } else {
        std::cerr << "Dimension error in matrix_dense::permute_columns"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}


template<class T> matrix_dense<T> matrix_dense<T>::permute_columns(const index_list& perm) const {
    matrix_dense<T> B;
    B.permute_columns(*this,perm);
    return B;
}


template<class T> void matrix_dense<T>::permute_rows(const matrix_dense<T>& A, const index_list& perm){
    if (A.number_rows==perm.dimension()){
        resize(A.number_rows,A.number_columns);
        for(Integer i=0;i<A.number_rows;i++)
            for(Integer j=0;j<A.number_columns;j++)
                data[i][j] = A.data[perm[i]][j];
    } else {
        std::cerr << "Dimension error in matrix_dense::permute_rows"<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
}


template<class T> matrix_dense<T> matrix_dense<T>::permute_rows(const index_list& perm) const {
    matrix_dense<T> B;
    B.permute_rows(*this,perm);
    return B;
}

template<class T> void matrix_dense<T>::elementwise_addition(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_addition: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] += A.data[i][j];
}

template<class T> void matrix_dense<T>::elementwise_subtraction(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_subtraction: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] -= A.data[i][j];
}

template<class T> void matrix_dense<T>::elementwise_multiplication(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_multiplication: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] *= A.data[i][j];
}

template<class T> void matrix_dense<T>::elementwise_division(const matrix_dense& A){
#ifdef DEBUG
    if(rows() != A.rows() || columns() != A.columns()){
        std::cerr<<"matrix_dense<T>::elementwise_division: dimensions incompatible."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
#endif
    Integer i,j;
    for(i=0;i<rows();i++)
        for(j=0;j<columns();j++)
            data[i][j] /= A.data[i][j];
}


template<class T> void matrix_dense<T>::overwrite(const matrix_dense& A, Integer m, Integer n){
    Integer i,j;
    if(m+A.number_rows>number_rows || n+A.number_columns>number_columns||m<0||n<0){
        std::cerr<<"matrix_dense::overwrite: This matrix does not fit as desired."<<std::endl;
        return;
    }
    for(i=0;i<A.number_rows;i++)
        for(j=0;j<A.number_columns;j++)
            data[m+i][n+j] = A.data[i][j];
}



//***********************************************************************************************************************
// Class matrix_dense: functions, information                                                                           *
//***********************************************************************************************************************

template<class T> Integer matrix_dense<T>::rows() const {
    return number_rows;
}

template<class T> Integer matrix_dense<T>::columns() const {
    return number_columns;
}

template<class T> Real matrix_dense<T>::normF() const {
    Real normsq=0.0;
    for(Integer i=0;i<number_rows;i++)
        for(Integer j=0;j<number_columns;j++)
            normsq=normsq+pow(data[i][j],2);
    return sqrt(normsq);
}

template<class T> Real matrix_dense<T>::norm1() const {
    Real norm=0.0;
    Real column_sum;
    for(Integer j=0;j<number_columns;j++){
        column_sum=0.0;
        for(Integer i=0;i<number_rows;i++) column_sum+=fabs(data[i][j]);
        if(column_sum>norm) norm=column_sum;
    }
    return norm;
}

template<class T> T& matrix_dense<T>::operator()(Integer i, Integer j){
#ifdef DEBUG
    if(0<=i&&i<number_rows&&0<=j&&j<number_columns)return data[i][j];
    else {std::cerr<<"matrix_dense(*,*): this matrix entry does not exist. Accessing Element ("<<i<<","<<j<<") of a ("<<number_rows<<","<<number_columns<<") matrix."<<std::endl; throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);}
#endif
    return data[i][j];
}

template<class T> const T& matrix_dense<T>::operator()(Integer i, Integer j) const {
    return const_cast<matrix_dense<T>&>(*this)(i, j);
}

//***********************************************************************************************************************
// Class matrix_dense: Conversion                                                                                       *
//***********************************************************************************************************************



template<class T> matrix_sparse<T> matrix_dense<T>::compress(orientation_type o, double threshold){
std::cerr<<"The use of this function is deprecated."<<std::endl;
std::cerr<<"Use member function matrix_sparse<T>::compress"<<std::endl;
std::cerr<<"Returning NULL matrix."<<std::endl;
matrix_sparse<T> M;
return M;
/*
     Integer counter=0;
     Integer i,j;
     for(i=0;i<number_rows;i++)
        for(j=0;j<number_columns;j++)
            if (fabs(data[i][j]) > threshold) counter++;
             M.reformat(number_rows, number_columns,counter,o);
     counter = 0;
     if(o == ROW){
         for(i=0;i<number_rows;i++){
             M.pointer[i]=counter;
             for(j=0;j<number_columns;j++)
                 if(fabs(data[i][j]) > threshold) {
                     M.indices[counter] = j;
                     M.data[counter] = data[i][j];
                     counter++;
             }
         }
         M.pointer[number_rows]=counter;
     } else {
         for(j=0;j<number_columns;j++){
             M.pointer[j]=counter;
             for(i=0;i<number_rows;i++)
                 if (fabs(data[i][j]) > threshold) {
                     M.indices[counter] = i;
                     M.data[counter] = data[i][j];
                     counter++;
                 }
         }
         M.pointer[number_columns]=counter;
     }
     return M;
*/
  }

template<class T> void matrix_dense<T>::expand(const matrix_sparse<T>& B) {
#ifdef DEBUG
    if(non_fatal_error(!B.check_consistency(),"matrix_dense::expand: matrix is inconsistent.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    Integer i;
    Integer j;
    resize(B.rows(),B.columns());
    set_all(0.0);
    if (B.orient() == ROW){
        for(i=0;i<B.get_pointer_size()-1;i++){ // std::cout<<"i schleife i="<<i<<"pointer[i]= "<<pointer[i]<< "pointer[i+1]= "<<pointer[i+1]<<std::endl;
            for(j=B.get_pointer(i);j<B.get_pointer(i+1);j++){
                data[i][B.get_index(j)]+=B.get_data(j);
            }
        }
    } else {
        for(i=0;i<B.get_pointer_size()-1;i++){
            for(j=B.get_pointer(i);j<B.get_pointer(i+1);j++){
                data[B.get_index(j)][i]+=B.get_data(j);
            }
        }
    }
}

template<class T>  void matrix_dense<T>::compress(Real threshold){
     for(Integer i=0;i<number_rows;i++)
         for(Integer j=0;j<number_columns;j++)
              if (fabs(data[i][j])<threshold) data[i][j] = (T) 0;
}

//***************************************************************************************************************************************
//   Class matrix_dense: Input, Output                                                                                                  *
//***************************************************************************************************************************************

template<class T> std::istream& operator >> (std::istream& is, matrix_dense<T>& X){
    std::cout<<"Matrix elements for the ("<<X.number_rows<<"x"<<X.number_columns<<")-Matrix:"<<std::endl;
    for(Integer i=0;i<X.number_rows;i++)
        for(Integer j=0;j<X.number_columns;j++)
            is >> X.data[i][j];
    std::cout<<"End >>"<<std::endl;
    return is;
}

template<class T>std::ostream& operator << (std::ostream& os, const matrix_dense<T>& x){
    os<<std::endl;
    for(Integer i=0;i<x.rows();i++){
        os <<"(";
        for(Integer j=0;j<x.columns();j++) os << std::setw(14) << x(i,j)<< "  ";
        os << " )" << std::endl;
    }
    os<<std::endl;
    if(x.rows() == 0) os<<"( )"<<std::endl;
    return os;
}

//***************************************************************************************************************************************
//   Class matrix_dense: Gauss-Jordan Elimination                                                                                       *
//***************************************************************************************************************************************


template<class T> void matrix_dense<T>::pivotGJ(T **r, Integer k) const {
    Integer size=number_rows;
    T help;
    Integer p, i;
    p=k;
    for (i=k+1;i<size;i++)
        if (fabs(r[p][k])<fabs(r[i][k])) p=i;
    if (p!=k)
        for (i=0;i<=size;i++){
            help=r[p][i];
            r[p][i]=r[k][i];
            r[k][i]=help;
        }
}


template<class T> Integer matrix_dense<T>::minusGJ(T **r, Integer k) const {
    Integer size=number_rows;
    Integer i, j;
    for (i=0;i<size;i++)
        if (i!=k)
            for (j=k+1;j<=size;j++){
                if (r[k][k]==0) return 1;
                r[i][j]=r[i][j]-r[i][k]/r[k][k]*r[k][j];
            }
    for (j=size;j>=k;j--)
        r[k][j]=r[k][j]/r[k][k];
    return 0;
}


template<class T> void matrix_dense<T>::GaussJordan(const vector_dense<T> &b, vector_dense<T> &x) const {
    if(non_fatal_error(number_rows != number_columns,"Gauss-Jordan requires a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer k, j, i;
    Integer errorcode=0;
    Integer size=number_rows;
    x.resize(size,0.0);
    if(size == 0) return;
    T **r = new  T* [size];
    for (i=0;i<size;i++){
        r[i]=new  T [size+1];
    }
    for (k=0;k<size;k++){
        for (j=0;j<size;j++)
            r[k][j]=data[k][j];
        r[k][size]=b[k];
    }
    for (k=0;k<size;k++){
        pivotGJ(r,k);
        errorcode=minusGJ(r,k);
        if (errorcode==1) {std::cerr<<"System is not solvable;"<<std::endl;};
    }
    for (k=0;k<size;k++)
        x[k]=r[k][size];
    for (i=0;i<size;i++)
        delete []r[i];
    delete []r;
}



template<class T> bool matrix_dense<T>::solve(const vector_dense<T> &b, vector_dense<T> &x) const {
    Gauss(b,x);
    return true;
}



template<class T> Integer matrix_dense<T>::minus_invert(matrix_dense<T> &r, Integer k) const {
    Integer size=r.number_rows;
    Integer i, j;
    for (i=0;i<size;i++)
        if (i!=k)
            for (j=k+1;j<2*size;j++){
                if (r.read(k,k)== (T) 0) return 1;
                r(i,j)-= r.read(i,k)/r.read(k,k)*r.read(k,j);
            }
    for (j=2*size-1;j>=k;j--)
        r(k,j)/=r.read(k,k);
    return 0;
}

template<class T> void matrix_dense<T>::pivot_invert(matrix_dense<T> &r, Integer k) const {
    Integer size=r.number_rows;
    T help;
    Integer p, i;
    p=k;
    for (i=k+1;i<size;i++)
        if (fabs(r.read(p,k))<fabs(r.read(i,k))) p=i;
    if (p!=k)
        for (i=0;i<2*size;i++){
            help=r.read(p,i);
            r(p,i)=r.read(k,i);
            r(k,i)=help;
        }
}

template<class T> void matrix_dense<T>::invert(const matrix_dense<T> &B){
    if(non_fatal_error(number_rows != number_columns,"Gauss-Jordan requires a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer k, j;
    Integer errorcode=0;
    Integer size=B.number_rows;
    if(B.number_rows != B.number_columns){
        std::cerr<<"Matrix is not square;"<<std::endl;
        return;
    }
    resize(size,size);
    matrix_dense<T> r(size,2*size);
    for (k=0;k<size;k++){
        for (j=0;j<size;j++){
            r(k,j)=B.read(k,j);
            r(k,j+size)=0.0;
        }
        r(k,k+size)=1.0;
    }
    for (k=0;k<size;k++){
        pivot_invert(r,k);
        errorcode=minus_invert(r,k);
        if (errorcode==1) {std::cerr<<"Matrix is not invertible;"<<std::endl;};
    }
    for (k=0;k<size;k++)
        for (j=0;j<size;j++)
            (*this)(k,j)=r.read(k,j+size);
}


template<class T> Integer matrix_dense<T>::Gauss(const vector_dense<T> &b, vector_dense<T> &x) const{
    if(non_fatal_error((rows()!=columns()),"matrix_dense::Gauss: matrix must be square")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error((rows()!=b.dimension()),"matrix_dense::Gauss: the dimension of the right hand side is incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n=rows();
    index_list permut(n);
    vector_dense<T> y(n);
    Integer i,j,k;
    Integer maxc;
    T maxv, newmaxv;
    T *swap;
    Integer iswap;
    x.erase_resize_data_field(n);
    // begin factorization
    // searching for pivot.
    for (i=0;i<n;i++) {
        maxc = i;
        maxv = fabs(data[i][i]);
        for (j = i + 1; j < n; j++)
            if ((newmaxv = fabs(data[j][i])) > maxv) {
                maxv = newmaxv;
                maxc = j;
            }
        // matrix is singular, if no pivot can be found
        if (maxv == 0.0) {
            std::cerr << "matrix_dense::Gauss: Matrix is singular." << std::endl
                << "A row of zeroes occurred in the " << i << "th step."<< std::endl;
            return 0;
        }
        // Swap rows
        iswap = permut[maxc];
        permut[maxc] = permut[i];
        permut[i] = iswap;

        swap = data[maxc];
        data[maxc] = data[i];
        data[i] = swap;
        // Factorize
        for (j=i+1;j<n;j++) {
            data[j][i] /= data[i][i];
            for (k = i + 1; k < n; k++)
                data[j][k] -= data[j][i] * data[i][k];
        }
    }
    // Solve system
    // Forward elimination
    y=b;
    for (i=0;i<n;i++)
        for (j=i+1;j<n;j++)
            y[permut[j]] -= data[j][i] * y[permut[i]];
    // Backward elimination
    for (i=n-1;i>=0;i--) {
        x[i] = y[permut[i]];
        for (j=i+1;j<n;j++)
            x[i]-=data[i][j] * x[j];
        x[i] /= data[i][i];
    }
    // return with success
    return 1;
}

template<class T> bool matrix_dense<T>::ILUCP(const matrix_dense<T>& A, matrix_dense<T>& U, index_list& perm, Integer fill_in, Real tau, Integer& zero_pivots){
    if(tau>500.0) tau=0.0;
    else tau=std::exp(-tau*std::log(10.0));
    if(non_fatal_error(!A.square_check(),"matrix_dense::ILUCP: A must be a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(!U.square_check(),"matrix_dense::ILUCP: U must be a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(!square_check(),"matrix_dense::ILUCP: *this must be a square matrix.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(A.columns() != columns() || U.columns() != columns(),"matrix_dense::ILUCP: Dimensions are incompatible.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    Integer n=A.rows();
    Integer k,i,j,p;
    zero_pivots=0;
    Real val_larg_el;
    Integer pos_larg_el;
    //Real norm_L, norm_U;
    vector_dense<T> w(n), z(n);
    vector_dense<bool> non_pivot(n);
    index_list inverse_perm(n);
    if(fill_in<1) fill_in=1;
    if(fill_in>n) fill_in=n;
    perm.resize(n);
    non_pivot.set_all(true);
    for(k=0;k<n;k++) for(i=0;i<n;i++) U.data[k][i]=0.0;
    for(k=0;k<n;k++) for(i=0;i<n;i++) data[k][i]=0.0;
    for(k=0;k<n;k++){
        z.set_all(0.0);
        w.set_all(0.0);
        for(i=0;i<n;i++) if(non_pivot[i]) z[i]=A.data[k][i];
        for(i=0;i<k;i++)
            for(j=0;j<n;j++)
                if(non_pivot[j]) z[j]-=data[k][i]*U.data[i][j];
        val_larg_el=abs(z[0]);
        pos_larg_el=0;
        for(i=1;i<n;i++)
            if(non_pivot[i])
                if(abs(z[i])>val_larg_el){
                    pos_larg_el=i;
                    val_larg_el=abs(z[i]);
                }
        if(val_larg_el==0.0){
            zero_pivots++;
            return false;
        }
        for(i=0;i<n;i++) if(non_pivot[i]) U.data[k][i]=z[i];
        p=inverse_perm[pos_larg_el];
        inverse_perm.switch_index(perm[k],pos_larg_el);
        perm.switch_index(k,p);
        non_pivot[pos_larg_el]=false;
        for(i=k+1;i<n;i++) w[i]=A.data[i][pos_larg_el];
        for(i=0;i<k;i++)
            for(j=k+1;j<n;j++)
                w[j]-=U.data[i][pos_larg_el]*data[j][i];
        for(i=k+1;i<n;i++) data[i][k] = w[i]/U.data[k][pos_larg_el];
        data[k][k]=1.0;
    }   // end for k.
    return true;
}

template<class T>  Real matrix_dense<T>::memory() const{
    return (Real) ((sizeof(T))* (Real) number_rows *(Real) number_columns) +  2*sizeof(Integer);
}


//***********************************************************************************************************************
//                                                                                                                      *
//           The class matrix_oriented                                                                                  *
//                                                                                                                      *
//***********************************************************************************************************************

template<class T> class matrix_oriented    // will hopefully eventually replace the class matrix_dense
{
private:
    Integer number_rows;
    Integer number_columns;
    Integer size;
    orientation_type orientation;
    T* data;
    void insert_data_along_orientation(const vector_dense<T>& data_vector,Integer k);
    // copies the indices from list into indices and the data stored in data_vector corresponding to the appropriate indices into data.
    int compare_by_absolute_value (const void * a, const void * b); // for z
public:
    // constructors, destructors
    matrix_oriented(orientation_type o=ROW);      // does not allocate memory, as nnz is unknown
    matrix_oriented(orientation_type o, Integer m, Integer n); // allocates memory
    matrix_oriented(const matrix_oriented& X);    // copy constructor
    ~matrix_oriented();
    matrix_oriented operator = (const matrix_oriented<T>& X);
    // Information
    Integer rows() const;                       // returns the number of rows.
    Integer columns() const;                    // returns the number of columns.
    Integer dim_along_orientation() const;      // returns the dimension in the direction of the orientation, i.e. the number of rows for row matrix and number of colums for a column matrix.
    Integer dim_against_orientation() const;    // returns the dimension in the direction opposite to the orientation, i.e. the number of colums for a row matrix and vice versa.
    T get_data(Integer i) const;
    void print_all() const;
    // functionality
    bool square_check() const;
    void resize(orientation_type o, Integer m, Integer n);
    void set_all(T d);
    // conversion
    void extract(const matrix_sparse<T> &A, Integer m, Integer n); // makes a matrix_oriented containing the rows/columns m to m+n-1 of A - determined by orientation.
    // output
    // should no longer be FRIEND std::ostream& operator << <> (std::ostream& os, const matrix_oriented<T>& x);
    // Other functions
    Real norm(Integer k) const;
    Real memory() const;
    // friend void index_list::quicksort_by_absolute_value(const T* values);
};


//***********************************************************************************************************************
// Class matrix_oriented: private functions                                                                             *
//***********************************************************************************************************************

template<class T> void matrix_oriented<T>::insert_data_along_orientation(const vector_dense<T>& data_vector,Integer k){
    Integer dim_al_or = (Integer)( matrix_oriented<T>::dim_along_orientation());
    Integer offset=k*dim_al_or;
    if(k+1 >= dim_al_or){
        std::cerr<<"matrix_oriented<T>::insert_data_along_orientation: the dimension "<<k<<" is too large for this matrix."<<std::endl;
        throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    }
    for(Integer i = 0; i < dim_al_or; i++){
        data[offset+i]=data_vector[i];
    }
  }


//***********************************************************************************************************************
// Class matrix_oriented: constructors, destructor                                                                      *
//***********************************************************************************************************************

template<class T> matrix_oriented<T>::matrix_oriented(orientation_type o){
     number_rows    = 0;
     number_columns = 0;
     size           = 0;
     orientation    = o;
     data           = 0;
}

template<class T> matrix_oriented<T>::matrix_oriented(orientation_type o, Integer m, Integer n){
    number_rows    = 0;
    number_columns = 0;
    size           = 0;
    orientation    = o;
    data           = 0;
    resize(o,m,n);
}

template<class T> matrix_oriented<T>::matrix_oriented(const matrix_oriented& X){
    number_rows    = 0;
    number_columns = 0;
    size           = 0;
    data           = 0;
    resize(X.orientation,X.number_rows,X.number_columns);
    Integer i;
    for (i=0;i<size;i++) data[i] = X.data[i];
}

template<class T> matrix_oriented<T>::~matrix_oriented() {
     if (data != 0) delete [] data; data=0;
  }

//***********************************************************************************************************************
// Class matrix_oriented: basic operators                                                                               *
//***********************************************************************************************************************

template<class T> matrix_oriented<T> matrix_oriented<T>::operator = (const matrix_oriented<T>& X){
    if(this==&X)
        return *this;
    resize(X.orientation,X.number_rows,X.number_columns);
    Integer i;
    for (i=0;i<matrix_oriented<T>::nnz;i++) data[i] = X.data[i];
    return *this;
}

//***********************************************************************************************************************
// Class matrix_oriented: basic information                                                                             *
//***********************************************************************************************************************

template<class T> Integer matrix_oriented<T>::rows() const {
     return number_rows;
  }

template<class T> Integer matrix_oriented<T>::columns() const{
     return number_columns;
  }

template<class T> T matrix_oriented<T>::get_data(Integer i) const{
     #ifdef DEBUG
         if(i<0 || i> size){
             std::cerr<<"matrix_oriented::get_data: index out of range."<<std::endl;
             throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         }
     #endif
     return data[i];
  }

template<class T> bool matrix_oriented<T>::square_check() const {
    return (number_rows == number_columns);
  }


template<class T> Integer matrix_oriented<T>::dim_along_orientation() const {
     if (orientation == ROW) return number_rows;
     else return number_columns;
  }

template<class T> Integer matrix_oriented<T>::dim_against_orientation() const {
     if (orientation == ROW) return number_columns;
     else return number_rows;
  }

template<class T> void matrix_oriented<T>::print_all() const {
    std::cout<<"A ("<<number_rows<<"x"<<number_columns<<") matrix containing "<<size<< " elements and having ";
    if (orientation == ROW) std::cout<<"ROW"; else std::cout<<"COLUMN";
    std::cout<<" orientation."<<std::endl<<"The elements are: "<<std::endl;
    for (Integer k=0; k<size; k++) std::cout<<data[k]<<" ";
    std::cout<<std::endl;
  }

//***********************************************************************************************************************
// Class matrix_oriented: basic functions                                                                               *
//***********************************************************************************************************************

template<class T> void matrix_oriented<T>::set_all(T d){
    for(Integer i = 0; i < size; i++) data[i]=d;
  }

template<class T> void matrix_oriented<T>::resize(orientation_type o, Integer m, Integer n){
    if(m<0) m=0;
    if(n<0) n=0;
    Integer newsize = ((Integer)(m))*((Integer)(n));
    if (size != newsize) {
        size = newsize;
        if (data    != 0) delete [] data;
        if (newsize == 0){
            data = 0;
        } else {
            data = new T[newsize];
        }
    }
    number_rows = m;
    number_columns = n;
    orientation = o;
}

//***********************************************************************************************************************
// Class matrix_oriented: accessing data                                                                                *
//***********************************************************************************************************************

template<class T> void matrix_oriented<T>::extract(const matrix_sparse<T> &A, Integer m, Integer n){
    if(non_fatal_error(n == 0,"matrix_oriented::extract: a positive number must be specified to be extracted.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if(non_fatal_error(m+n > A.dim_against_orientation(),"matrix_oriented::extract: the rows/columns to be extracted do not exist." )) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
    if (A.orient() == ROW)
        resize(ROW, n, A.columns());
    else
        resize(COLUMN, A.rows(), n);
    Integer k,i;
    Integer offset;
    set_all(0.0);
    for(k = 0; k < n; k++){
        offset = ((Integer)(k))*((Integer)(dim_against_orientation()));
        for(i=A.get_pointer(k+m); i< A.get_pointer(k+m+1); i++){
            data[ offset + ((Integer)(A.get_index(i))) ]  = A.get_data(i);
        }
    }
}

//***********************************************************************************************************************
// Class matrix_oriented: input/output                                                                                  *
//***********************************************************************************************************************

template<class T> std::ostream& operator << (std::ostream& os, const matrix_oriented<T> & x){
     Integer i_data;
     os<<"The matrix has "<<x.rows()<<" rows and "<<x.columns()<<" columns."<<std::endl;
     if(x.orient() == ROW){
         for(Integer i_row=0;i_row<x.rows();i_row++){
             os<<"*** row: "<<i_row<<" ***"<<std::endl;
             for(i_data = i_row*x.columns(); i_data<(i_row+1)*x.columns();i_data++)
                  os<<" "<<x.get_data(i_data)<<" ";
             std::cout<<std::endl;
         }
     }
     else
         for(Integer i_column=0;i_column<x.columns();i_column++){
             os<<"*** column: "<<i_column<<" ***"<<std::endl;
             for(i_data=i_column*x.rows();i_data<(i_column+1)*x.rows();i_data++)
                 os<<x.get_data(i_data)<<" ";
             std::cout<<std::endl;
         }
     return os;
  }

//***********************************************************************************************************************
// Class matrix_oriented: other functions                                                                               *
//***********************************************************************************************************************



template<class T> Real matrix_oriented<T>::norm(Integer k) const{
    Real norm_squared = 0.0;
    Integer offset = ((Integer)(k)) * ( (Integer)(dim_against_orientation()) );
    for(Integer j = 0; j < dim_against_orientation(); j++){
        norm_squared += absvalue_squared(data[offset+j]);
    }
    return sqrt(norm_squared);
  }


template<class T>  Real matrix_oriented<T>::memory() const{
    return (Real) ((sizeof(T))* (Real) number_rows *(Real) number_columns) +  4*sizeof(Integer);
}



//***********************************************************************************************************************
// Helper functions
//***********************************************************************************************************************

template<class T> matrix_dense<T> matrix_sparse<T>::expand() const {
#ifdef DEBUG
    if(non_fatal_error(!check_consistency(),"matrix_sparse::expand(): matrix is inconsistent.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
#endif
    Integer i;
    Integer j;
    matrix_dense<T> A;
    A.resize(number_rows,number_columns);
    A.set_all(0.0);
    if (orientation == ROW){
        for(i=0;i<pointer_size-1;i++){ // std::cout<<"i schleife i="<<i<<"pointer[i]= "<<pointer[i]<< "pointer[i+1]= "<<pointer[i+1]<<std::endl;
            for(j=pointer[i];j<pointer[i+1];j++){
                //std::cout<<"expand:j scheife "<<i<<" "<<indices[j]<<"j "<<j;
                A.data[i][indices[j]]+=data[j];
                //std::cout<<" erledigt"<<std::endl;
            }
        }
    } else {
        for(i=0;i<pointer_size-1;i++){
            for(j=pointer[i];j<pointer[i+1];j++){
                //std::cout<<"expand:j scheife "<<i<<" "<<indices[j]<<"j "<<j<<std::flush;
                A.data[indices[j]][i]+=data[j];
                //std::cout<<" erledigt"<<std::endl;
            }
        }
    }
    return A;
  }

template<class T> void matrix_sparse<T>::compress(const matrix_dense<T>& A, orientation_type o, double threshold){
    Integer counter=0;
    Integer i,j;
    number_rows=A.rows();
    number_columns=A.columns();
    for(i=0;i<A.rows();i++)
        for(j=0;j<A.columns();j++)
            if (std::abs(A.read(i,j)) > threshold) counter++;
    reformat(number_rows, number_columns,counter,o);
    counter = 0;
    if(o == ROW){
        for(i=0;i<A.rows();i++){
            pointer[i]=counter;
            for(j=0;j<A.columns();j++)
                if(std::abs(A.read(i,j)) > threshold) {
                    indices[counter] = j;
                    data[counter] = A.read(i,j);
                    counter++;
                }
        }
        pointer[A.rows()]=counter;
    } else {
        for(j=0;j<A.columns();j++){
            pointer[j]=counter;
            for(i=0;i<A.rows();i++)
                if (std::abs(A.read(i,j)) > threshold) {
                    indices[counter] = i;
                    data[counter] = A.read(i,j);
                    counter++;
                }
        }
        pointer[A.columns()]=counter;
    }
  }



template<class T> T scalar_prod(const matrix_sparse<T> &A, Integer m, const matrix_oriented<T> &B, Integer n);

template<class T> std::istream& operator >> (std::istream& is, matrix_dense<T>& X);

template<class T> std::ostream& operator << (std::ostream& os, const matrix_oriented<T> & x);
template<class T> std::ostream& operator << (std::ostream& os, const matrix_dense<T>& x);



template<class T> T scalar_prod(const matrix_sparse<T> &A, Integer m, const matrix_oriented<T> &B, Integer n){
     #ifdef DEBUG
         if(non_fatal_error((m>=A.dim_along_orientation())||(n>=B.dim_along_orientation()),"scalar_prod: these rows/columns do not exist.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         if(non_fatal_error((A.orientation != B.orientation),"scalar_prod: the arguments must have the same orientation.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
         if(non_fatal_error((A.dim_against_orientation() != B.dim_against_orientation()),"scalar_prod: the rows/columns of the arguments have incompatible dimensions.")) throw iluplusplus_error(INCOMPATIBLE_DIMENSIONS);
     #endif
     T prod=0.0;
     Integer offset = ((Integer)(n))*((Integer)(B.dim_against_orientation()));
     for(Integer k=A.read_pointer(m); k<A.read_pointer(m+1);k++)
         prod += A.read_data(k)*B.read_data(offset+A.read_indices(k));
     return prod;
   }


} // end namespace iluplusplus
