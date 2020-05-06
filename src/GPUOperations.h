#ifndef __GPU_OPERATIONS_H__
#define __GPU_OPERATIONS_H__

#include <iostream> 
#include <tuple>
// #include "futureHeadersOfGPUMatrixClass"

template <typename T> class Matrix<T>;
template <typename T> Matrix<T> inverseMatrixHighPrecision(Matrix<T>);
template <typename T> Matrix<T> inverseMatrixNormalPrecision(Matrix<T>);
template <typename T> std::tuple<double*, double*, double*, double*> eigenValuesHighPrecisionNormalMatrix(Matrix<T>, bool, bool);
template <typename T> std::tuple<float*, float*, float*, float*> eigenValuesNormalPrecisionNormalMatrix(Matrix<T>, bool, bool);
template <typename T> std::tuple<float*, float*> eigenValuesNormalPrecisionSparseMatrix(Matrix<T>, bool);
template <typename T> std::tuple<double*,double*> eigenValuesHighPrecisionSparseMatrix(Matrix<T>, bool);
template <typename T> std::tuple<double*, double*, double*> singularValueDecompositionHighPrecision(Matrix<T>); 
template <typename T> std::tuple<float*, float*, float*> singularValueDecompositionNormalPrecision(Matrix<T> m);
template <typename T> Matrix<T> transposeInterface(Matrix<T>);
template <typename T> Matrix<T> dotInterface(Matrix<T>, Matrix<T>);

#endif __GPU_OPERATIONS_H__