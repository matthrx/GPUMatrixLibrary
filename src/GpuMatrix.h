#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__

#include <iostream>
/*all includes */

template <typename T> class GpuMatrix {
public:
    unsigned int ROWS;
    unsigned int COLUMNS;
    T* data;

public:
    GpuMatrix(unsigned int rows, unsigned int colums);
    GpuMatrix(unsigned int rows, unsigned int columns, T* data);
    ~GpuMatrix();
    T minGpuMatrix(void);
    T maxGpuMatrix(void);
    T meanGpuMatrix(void);
    static GpuMatrix<T> dot(const GpuMatrix<T> a, const GpuMatrix<T> b);
    static void matrixGPU_init(void);
    void matrixGPU_print(unsigned int rows, unsigned int columns);
    void matrixGPU_print(unsigned int rows, unsigned int columns, bool);

    void matrixGPU_print(unsigned int rows, unsigned int columns, bool, bool);
    GpuMatrix<T> transpose(void);
    static GpuMatrix<T> inverseMatrixHighPrecision(const GpuMatrix<T> a);
    static GpuMatrix<T> inverseMatrixNormalPrecision(const GpuMatrix<T> a);
    std::tuple<float*, float*, float*> getEigenValuesNormalPrecisionNormalMatrix(void);
    std::tuple<double*, double*, double*> getEigenValuesHighPrecisionNormalMatrix(void);
    std::tuple<float*, float*, float*> getEigenValuesNormalPrecisionNormalMatrix(bool);
    std::tuple<double*, double*, double*> getEigenValuesHighPrecisionNormalMatrix(bool);
    std::tuple<float*, float*, float*> getEigenValuesNormalPrecisionNormalMatrix(bool, bool);
    std::tuple<double*, double*, double*> getEigenValuesHighPrecisionNormalMatrix(bool, bool);
    std::tuple<double*, double*, double*> getSingularValueDecompositionHighPrecision(void);
    std::tuple<float*, float*, float*> getSingularValueDecompositionNormalPrecision(void);


    GpuMatrix<T> operator+(const GpuMatrix<T>& a){
        return add(*this, a);
    }

    GpuMatrix<T> operator-(const GpuMatrix<T>& a){
        return substract(*this, a);
    }

    GpuMatrix<T> operator*(const GpuMatrix<T>& a){
        return multiply(*this, a);
    }

    GpuMatrix<T> operator*(const T& a){
        return scalarMultiply(*this, a);
    }


    GpuMatrix<T> operator/(const GpuMatrix<T>& a){
        return divide(*this, a);
    }

};
#endif 