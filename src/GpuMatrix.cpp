#include <iostream>
#include <assert.h>

// Headers of file declaration
#include "GPUOperations.h"
#include ""


template <typename T> class GpuMatrix {
public:
    unsigned int ROWS;
    unsigned int COLUMNS;
    T* data;

private:
    /* data */
public:
    GpuMatrix(unsigned int rows, unsigned int colums);
    ~GpuMatrix();
    T min(void);
    T max(void);
    T mean(void);
    static GPUMatrix<T> dot(const GpuMatrix<T>& a, const GpuMatrix<T>& b);
    GpuMatrix<T> transpose(void);
    GpuMatrix<T> inverseDoublePrecision(void);
    GpuMatrix<T> inverseFloatPrecision(void);

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
        return multiply(*this, a);
    }

};

template <typename T>
GpuMatrix<T>::GpuMatrix(unsigned int rows, unsigned int columns) : ROWS(rows), COLUMNS(columns) {
    this->data = new T[rows*columns];
}

template <typename T>
GpuMatrix<T> GpuMatrix<T>::inverseDoublePrecision(void){
    return inverseMatrixHighPrecision(*this);
}

template <typename T>
GpuMatrix<T> GpuMatrix<T>::inverseFloatPrecision(void){
    return inverseMatrixNormalPrecision(*this);
}

template <typename T>
T GpuMatrix<T>::min(void){
    return minGPUMatrixFunction(*this);
}

template <typename T>
T GpuMatrix<T>::max(void){
    return maxGPUMatrixFunction(*this);
}

template <typename T>
T GpuMatrix<T>::mean(void){
    return meanGPUMatrixFunction(*this);
}

template <typename T>
GpuMatrix<T> GpuMatrix<T>::transpose(void){
    return transposeInterface(*this);
}

template <typename T>
GpuMatrix<T>::~GpuMatrix()
{
    delete [] this->data;
}

template <typename T>
GpuMatrix<T> dot(const GpuMatrix<T>& a, const GpuMatrix<T>& b){
    return dotInterface(a,b);
}

