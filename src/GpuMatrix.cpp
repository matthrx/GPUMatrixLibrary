#include <iostream>
#include <assert.h>

// Headers of file declaration
#include "GpuMatrix.hpp"

template <typename T>
GpuMatrix<T>::GpuMatrix(unsigned int rows, unsigned int columns) : ROWS(rows), COLUMNS(columns) {
    this->data = new T[rows*columns];
}

template <typename T>
GpuMatrix<T>::GpuMatrix(unsigned int rows, unsigned int columns, T* data) : ROWS(rows), COLUMNS(columns), data(data) {
}

template <typename T>
void GpuMatrix<T>::freeMatrixGPU(void)
{
    delete [] this->data;
}

