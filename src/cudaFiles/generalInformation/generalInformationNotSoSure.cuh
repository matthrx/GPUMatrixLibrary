#ifndef __INITIALISATION_H__
#define __INITIALISATION_H__

#include <cuda_runtime.h>
#include <cuda.h>

extern cudaDeviceProp deviceProps;
template <typename T> struct Matrix<T>;
template <typename T> void gpuPrint(Matrix<T>, unsigned int, unsigned int, bool, bool);

#endif 