#ifndef __ARITHMETIC_OPERATIONS_KERNEL_CUH__
#define __ARITHMETIC_OPERATIONS_KERNEL_CUH__

template <typename T> __global__ void addGPU(T*, T*, T*, int, int);
template <typename T> __global__ void substractGPU(T*, T*, T*, int, int);
template <typename T> __global__ void multiplyGPU(T*, T*, T*, int, int);
template <typename T> __global__ void divideGPU(T*, T*, T*, int, int);
template <typename T, typename U > __global__ void scalarMultiplyGPU(U, T*, T*, int, int);
template <typename T, typename F> __global__ void applyLambdaToElementMatrixGPU(T*, F*, F);


#endif 