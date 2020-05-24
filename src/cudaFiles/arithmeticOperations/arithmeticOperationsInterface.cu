#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>       /* ceil */
#include <functional>
#include <nvfunctional>
#include <assert.h>

// #include "../initialisation/initialisation.cuh"
// #include "../../GPUOperations.h"

#include "arithmeticOperationsKernel.cuh"
#include "../generalInformation/generalInformation.h"
#include "../../GpuMatrix.h"

// It would be ideal to transfert data while executing kernel device operations

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define EXIT_SUCESS 0 
#define __GPU_EXP__
#define THREADS_PER_BLOCK_DIM 16
#define min(a, b) (((a)>(b)) ? (b) : (a))
#define carre(x) (x*x)
#define assertm(exp, msg) assert(((void)msg, exp))
#define functionDeviceHost(func) (__device__ __host__  func)

// const cudaDeviceProp deviceProps;
// typedef __device__ __host__ auto lambdaExpression;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
      std::cerr << cudaGetErrorString(code) << " file : " << file << " line : " <<  line << std::endl;
      if (abort) { exit(code); }
    }
}

/**************************************************************************************************
Kernel functions in cuh file
************************************************************************************************/

template <typename T>
GpuMatrix<T> GpuMatrix<T>::add(GpuMatrix<T> a, GpuMatrix<T> b){
    assertm((a.ROWS==b.ROWS && a.COLUMNS==b.COLUMNS), "Error incompatible dimensions, can't apply the operator");
    const size_t SIZE = a.ROWS*b.COLUMNS*sizeof(T);
    T *da, *db, *dc, *hc;
    
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&db, SIZE));
    gpuErrchk(cudaMalloc((void**)&dc, SIZE));
    gpuErrchk(cudaHostAlloc((void**)&hc, SIZE, cudaHostAllocDefault));

    gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b.data, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)a.ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)a.COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    addGPU<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, a.ROWS, a.COLUMNS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(hc,  dc, SIZE, cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(dc));
    GpuMatrix<T> toReturn = GpuMatrix<T>(a.ROWS, a.COLUMNS, hc);
    return toReturn;

}

template <typename T>
GpuMatrix<T> GpuMatrix<T>::substract(GpuMatrix<T> a, GpuMatrix<T> b){
    assertm((a.ROWS==b.ROWS && a.COLUMNS==b.COLUMNS), "Error incompatible dimensions, can't apply the operator");
    const size_t SIZE = a.ROWS*b.COLUMNS*sizeof(T);
    T *da, *db, *dc, *hc;
    
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&db, SIZE));
    gpuErrchk(cudaMalloc((void**)&dc, SIZE));
    gpuErrchk(cudaHostAlloc((void**)&hc, SIZE, cudaHostAllocDefault));

    gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b.data, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)a.ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)a.COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    substractGPU<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, a.ROWS, a.COLUMNS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(hc,  dc, SIZE, cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(dc));

    GpuMatrix<T> toReturn = GpuMatrix<T>(a.ROWS, a.COLUMNS, hc);
    return toReturn;


}

template <typename T>
GpuMatrix<T> GpuMatrix<T>::multiply(GpuMatrix<T> a, GpuMatrix<T> b){
    assertm((a.ROWS==b.ROWS && a.COLUMNS==b.COLUMNS), "Error incompatible dimensions, can't apply the operator");
    const size_t SIZE = a.ROWS*b.COLUMNS*sizeof(T);
    T *da, *db, *dc, *hc;
    
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&db, SIZE));
    gpuErrchk(cudaMalloc((void**)&dc, SIZE));
    gpuErrchk(cudaHostAlloc((void**)&hc, SIZE, cudaHostAllocDefault));

    gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b.data, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)a.ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)a.COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    multiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, a.ROWS, a.COLUMNS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(hc,  dc, SIZE, cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(dc));

    GpuMatrix<T> toReturn = GpuMatrix<T>(a.ROWS, b.COLUMNS, hc);
    return toReturn;


}

template <typename T>
GpuMatrix<T> GpuMatrix<T>::scalarMultiply(T a, GpuMatrix<T> b){
    // assertm((a.ROWS==b.ROWS && a.COLUMNS*b.COLUMNS), "Error incompatible dimensions, can't apply the operator");
    const size_t SIZE = b.ROWS*b.COLUMNS*sizeof(T);
    T *da, *db, *dc, *hc;
    
    gpuErrchk(cudaMalloc((void**)&da, sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&db, SIZE));
    gpuErrchk(cudaMalloc((void**)&dc, SIZE));
    gpuErrchk(cudaHostAlloc((void**)&hc, SIZE, cudaHostAllocDefault));

    gpuErrchk(cudaMemcpy(da, &a, sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b.data, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)b.ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)b.COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    scalarMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, b.ROWS, b.COLUMNS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(hc,  dc, SIZE, cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(dc));

    GpuMatrix<T> toReturn = GpuMatrix<T>(b.ROWS, b.COLUMNS, hc);
    return toReturn;


}

template <typename T>
 GpuMatrix<T> GpuMatrix<T>::divide(GpuMatrix<T> a, GpuMatrix<T> b){
    assertm((a.ROWS==b.ROWS && a.COLUMNS == b.COLUMNS), "Error incompatible dimensions, can't apply the operator");
    const size_t SIZE = a.ROWS*a.COLUMNS*sizeof(T);
    T *da, *db, *dc, *hc;
    
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&db, SIZE));
    gpuErrchk(cudaMalloc((void**)&dc, SIZE));
    gpuErrchk(cudaHostAlloc((void**)&hc, SIZE, cudaHostAllocDefault));

    gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b.data, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)a.ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)a.COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    divideGPU<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, a.ROWS, a.COLUMNS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(hc,  dc, SIZE, cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(dc));

    GpuMatrix<T> toReturn = GpuMatrix<T>(a.ROWS, a.COLUMNS, hc);
    return toReturn;


}
// template <typename T, typename F>
// __host__ __device__ F lambdaGPU(T x, const nvstd::function<F(T&)> func){
//     return func(x);
// }


template <typename T, typename F>
 GpuMatrix<F> applyLambdaToElementMatrix(GpuMatrix<T> a, F lambdaFunction){
    
    // #ifndef __CUDACC_EXTENDED_LAMBDA__
    // #error "please compile with --expt-extended-lambda add it to make file"
    // #endif

    const size_t SIZE_T = a.ROWS*a.COLUMNS*sizeof(T);
    const size_t SIZE_F = a.ROWS*a.COLUMNS*sizeof(F);
    T* d_a;
    F* d_result, *result;
    // int x = 2;
    // lambdaGPU(x , lambdaFunction);
    // __device__ auto lambdaFunctionOnGPU = lambdaFunction; // need of --expt-extended-lambda
    
    // std::cout  << lambdaFunctionOnGPU(x) << std::endl;
    gpuErrchk(cudaMalloc((void**)&d_a, SIZE_T));
    gpuErrchk(cudaMalloc((void**)&d_result, SIZE_F));
    gpuErrchk(cudaHostAlloc((void**)&result, SIZE_F, cudaHostAllocDefault));

    gpuErrchk(cudaMemcpy(d_a, a, SIZE_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_result, result, SIZE_F, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)a.ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)a.COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);
    // std::cout << *(a + 20000) << std::endl;

    applyLambdaToElementMatrixGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_result, lambdaFunction, a.ROWS, a.COLUMNS);
    // applyLambdaToElementMatrixGPU<<<(1, 1), (1, 1) >>>(d_a, d_result, lambdaFunctionOnGPU, ROWS, COLUMN);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(result, d_result, SIZE_F, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_result));

    // return result;c
    GpuMatrix<F> toReturn = GpuMatrix<F>(a.ROWS, a.COLUMNS, result);
    return toReturn;

}

// int main(void){
//     struct Matrix<double> matrix= Matrix<double>{16, 16, new double[16*16]};

//     for (unsigned int i = 0; i<matrix.ROWS*matrix.COLUMNS; i++){
//         matrix.data[i] = 2;
//         // std::cout << "Value " << i << " : " << matrix.data[i] << " ---" << std::flush;
//     }

//     struct Matrix<double> result = applyLambdaToElementMatrix<double>(matrix, carre);
//     // gpuPrint(matrix, 10, 10);
//     // gpuPrint(result, 10, 10);
//     delete [] matrix.data;
//     // delete [] matrixR.data;
//     return 0;
// }

// template float twice<float>(float original)
// template GpuMatrix<double> GpuMatrix<double>::add(GpuMatrix<double>, GpuMatrix<double>);