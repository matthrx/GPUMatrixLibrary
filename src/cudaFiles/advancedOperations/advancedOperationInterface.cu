#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <math.h> 
#include <string>
#include <sstream>

#include "advancedOperationsKernel.cuh"
#include "../generalInformation/generalInformation.cuh"
// #include "../../GPUOperations.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define assertm(exp, msg) assert(((void)msg, exp))
#define GET_VARIABLE_NAME(a) (#a)
#define EXIT_SUCESS 0 
#define __GPU_EXP__
#define THREADS_PER_BLOCK_DIM 16
#define min(a, b) (((a)>(b)) ? (b) : (a))
#define carre(x) (x*x)


struct typeProps{
    int maxGridSize;
} variableProps = {65355};

// const cudaDeviceProp deviceProps;
// typedef __device__ __host__ auto lambdaExpression;
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
// {
//     if (code != cudaSuccess)
//     {
//       std::cerr << cudaGetErrorString(code) << " file : " << file << " line : " <<  line << std::endl;
//       if (abort) { exit(code); }
//     }
// }
    
template <typename T>
__host__ Matrix<T> transposeInterface(Matrix<T> a){
    const size_t SIZE = a.ROWS*a.COLUMNS*sizeof(T);
    T *da, *dataTranspose, *d_dataTranspose;


    gpuErrchk(cudaHostAlloc((void**)&dataTranspose, SIZE, cudaHostAllocDefault));
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&d_dataTranspose, SIZE));

    gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)a.ROWS/(float)THREADS_PER_BLOCK_DIM), variableProps.maxGridSize) , min(ceil((float)a.COLUMNS/(float)THREADS_PER_BLOCK_DIM), variableProps.maxGridSize));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    float sharedMemorySize = (float)(a.ROWS*a.COLUMNS)/(float)(carre(THREADS_PER_BLOCK_DIM)* blocksPerGrid.x * blocksPerGrid.y);
    transpose<<<blocksPerGrid, threadsPerBlock, ceil(sharedMemorySize)*carre(THREADS_PER_BLOCK_DIM)*sizeof(T)>>>(da, d_dataTranspose, a.ROWS, a.COLUMNS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(dataTranspose,  d_dataTranspose, SIZE, cudaMemcpyDeviceToHost));
    
    // gpuErrchk(cudaFreeHost(dataTranpose));

    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(d_dataTranspose));

    Matrix<T> toReturn;
    toReturn.ROWS = a.COLUMNS;
    toReturn.COLUMNS = a.ROWS;
    toReturn.data = dataTranspose;
    return toReturn;
}

template <typename T>
__host__ Matrix<T> dotInterface(Matrix<T> a, Matrix<T> b){
    std::ostringstream alertMessage;
    alertMessage << "Error : Those matrixes can't be multiplied check their dimensions \n In product A.B where A is "<< GET_VARIABLE_NAME(a) << "and B is " << GET_VARIABLE_NAME(b) << " : dim(A)=[" << a.ROWS
    << "," << a.COLUMNS << "] & dim(B) = ["<< b.ROWS << "," << b.COLUMNS << "]";
    assertm(a.ROWS == b.COLUMNS && a.COLUMNS== b.ROWS, alertMessage.str());
    const size_t SIZE = a.ROWS*b.COLUMNS*sizeof(T);
    T *da, *db, *d_dataProduct, *dataProduct;

    gpuErrchk(cudaHostAlloc((void**)&dataProduct, SIZE, cudaHostAllocDefault));
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&db, SIZE));
    gpuErrchk(cudaMalloc((void**)&d_dataProduct, SIZE));

    gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b.data, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)a.ROWS/(float)THREADS_PER_BLOCK_DIM), variableProps.maxGridSize) , min(ceil((float)a.COLUMNS/(float)THREADS_PER_BLOCK_DIM), variableProps.maxGridSize));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    dot<<<blocksPerGrid, threadsPerBlock>>>(da, db, d_dataProduct, a.ROWS, b.COLUMNS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(dataProduct,  d_dataProduct, SIZE, cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(d_dataProduct));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(da));
    
    Matrix<T> toReturn;
    toReturn.ROWS = a.ROWS;
    toReturn.COLUMNS = a.COLUMNS;
    toReturn.data = dataProduct;
    return toReturn;
}

int main(void){
    struct Matrix<double> matrix= Matrix<double>{16, 16, new double[16*16]};
    struct Matrix<double> matrixR= Matrix<double>{16, 16, new double[16*16]};

    for (unsigned int i = 0; i<matrix.ROWS*matrix.COLUMNS; i++){
        matrix.data[i] = 1;
        matrixR.data[i] = 2;
        // std::cout << "Value " << i << " : " << matrix.data[i] << " ---" << std::flush;
    }
    struct Matrix<double> result = dotInterface(matrix, matrixR);
    gpuPrint(matrix, 10, 10);
    gpuPrint(result, 10, 10);
    delete [] matrix.data;
    delete [] matrixR.data;
    return 0;
}