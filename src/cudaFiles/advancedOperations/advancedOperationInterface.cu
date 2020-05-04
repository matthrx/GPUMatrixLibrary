#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <math.h> 
#include <string>
#include <sstream>

#include "advancedOperationsKernel.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define assertm(exp, msg) assert(((void)msg, exp))
#define THREAD_PER_BLOCK 32
#define EXIT_SUCESS 0 
#define __GPU_EXP__

// typedef __device__ __host__ auto lambdaExpression;
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
      std::cerr << cudaGetErrorString(code) << " file : " << file << " line : " <<  line << std::endl;
      if (abort) { exit(code); }
    }
}

namespace AdvanceOperation {

    template <typename T>
    __host__ Matrix transposeInterface(Matrix a){
        const size_t SIZE = a.ROWS*a.COLUMN*sizeof(T);
        T *da, *dataTranspose, *d_dataTranspose;


        gpuErrchk(cudaHostAlloc((void**)&dataTranspose, SIZE, cudaHostAllocDefault));
        gpuErrchk(cudaMalloc((void**)&da, SIZE));
        gpuErrchk(cudaMalloc((void**)&d_dataTranspose, SIZE));

        gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));

        dim3 blocksPerGrid(16, 16);
        dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);

        transpose<<<blocksPerGrid, threadsPerBlock, ceil(((a.ROWS*a.COLUMNS)/(16*THREAD_PER_BLOCK*16*THREAD_PER_BLOCK))*256*sizeof(T)>)>>(da, d_dataTranspose, a.ROWS, a.COLUMN);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(dataTranpose,  d_dataTranspose, SIZE, cudaMemcpyDeviceToHost));
        
        gpuErrchk(cudaFreeHost(dataTranpose));

        gpuErrchk(cudaFree(da));
        gpuErrchk(cudaFree(d_dataTranspose));

        Matrix finalMatrix{
            a.COLUMN; a.ROWS; dataTranpose;
        };
        return finalMatrix;
    }

    template <typename T>
    __host__ Matrix dotInterface(Matrix a, Matrix b){
        std::ostringstream alertMessage;
        alertMessage << "Error : Those matrixes can't be multiplied check their dimensions \n In product A.B where A is "<< #A "and B is " << #B << " : dim(A)=[" << a.ROWS
        << "," << a.COLUMNS << "] & dim(B) = ["<< b.ROWS << "," << b.COLUMNS << "]";
        assertm(a.ROWS == b.COLUMN && a.COLUMN == b.ROWS, alertMessage.str());
        const size_t SIZE = ROWS*COLUMN*sizeof(T);
        T *da, *db, *d_dataProduct, *dataProduct;


        gpuErrchk(cudaHostAlloc((void**)&dataProduct SIZE, cudaHostAllocDefault));

        gpuErrchk(cudaMalloc((void**)&da, SIZE));
        gpuErrchk(cudaMalloc((void**)&db, SIZE));
        gpuErrchk(cudaMalloc((void**)&d_dataProduct, SIZE));

        gpuErrchk(cudaMemcpy(da, a.data, SIZE, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(db, b.data, SIZE, cudaMemcpyHostToDevice));

        dim3 blocksPerGrid(16, 16);
        dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);

        dot<<<blocksPerGrid, threadsPerBlock>>>(da, db, d_dataProduct, ROWS, COLUMN);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(dataProduct,  d_dataProduct, SIZE, cudaMemcpyDeviceToHost));
        
        gpuErrchk(cudaFreeHost(dataProduct));
        gpuErrchk(cudaFree(d_dataProduct));
        gpuErrchk(cudaFree(db));
        gpuErrchk(cudaFree(da));
        
        Matrix toReturn {
            a.ROWS b.COLUMN; dataProduct;
        };
        return toReturn;
    }
}
