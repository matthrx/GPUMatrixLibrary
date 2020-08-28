// Must be done... 
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <math.h> 
#include <string>
#include <sstream>
#include <cstdlib>
#include <limits>

// #include "../../GPUOperations.h"
// #include "../initialisation/initialisation.cuh"
#include "statisticOperationsKernel.cuh"
#include "../../GpuMatrix.hpp"
#include "../generalInformation/generalInformation.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define assertm(exp, msg) assert(((void)msg, exp))
#define THREADS_PER_BLOCK_DIM 16
#define carre(x) (x*x)
#define minHost(a, b) (((a) < (b)) ? (a) : (b))

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
      std::cerr << cudaGetErrorString(code) << " file : " << file << " line : " <<  line << std::endl;
      if (abort) { exit(code); }
    }
}

template <typename T> 
T GpuMatrix<T>::minGpuMatrix(void){
    const size_t SIZE = this->ROWS*this->COLUMNS*sizeof(T);
    int* mutex = 0;
    T *dmin, *da;
    T *minValue = new T;
  
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&dmin, sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&mutex, sizeof(int)));
    if (std::numeric_limits<T>::has_infinity){
        const T max = std::numeric_limits<T>::max();
        gpuErrchk(cudaMemset(dmin, max , sizeof(T)));
    }
    else {
        exit(1);
    }
    gpuErrchk(cudaMemset(mutex, 0, sizeof(int)));
    gpuErrchk(cudaMemcpy(da, this->data, SIZE, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(dmin, min, sizeof(T), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(minHost(ceil((float)this->ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , minHost(ceil((float)this->COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    // minGPU<<<blocksPerGrid, threadsPerBlock, ceil(((this->ROWS*this->COLUMNS)/(blocksPerGrid.x*THREADS_PER_BLOCK_DIM*blocksPerGrid.y*THREADS_PER_BLOCK_DIM+1))*256*sizeof(T))>>>(da, dmin, this->ROWS, this->COLUMNS, mutex);
    float sharedMemorySize = (float)(this->ROWS*this->COLUMNS)/(float)(carre(THREADS_PER_BLOCK_DIM)* blocksPerGrid.x * blocksPerGrid.y);
    minGPU<<<blocksPerGrid, threadsPerBlock, ceil(sharedMemorySize)*carre(THREADS_PER_BLOCK_DIM)*sizeof(T)>>>(da, dmin, this->ROWS, this->COLUMNS, mutex);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(minValue,  dmin, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(dmin));
    gpuErrchk(cudaFree(da));
    // std::cout << "MinGPU will be " << *minValue << std::endl;
    gpuErrchk(cudaFree(mutex));    
    return *minValue;
}


template <typename T>
T GpuMatrix<T>::maxGpuMatrix(void){
    const size_t SIZE = this->ROWS*this->COLUMNS*sizeof(T);
    int* mutex = 0;
    T *dmax, *da;
    T *maxValue = new T;
  
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&dmax, sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&mutex, sizeof(int)));
    if (std::numeric_limits<T>::has_infinity){
        const T min = std::numeric_limits<T>::min();
        gpuErrchk(cudaMemset(dmax, min , sizeof(T)));
    }
    else {
        exit(1);
    }
    gpuErrchk(cudaMemset(mutex, 0, sizeof(int)));
    gpuErrchk(cudaMemcpy(da, this->data, SIZE, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(dmin, min, sizeof(T), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(minHost(ceil((float)this->ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , minHost(ceil((float)this->COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    // minGPU<<<blocksPerGrid, threadsPerBlock, ceil(((this->ROWS*this->COLUMNS)/(blocksPerGrid.x*THREADS_PER_BLOCK_DIM*blocksPerGrid.y*THREADS_PER_BLOCK_DIM+1))*256*sizeof(T))>>>(da, dmin, this->ROWS, this->COLUMNS, mutex);
    float sharedMemorySize = (float)(this->ROWS*this->COLUMNS)/(float)(carre(THREADS_PER_BLOCK_DIM)* blocksPerGrid.x * blocksPerGrid.y);
    maxGPU<<<blocksPerGrid, threadsPerBlock, ceil(sharedMemorySize)*carre(THREADS_PER_BLOCK_DIM)*sizeof(T)>>>(da, dmax, this->ROWS, this->COLUMNS, mutex);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(maxValue,  dmax, sizeof(T), cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(dmax));
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(mutex));    
    return *maxValue;
}


template <typename T> 
T GpuMatrix<T>::meanGpuMatrix(void){
    const size_t SIZE = this->ROWS*this->COLUMNS*sizeof(T);
    int* mutex = 0;
    T *dmean, *da;
    T *meanValue = new T;
  
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&dmean, sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&mutex, sizeof(int)));

    gpuErrchk(cudaMemset(mutex, 0, sizeof(int)));
    gpuErrchk(cudaMemset(dmean, 0, sizeof(T)));

    gpuErrchk(cudaMemcpy(da, this->data, SIZE, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(dmin, min, sizeof(T), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(minHost(ceil((float)this->ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , minHost(ceil((float)this->COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    // minGPU<<<blocksPerGrid, threadsPerBlock, ceil(((this->ROWS*this->COLUMNS)/(blocksPerGrid.x*THREADS_PER_BLOCK_DIM*blocksPerGrid.y*THREADS_PER_BLOCK_DIM+1))*256*sizeof(T))>>>(da, dmin, this->ROWS, this->COLUMNS, mutex);
    float sharedMemorySize = (float)(this->ROWS*this->COLUMNS)/(float)(carre(THREADS_PER_BLOCK_DIM)* blocksPerGrid.x * blocksPerGrid.y);
    meanGPU<<<blocksPerGrid, threadsPerBlock, ceil(sharedMemorySize)*carre(THREADS_PER_BLOCK_DIM)*sizeof(T)>>>(da, dmean, this->ROWS, this->COLUMNS, mutex);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(meanValue,  dmean, sizeof(T), cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(dmean));
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(mutex));    
    return *(meanValue)/(this->ROWS*this->COLUMNS);
}


