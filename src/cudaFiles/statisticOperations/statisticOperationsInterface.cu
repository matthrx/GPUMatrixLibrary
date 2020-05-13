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
#include "../../GpuMatrix.h"
#include "../generalInformation/generalInformation.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define assertm(exp, msg) assert(((void)msg, exp))
#define THREADS_PER_BLOCK_DIM 16
#define carre(x) (x*x)
#define max(a,b) (((a) > (b)) ? (a) : (b)) 
#define min(a, b) (((a) < (b)) ? (a) : (b))

// struct typeProps deviceProps;
// for (unsigned int i = 0; i < 3; i++){
//      deviceProps.(maxGridSize[i]) = 65355;
// }

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
        const T inf = std::numeric_limits<T>::infinity();
        gpuErrchk(cudaMemset(dmin, inf , sizeof(T)));
    }
    else {
        return NULL;
    }
    gpuErrchk(cudaMemset(mutex, 0, sizeof(int)));
    gpuErrchk(cudaMemcpy(da, this->data, SIZE, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(dmin, min, sizeof(T), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)this->ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)this->COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    // minGPU<<<blocksPerGrid, threadsPerBlock, ceil(((this->ROWS*this->COLUMNS)/(blocksPerGrid.x*THREADS_PER_BLOCK_DIM*blocksPerGrid.y*THREADS_PER_BLOCK_DIM+1))*256*sizeof(T))>>>(da, dmin, this->ROWS, this->COLUMNS, mutex);
    float sharedMemorySize = (float)(this->ROWS*this->COLUMNS)/(float)(carre(THREADS_PER_BLOCK_DIM)* blocksPerGrid.x * blocksPerGrid.y);
    minGPU<<<blocksPerGrid, threadsPerBlock, ceil(sharedMemorySize)*carre(THREADS_PER_BLOCK_DIM)*sizeof(T)>>>(da, dmin, this->ROWS, this->COLUMNS, mutex);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(minValue,  dmin, sizeof(T), cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(dmin));
    gpuErrchk(cudaFree(da));
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
        const T inf = (T)(-1) * std::numeric_limits<T>::infinity();
        gpuErrchk(cudaMemset(dmax, 0 , sizeof(T)));
    }
    else {
        return NULL;
    }
    gpuErrchk(cudaMemset(mutex, 0, sizeof(int)));
    gpuErrchk(cudaMemcpy(da, this->data, SIZE, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(dmin, min, sizeof(T), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(min(ceil((float)this->ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)this->COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);

    std::cout << ceil((float)this->ROWS/(float)THREADS_PER_BLOCK_DIM) << std::endl;

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

    dim3 blocksPerGrid(min(ceil((float)this->ROWS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[0]) , min(ceil((float)this->COLUMNS/(float)THREADS_PER_BLOCK_DIM), deviceProps.maxGridSize[1]));
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
    std::cout << *meanValue << std::endl;
    return *(meanValue)/(this->ROWS*this->COLUMNS);
}



// int main(void){
//     double sum;
//     struct Matrix<double> matrix= Matrix<double>{10000, 10000, new double[10000*10000]};
//     for (unsigned int i = 0; i<matrix.ROWS*matrix.COLUMNS; i++){
//         matrix.data[i] = (rand() % 100)+5;
//         sum += matrix.data[i];
//         // std::cout << "Value " << i << " : " << matrix.data[i] << " ---" << std::flush;
//     }
//     double minGPU = maxGPUMatrixFunction(matrix);
//     std::cout << "Max GPU : " << minGPU << std::endl; 
//     std::cout << "Moyenne CPU : " << sum/(matrix.COLUMNS*matrix.ROWS) << std::endl;

//     delete [] matrix.data;
//     return 0;
// }
