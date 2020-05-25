#ifndef __ADVANCED_OPERATIONS_KERNEL__
#define __ADVANCED_OPERATIONS_KERNEL__

#include <cuda.h>
#include <iostream>
#include <stdlib.h>

template <typename T>
__global__ void transposeKernel(const T*  __restrict__ a, T* b, int ROWS, int COLUMNS) {
    // easy columns become rows
    unsigned long int stride = gridDim.x*blockDim.x*gridDim.y*blockDim.y; // Total amount of threads. 
    unsigned int offset{};
    unsigned int tidX = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tidY = blockIdx.y*blockDim.y+threadIdx.y;
    // we'll use cache memory L2 because of its rate of 2,000GBps higher thant GDRAM (300GBps) and PCIe (16GBps) - SM is at 20,000GBps
    
    extern __shared__ __align__(sizeof(T)) unsigned char my_cache[];
    T* cache = reinterpret_cast<T*>(my_cache);
    
    #pragma unroll
    while (tidY*ROWS+tidX+offset < ROWS*COLUMNS){
        cache[threadIdx.x+threadIdx.y*blockDim.y*(offset+stride)/stride]= *(a+ tidY*ROWS+tidX+offset);
        offset += stride;
    }

    __syncthreads();
    offset = 0;
    unsigned int dest =  (blockIdx.y * blockDim.y + threadIdx.x) + (blockIdx.x * blockDim.x + threadIdx.y)*COLUMNS;
    #pragma unroll
    while (dest + offset < ROWS*COLUMNS){
        *(b+ dest + offset)  = cache[threadIdx.y+threadIdx.x*blockDim.x*(offset+stride)/stride];
        offset += stride;
    }
}

template <typename T>
__global__ void dotKernel(const T*  __restrict__ a, const T* __restrict__  b, T* c, int ROWS, int COLUMNS) {
    /*
    Following the hypothesis that we are using digital type, double , float or int
    */
    unsigned int tidX = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
    // unsigned long int stride = gridDim.x*blockDim.x * gridDim.y*blockDim.y; // Total amount of threads. 
    // unsigned int offset{};
    
    T intermediateValue{};
    if (tidX < COLUMNS && tidY < ROWS){
        #pragma unroll
        for (unsigned int i = 0; i < COLUMNS; i ++){
            intermediateValue += a[tidY*ROWS+i] * b[i*COLUMNS+tidX] ;
            }
        *(c+ tidY*ROWS + tidX) = intermediateValue;
        }
}


#endif 