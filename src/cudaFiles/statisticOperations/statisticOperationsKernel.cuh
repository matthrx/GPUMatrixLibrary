#ifndef __STATISTIC_OPERATIONS_KERNEL_CUH__
#define __STATISTIC_OPERATIONS_KERNEL_CUH__

#include <cuda.h>
#include <iostream>
#include <stdlib.h>

template <typename T>
__global__ void minGPU(T* a, T* min, int amountColumns, int amountRows, int* mutex) {
    /*
    - the min will just be defined for numerical values ! 
    we'll use fminf which is CUDA optimised
    T must be float or double or int
    */
    unsigned int tidX = threadIdx.x + blockDim.x* blockIdx.x;
    unsigned int tidY = threadIdx.y + blockDim.y* blockIdx.y;
    unsigned long int stride = gridDim.x*blockDim.x + gridDim.y*blockDim.y;
    unsigned int offset{};
    unsigned int index = tidX*amountColumns + tidY;
    T tempMin{};
    extern __shared__ T cache[];
    while (index + offset < amountColumns*amountRows){
        tempMin = fminf(tempMin, *(a + index + offset));
        offset += stride;
    }
    unsigned int indexCache = (threadIdx.x*(blockDim.x-1)+  threadIdx.y);
    cache[indexCache] = tempMin;
    __syncthreads();
    
    unsigned long int i = (blockDim.x *blockDim.y)/2;
    while (i > 0){
        if (indexCache < i) {
            cache[indexCache] = fminf(cache[indexCache], cache[indexCache+i]);
            __syncthreads();
        }
        i /= 2;

    }
    if (threadIdx.x == 0 && threadIdx.y == 0){
        while (atomicCAS(mutex, 0, 1) == 1); // Lock a mutex to be the only one dealing 
        *min = fminf(*min, cache[0]);
        atomicExch(mutex, 0);
    }
}

template <typename T>
__global__ void maxGPU(T* a, T* max, int amountColumns, int amountRows, int* mutex) {
    /*
    - the min will just be defined for numerical values ! 
    we'll use fmaxf which is CUDA optimised
    T must be float or double or int
    */
    unsigned int tidX = threadIdx.x + blockDim.x* blockIdx.x;
    unsigned int tidY = threadIdx.y + blockDim.y* blockIdx.y;
    unsigned long int stride = gridDim.x*blockDim.x + gridDim.y*blockDim.y; // Total amount of threads. 
    unsigned int offset{};
    unsigned int index = tidX*amountColumns + tidY;
    T tempMax{};
    extern __shared__ T cache[];
    while (index + offset < amountColumns*amountRows){
        tempMax = fmaxf(tempMax, *(a + index + offset));
        offset += stride;
    }
    unsigned int indexCache = (threadIdx.x*(blockDim.x-1)+  threadIdx.y);
    cache[indexCache] = tempMax;
    __syncthreads();
    
    unsigned long int i = (blockDim.x *blockDim.y)/2;
    while (i > 0){
        if (indexCache < i) {
            cache[indexCache] = fmaxf(cache[indexCache], cache[indexCache+i]);
            __syncthreads();
        }
        i /= 2;

    }
    if (threadIdx.x == 0 && threadIdx.y == 0){
        while (atomicCAS(mutex, 0, 1) == 1); // Lock a mutex to be the only one dealing 
        *max = fminf(*max, cache[0]);
        atomicExch(mutex, 0);
    }
}

template <typename T>
__global__ void meanGPU(T* a, T* mean, int amountColumns, int amountRows, int* mutex) {
    unsigned int tidX = threadIdx.x + blockDim.x* blockIdx.x;
    unsigned int tidY = threadIdx.y + blockDim.y* blockIdx.y;
    unsigned long int stride = gridDim.x*blockDim.x + gridDim.y*blockDim.y; // Total amount of threads. 
    unsigned int offset{};
    unsigned int index = tidX*amountColumns + tidY;
    
    T tempMin{}, sum{};
    extern __shared__ T cache[];
    while(index+offset < amountRows*amountColumns){
        sum += *(a+index+offset);
        offset += stride;
    }
    unsigned int indexCache = (threadIdx.x*(blockDim.x-1)+  threadIdx.y);
    cache[indexCache] = (sum/(offset/stride));
    __syncthreads();
     
    #pragma unroll
    sum = reinterpret_cast<T>(0);// Hardware is optimized to use all SIMT threads at once, how many cache lines - we'll wirte  vector4
    if (threadIdx.x + threadIdx.y == 0){
        for (unsigned int i = 0; i < (blockDim.x*blockDim.y); i++){
            sum += cache[i];
        } 
        while (atomicCAS(mutex, 0, 1) == 1); // Lock a mutex to be the only one dealing 
        *mean += sum;
        atomicExch(mutex, 0);
    }
    if ((blockIdx.x +blockIdx.y + threadIdx.x + threadIdx.y) == 0){
        *mean/=(gridDim.x*gridDim.t);
    }

}

int main(void){
    // kernel call is structured this way <<<Dg, Db, Ns, S>>>
    return EXIT_SUCESS;
}


#endif