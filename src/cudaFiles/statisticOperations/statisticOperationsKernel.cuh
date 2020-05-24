#ifndef __STATISTIC_OPERATIONS_KERNEL_CUH__
#define __STATISTIC_OPERATIONS_KERNEL_CUH__

#include <cuda.h>
#include <iostream>
#include <stdlib.h>

// template <typename T>
// __device__ T max(T a, T b){
//     return (a > b ? a : b);
// }

template <typename T>
__global__ void minGPU(T* a, T* min, int amountColumns, int amountRows, int* mutex) {
    /*
    - the min will just be defined for numerical values ! 
    we'll use fminf which is CUDA optimised
    T must be float or double or int
    */
    unsigned long int stride = gridDim.x*blockDim.x * gridDim.y*blockDim.y;
    unsigned int offset{};
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    T tempMin = *min;
    while (index + offset < amountColumns*amountRows){
        tempMin = fminf(tempMin, *(a + index + offset));
        offset += stride;
    }
    extern __shared__ __align__(sizeof(T)) unsigned char my_cache[];
    T* cache = reinterpret_cast<T*>(my_cache);
    if (threadIdx.x == 0 && threadIdx.y == 0){
        for (unsigned int i = 0; i < (offset/stride)*(threadIdx.x*(blockDim.x-1)+  threadIdx.y); i++){
            cache[i] = *min;
        }
    }
    unsigned int indexCache = (threadIdx.x*(blockDim.x-1)+  threadIdx.y);
    cache[indexCache] = tempMin;
    __syncthreads();
   
    unsigned long int i = (blockDim.x *blockDim.y)/2;
    while (i > 0){
        if (indexCache < i) {
            __syncthreads();
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
    we'll use fminf which is CUDA optimised
    T must be float or double or int
    */
    unsigned long int stride = gridDim.x*blockDim.x*gridDim.y*blockDim.y;
    unsigned int offset{};
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    T tempMax = *max;
    if (index < amountColumns*amountRows) {
        while (index + offset < amountColumns*amountRows){
            tempMax = fmaxf(tempMax, *(a + index + offset));
            offset += stride;
        }
        extern __shared__ __align__(sizeof(T)) unsigned char my_cache[];
        T* cache = reinterpret_cast<T*>(my_cache);
        if (threadIdx.x == 0 && threadIdx.y == 0){
            for (unsigned int i = 0; i < (offset/stride)*blockDim.x*blockDim.y; i++){
                cache[i] = *max;
            }
        }
        unsigned int indexCache = (threadIdx.x*(blockDim.x)+threadIdx.y);
        cache[indexCache] = tempMax;
        unsigned int i = (blockDim.x*blockDim.y)/2;
        while (i > 0){
            if (indexCache < i) {
                __syncthreads();
                cache[indexCache] = fmaxf(cache[indexCache], cache[indexCache+i]);
                __syncthreads();
            }
            i /= 2;

            // if (debugGlobal){
            //     printf("In reduction cache[0] is %lf step %i\n", cache[0], i);
            // }

        }
        if (threadIdx.x == 0 && threadIdx.y == 0){
            while (atomicCAS(mutex, 0, 1) == 1); // Lock a mutex to be the only one dealing 
            *max = fmaxf(*max, cache[0]);
            atomicExch(mutex, 0);
        }
    }
}

template <typename T>
__global__ void meanGPU(T* a, T* mean, int amountColumns, int amountRows, int* mutex) {
    unsigned long int stride = gridDim.x*blockDim.x*gridDim.y*blockDim.y; // Total amount of threads. 
    unsigned int offset{};
    // unsigned int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    
    T sum{};
    extern __shared__ __align__(sizeof(T)) unsigned char my_cache[];
    T* cache = reinterpret_cast<T*>(my_cache);
    if (index < amountColumns*amountRows) {
        while(index+offset < amountRows*amountColumns){
            sum += a[index+offset];
            offset += stride;
        }
        if (threadIdx.x == 0 && threadIdx.y == 0){
            for (unsigned int i = 0; i < (offset/stride) *(blockDim.x*blockDim.y); i++){
                cache[i] = 0;
            }
        }
    
        __syncthreads();
        unsigned int indexCache = (threadIdx.x*blockDim.x+threadIdx.y);
        
        cache[indexCache] = sum;
        // if (blockIdx.x + blockIdx.y == 0){
        //     for (unsigned int i = 0; i<blockDim.x*blockDim.y; i++){
        //     printf("Sum %lf --------- %lf cache for index %d\n", sum, cache[2], indexCache);
        //     }
        // }       
        //  printf("Here is sum in a : %lf ans index is %d\n", sum, index);

        unsigned int i = ((blockDim.x*blockDim.y)*(stride/offset))/2;
        while (i > 0){
            if (indexCache < i) {
                cache[indexCache] += cache[indexCache+i];
                __syncthreads();
            }
            i /= 2;
        }
        // for (unsigned int i = 1; i < (blockDim.x*blockDim.y)*(stride/offset); i++){
        //     cache[0] += cache[i];
        //     __syncthreads();
        // }
        if (threadIdx.x == 0 && threadIdx.y == 0){
            // printf("In null condition %lf -- (%d, %d)\n", cache[0], blockIdx.x, blockIdx.y);
            while (atomicCAS(mutex, 0, 1) == 1); // Lock a mutex to be the only one dealing 
            *mean += cache[0];
            atomicExch(mutex, 0);
        }

    }

}

#endif