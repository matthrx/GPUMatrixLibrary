#ifndef __ADVANCED_OPERATIONS_KERNEL__
#define __ADVANCED_OPERATIONS_KERNEL__

#include <cuda.h>
#include <iostream>
#include <stdlib.h>

template <typename T>
typedef struct {
    size_t rows;
    size_t columns;
    T* elements
} Matrix;


__global__ void transpose(const T*  __restrict a, T* b, int ROWS, int COLUMNS) {
    unsigned int tidX = threadIdx.x + blockDim.x* blockIdx.x;
    unsigned int tidY = threadIdx.y + blockDim.y* blockIdx.y;
    unsigned long int stride = gridDim.x*blockDim.x + gridDim.y*blockDim.y; // Total amount of threads. 
    unsigned int offset{};
    unsigned int index = tidX*COLUMNS + tidY;    // we'll use cache memory L2 because of its rate of 2,000GBps higher thant GDRAM (300GBps) and PCIe (16GBps) - SM is at 20,000GBps
    
    extern __shared__ T cache[]; //size of ceil((ROWS*COLUMNS)/(gridX*gridY*blockX*blockY))
    
    #pragma unroll
    while (index+offset < ROWS*COLUMNS){
        cache[((threadIdx.x+blockDim.x*(offset/stride))*(blockDim.x-1)+ threadIdx.y)] = *(a+index+offset);
        offset += stride;
    }
    __syncthreads();
    offset = 0;
    #pragma unroll
    while (index+offset < ROWS*COLUMNS){
        *(b + tidY*amountRows + tidX + offset)  = cache[((threadIdx.x+blockDim.x*(offset/stride))*(blockDim.x-1)+ threadIdx.y)];
        offset += stride;
    }
}

__global__ void dot(const T*  __restrict a, const T* __restrict  b, T* c, int ROWS, int COLUMNS) {
    /*
    Following the hypothesis that we are using digital type, double , float or int
    */
    unsigned int tidX = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned long int stride = gridDim.x*blockDim.x + gridDim.y*blockDim.y; // Total amount of threads. 
    unsigned int offset{};
    
    T intermediateValue{};
    if (tidX < COLUMNS && tidY < ROWS){
        #pragma unroll
        unsigned int i;
        for (i = 0; i < COLUMNS; i += 4){
            double4 a_tmp = reinterpret_cast<double4*>(&a.elements[i+COLUMNS*tidY])[0];
            double4 b_tmp = reinterpret_cast<double4*>(&b.elements[i*COLUMNS+tidX])[0]; // columns for a is row for b
            intermediateValue += (a_tmp.x * b_tmp.x);
            intermediateValue += (a_tmp.y * b_tmp.y);
            intermediateValue += (a_tmp.z  * b_tmp.z);
            intermediateValue += (a_tmp.w * b_tmp.w);
            }
        int rest = COLUMNS%4;
        if(rest != 0)
            rest--;
            intermediateValue += (a[(COLUMNS-rest)+COLUMNS*tidY]*b[(COLUMNS-res)*COLUMNS+tidX])
        *c(tidY*a.rows + tidX) = intermediateValue;
        }
}


__global__ void inverse(const Matrix __restrict__ a, Matrix b) {
    
}

int main(void){
    // Test
    return 0; 
}


#endif 