#include <cutil.h>

#include "arithmeticOperationsKernel.cuh"
/*
It is the user responsability to choose the right type.
interacting with high value, it's highly recommended to use types like
long double...

*/

template<typename T> 
__global__ void addGPU(T* a, T* b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the + operator on the template
    will be verified in the class call
    */
    unsigned int tidColumns = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tidRows = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int stride = blockDim.x*gridDim.x + blockDim.y+gridDim.y;
    unsigned int offset{};
    unsigned int index = tidRows*amountColumns + tidColumns; 
    #pragma unroll
    while (tidColumns < amountColumns && tidRows < amountRows){
        *(c + index + offset) = *(a + index + offset) + *(b + index + offset);
        offset += stride;
    }
}


template<typename T> 
__global__ void substractGPU(T* a, T* b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the + operator on the template
    will be verified in the class call
    */
    unsigned int tidColumns = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tidRows = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int stride = blockDim.x*gridDim.x + blockDim.y+gridDim.y;
    unsigned int offset{};
    unsigned int index = tidRows*amountColumns + tidColumns; 
    #pragma unroll
    while (tidColumns < amountColumns && tidRows < amountRows){
        *(c + index + offset) = *(a + index + offset) - *(b + index + offset);
        offset += stride;
    }
}


template<typename T> 
__global__ void multiplyGPU(T* a, T* b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the + operator on the template
    will be verified in the class call
    */
    unsigned int tidColumns = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tidRows = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int stride = blockDim.x*gridDim.x + blockDim.y+gridDim.y;
    unsigned int offset{};
    unsigned int index = tidRows*amountColumns + tidColumns; 
    #pragma unroll
    while (tidColumns < amountColumns && tidRows < amountRows){
        *(c + index + offset) = *(a + index + offset) * *(b + index + offset);
        offset += stride;
    }
}

template<typename T, typename U>
__global__ void scalarMultiplyGPU(U a, T* b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the + operator on the template
    will be verified in the class call
    */
    unsigned int tidColumns = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tidRows = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int stride = blockDim.x*gridDim.x + blockDim.y+gridDim.y;
    unsigned int offset{};
    unsigned int index = tidRows*amountColumns + tidColumns; 
    #pragma unroll
    while (tidColumns < amountColumns && tidRows < amountRows){
        *(c + index + offset) = a *(b + index + offset);
        offset += stride;
    }
}

template<typename T>
__global__ void divideGPU(T* a, T* b, T* c, int *sizeRows, int* sizeColumn) {


    template<typename T> 
    __global__ void addGPU(T* a, T* b, T* c, int amountRows, int amountColumns) {
        /*
        Dimensions of matrix && application of the / operator on the template
        will be verified in the class call
        According to IEEE-754 a division by zero will return inf (float type)
        */
        unsigned int tidColumns = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int tidRows = threadIdx.y + blockIdx.y*blockDim.y;
        unsigned int stride = blockDim.x*gridDim.x + blockDim.y+gridDim.y;
        unsigned int offset{};
        unsigned int index = tidRows*amountColumns + tidColumns; 
        #pragma unroll
        while (tidColumns < amountColumns && tidRows < amountRows){
            *(c + index + offset) = *(a + index + offset) + *(b + index + offset);
            offset += stride;
        }
    }

template <typename T, typename F>
__global__ void applyLambdaToElementMatrixGPU(T* a, F* b, int amountRows, int amountColumns) {
    /*
    Function parameter will be a device side lambda function on the host
    */
    unsigned int tidColumns = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int tidRows = threadIdx.y + blockIdx.y*blockDim.y;
        unsigned int stride = blockDim.x*gridDim.x + blockDim.y+gridDim.y;
        unsigned int offset{};
        unsigned int index = tidRows*amountColumns + tidColumns; 
        #pragma unroll
        while (tidColumns < amountColumns && tidRows < amountRows){
            *(b + index + offset) = lambdaGPU(*(a + index + offset));
            offset += stride;
        }
    }





