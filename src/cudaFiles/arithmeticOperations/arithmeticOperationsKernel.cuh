/*
It is the user responsability to choose the right type.
interacting with high value, it's highly recommended to use types like
long double...
*/
#ifndef __ARITHMETIC_OPERATIONS_KERNEL_CUH__
#define __ARITHMETIC_OPERATIONS_KERNEL_CUH__

template<typename T> 
__global__ void addGPU(const T* __restrict__ a, const T* __restrict__ b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the + operator on the template
    will be verified in the class call
    */
    // unsigned int tidColumns = threadIdx.x + blockIdx.x*blockDim.x;
    // unsigned int tidRows = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned long int stride = blockDim.x*gridDim.x*blockDim.y+gridDim.y;
    unsigned int offset{};
    // unsigned int index = tidRows*amountColumns + tidColumns; 
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    #pragma unroll
    while (index+offset < amountRows*amountColumns){
        *(c + index + offset) = *(a + index + offset) + *(b + index + offset);
        offset += stride;
    }
}


template<typename T> 
__global__ void substractGPU(const T* __restrict__  a, const T* __restrict__  b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the - operator on the template
    will be verified in the class call
    */

    unsigned long int stride = blockDim.x*gridDim.x * blockDim.y+gridDim.y;
    unsigned int offset{};
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    #pragma unroll
    while (index+offset < amountRows*amountColumns){
        *(c + index + offset) = *(a + index + offset) - *(b + index + offset);
        offset += stride;
    }
}


template<typename T> 
__global__ void multiplyGPU(const T* __restrict__  a, const T* __restrict__  b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the x operator on the template
    will be verified in the class call
    */
    unsigned long int stride = blockDim.x*gridDim.x * blockDim.y+gridDim.y;
    unsigned int offset{};
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    #pragma unroll
    while (index+offset < amountRows*amountColumns){
        *(c + index + offset) = *(a + index + offset) * *(b + index + offset);
        offset += stride;
    }
}

template<typename T> 
__global__ void divideGPU(const T* __restrict__  a, const T* __restrict__  b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the / operator on the template
    will be verified in the class call
    */
    unsigned long int stride = blockDim.x*gridDim.x * blockDim.y+gridDim.y;
    unsigned int offset{};
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    #pragma unroll
    while (index+offset < amountRows*amountColumns){
        *(c + index + offset) = *(a + index + offset) / *(b + index + offset);
        offset += stride;
    }
}

template<typename T>
__global__ void scalarMultiplyGPU(const T* __restrict__  a, const T* __restrict__  b, T* c, int amountRows, int amountColumns) {
    /*
    Dimensions of matrix && application of the * operator on the template
    will be verified in the class call
    */
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    unsigned long int stride = blockDim.x*gridDim.x * blockDim.y+gridDim.y;
    unsigned int offset{};
    #pragma unroll
    while (index+offset < amountRows*amountColumns){
        *(c + index + offset) = *a * *(b + index + offset);
        offset += stride;
    }
}

template <typename T, typename F, typename Function>
__global__ void applyLambdaToElementMatrixGPU(const T* __restrict__  a, F* b, Function Func, int amountRows, int amountColumns) {
    /*
    Function parameter will be a device side lambda function created as a __dev
    */
    // printf("%d", amountRows);
    // cuPrintf("Running\n");
    unsigned int index = (blockIdx.x+ blockIdx.y * gridDim.x)*(blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    unsigned long int stride = blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    unsigned int offset{};
    #pragma unroll
    while (index+offset < amountRows*amountColumns){
        *(b + index + offset) = Func(*(a + index + offset)); 
        offset += stride;
    }
}
#endif