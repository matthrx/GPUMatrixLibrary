/*
It is the user responsability to choose the right type.
interacting with high value, it's highly recommended to use types like
long double...
*/
#ifndef __ARITHMETIC_OPERATIONS_KERNEL_CUH__
#define __ARITHMETIC_OPERATIONS_KERNEL_CUH__

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
    if (tidColumns < amountColumns && tidRows < amountRows){
        *(c + index + offset) = *(a + index + offset) - *(b + index + offset);
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

template <typename T, typename F, typename Function>
__global__ void applyLambdaToElementMatrixGPU(T* a, F* b, Function Func, int amountRows, int amountColumns) {
    /*
    Function parameter will be a device side lambda function on the host
    */
    // printf("%d", amountRows);
    // cuPrintf("Running\n");
    unsigned int tidX = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tidY = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int stride = blockDim.x*gridDim.x*tidY + tidX;
    unsigned int offset{};
    unsigned int index = tidX*amountColumns + tidY; 
    #pragma unroll
    while (index + offset < amountColumns*amountRows){
        *(b + index + offset) = Func(*(a + index + offset));
        offset += stride;
    }
}
#endif