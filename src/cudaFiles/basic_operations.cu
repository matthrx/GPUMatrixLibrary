#include <assert.h>
#include <cutil.h>

#define BLOCKS = 64
#define THREAD_PER_BLOCK 256 // can be 128 for some GPU (mumtiple of warp 32)
/*
It is the user responsability to choose the right type.
interacting with high value, it's highly recommended to use types like
long double...

*/


template<typename T> 
__global__ void addGPU(T* a, T* b, T* c, int *sizeRows, int* sizeColumn) {
    /*
    Dimensions of matrix && application of the + operator on the template
    will be verified in the class call
    */
    unsigned int rows = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int column = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int strideRows = gridDim.x*blockDim.x
    unsigned int strideColumn = gridDim.y*blockDim.y
    unsigned int offsetRows{}, offsetColumn{};
    while (rows + offsetRows < sizeRows) {
        while(column + offsetColumn < sizeColumn){
            c[rows][column] = a[rows][column] + b[rows][column];
            offsetColumn += strideColumn;
        }
        offsetColumn = 0;
        offsetRows += strideRows;
    }
}

template<typename T>
__global__ void substractGPU(T* a, T* b, T* c, int *sizeRows, int* sizeColumn) {
        /*
    Dimensions of matrix && application of the - operator on the template
    will be verified in the class call
    */
    unsigned int rows = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int column = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int strideRows = gridDim.x*blockDim.x
    unsigned int strideColumn = gridDim.y*blockDim.y
    unsigned int offsetRows{}, offsetColumn{};
    while (rows + offsetRows < sizeRows) {
        while(column + offsetColumn < sizeColumn){
            c[rows][column] = a[rows][column] - b[rows][column];
            offsetColumn += strideColumn;
        }
        offsetColumn = 0;
        offsetRows += strideRows;
    }
}

template<typename T>
__global__ void multiplyGPU(T* a, T* b, T* c, int *sizeRows, int* sizeColumn) {
        /*
    Dimensions of matrix && application of the * operator on the template
    will be verified in the class call
    */
    unsigned int rows = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int column = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int strideRows = gridDim.x*blockDim.x
    unsigned int strideColumn = gridDim.y*blockDim.y
    unsigned int offsetRows{}, offsetColumn{};
    while (rows + offsetRows < sizeRows) {
        while(column + offsetColumn < sizeColumn){
            c[rows][column] = a[rows][column] * b[rows][column];
            offsetColumn += strideColumn;
        }
        offsetColumn = 0;
        offsetRows += strideRows;
    }
}

template<typename T, typename U>
__global__ void scalarMultiplyGPU(U* a, T* b, T* c, int *sizeRows, int* sizeColumn) {
        /*
    Dimensions of matrix && application of the * operator on the template
    will be verified in the class call
    */
    unsigned int rows = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int column = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int strideRows = gridDim.x*blockDim.x
    unsigned int strideColumn = gridDim.y*blockDim.y
    unsigned int offsetRows{}, offsetColumn{};
    while (rows + offsetRows < sizeRows) {
        while(column + offsetColumn < sizeColumn){
            c[rows][column] = a*b[rows][column];
            offsetColumn += strideColumn;
        }
        offsetColumn = 0;
        offsetRows += strideRows;
    }
}

template<typename T>
__global__ void divideGPU(T* a, T* b, T* c, int *sizeRows, int* sizeColumn) {
        /*
    Dimensions of matrix && application of the / operator on the template
    will be verified in the class call
    According to IEEE-754 a division by zero will return inf (float type)
    */
    unsigned int rows = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int column = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int strideRows = gridDim.x*blockDim.x
    unsigned int strideColumn = gridDim.y*blockDim.y
    unsigned int offsetRows{}, offsetColumn{};
    while (rows + offsetRows < sizeRows) {
        while(column + offsetColumn < sizeColumn){
            c[rows][column] = a[rows][column] / b[rows][column];
            offsetColumn += strideColumn;
        }
        offsetColumn = 0;
        offsetRows += strideRows;
    }
}

template <typename T, typename F>
__global__ void applyLanmbdaToElementMatrixGPU(T* a, F* b, F function) {
    /*
    Function parameter will be a device side lambda function on the host
    */
    unsigned int rows = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int column = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int strideRows = gridDim.x*blockDim.x
    unsigned int strideColumn = gridDim.y*blockDim.y
    unsigned int offsetRows{}, offsetColumn{};
    while (rows + offsetRows < sizeRows) {
        while(column + offsetColumn < sizeColumn){
            b[rows][column] = function(a[rows][column]);
            offsetColumn += strideColumn;
        }
        offsetColumn = 0;
        offsetRows += strideRows;
    }
}

template <typename T, typename F, typename... Args>
__host__ F* applyLanmbdaToElementMatrix(T* a, F* b, F lambdaFunction, int ROWS, int COLUMN){
    size_t SIZE = ROWS*COLUMN*sizeof(T);
    F* result[ROWS];
    for (unsigned int i = 0; i<ROWS; i++)
        F* result[i] = new F[COLUMN];

    F lamndaGPU = [](Args... param){return lambdaFunction(param)};
    // TO DO 
}

template <typename T>
__host__ T* addHost(T* a, T* b, int ROWS, int COLUMN){
    size_t SIZE = ROWS*COLUMN*sizeof(T);
    T* result[ROWS];
    for (unsigned int i = 0; i<ROWS; i++)
        T* result[i] = new T[COLUMN];

    
    cudaMallocManaged((void**)&result, SIZE);
    cudaMallocManaged((void**)&a, SIZE);
    cudaMallocManaged((void**)&b, SIZE);
    
    dim3 blocksPerGrid() // Amount of blocks 
    dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK) // Amount of threads

    addGPU<<<blocksPerGrid, threadsPerBlock>>>(a, b, result, ROWS, COLUMN);
    cudaDeviceSynchronize();
    
    assert(cudaSuccess==cudaFree(result));
    assert(cudaSuccess==cudaFree(a));
    assert(cudaSuccess==cudaFree(b));

    return result;
        
}

