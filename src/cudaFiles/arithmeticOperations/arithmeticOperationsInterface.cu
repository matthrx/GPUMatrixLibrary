#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>       /* ceil */
#include <functional>
// #include <nvfunctional>

// #include "arithmeticOperationsKernel.cuh"

// It would be ideal to transfert data while executing kernel device operations
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define THREAD_PER_BLOCK 32
#define EXIT_SUCESS 0 

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
      std::cerr << cudaGetErrorString(code) << " file : " << file << " line : " <<  line << std::endl;
      if (abort) { exit(code); }
    }
}

template <typename T>
__host__ void addHost(T* a, T* b, T* c, int ROWS, int COLUMN){
    const size_t SIZE = ROWS*COLUMN*sizeof(T);
    T *da, *db, *dc;
 
    gpuErrchk(cudaMalloc((void**)&da, SIZE));
    gpuErrchk(cudaMalloc((void**)&db, SIZE));
    gpuErrchk(cudaMalloc((void**)&dc, SIZE));

    gpuErrchk(cudaMemcpy(da, a, SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b, SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dc, c, SIZE, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);

    addGPU<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, ROWS, COLUMN);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(c,  dc, SIZE, cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(dc));

}

template <typename T, typename F>
__host__ __device__ F lambdaGPU(T x, std::function<F(T&)> func){
    return func(x);
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
    // #pragma unroll
    while (index + offset < amountColumns*amountRows){
        *(b + index + offset) = Func(*(a + index + offset));
        offset += stride;
    }
}


template <typename T, typename F>
__host__ void applyLambdaToElementMatrix(const T* a, std::function<F(T&)> lambdaFunction, int ROWS, int COLUMN){
    const size_t SIZE_T = ROWS*COLUMN*sizeof(T);
    const size_t SIZE_F = ROWS*COLUMN*sizeof(F);
    T* d_a;
    F* d_result;

    F* result = new F[ROWS*COLUMN];
    // int x = 2;
    // lambdaGPU(x , lambdaFunction);
    auto lambdaFunctionOnGPU = [=]  __host__ __device__  (T& x) -> F {return pow(x,2);}; // need of --expt-extended-lambda
    
    // std::cout  << lambdaFunctionOnGPU(x) << std::endl;
    gpuErrchk(cudaMalloc((void**)&d_a, SIZE_T));
    gpuErrchk(cudaMalloc((void**)&d_result, SIZE_F));
    
    gpuErrchk(cudaMemcpy(d_a, a, SIZE_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_result, result, SIZE_F, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(ceil(ROWS/THREAD_PER_BLOCK), ceil(COLUMN/THREAD_PER_BLOCK));

    // dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    std::cout << *(a + 20000) << std::endl;

    applyLambdaToElementMatrixGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_result, lambdaFunctionOnGPU, ROWS, COLUMN);
    // applyLambdaToElementMatrixGPU<<<(1, 1), (1, 1) >>>(d_a, d_result, lambdaFunctionOnGPU, ROWS, COLUMN);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(result, d_result, SIZE_T, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_result));

    std::cout << *(result + 20000) << std::endl;
    // return result;
}


// double carre(int x){
//     return pow(x, 2);
// }

int main(void){
    const long int ROWS_MATRIX = 32000;
    const long int COLUMNS_MATRIX = 32000;
    int* matrix = new int[ROWS_MATRIX*COLUMNS_MATRIX];

    auto carre = [](int x){return pow(x,2);};
    for (unsigned int i = 0; i < (ROWS_MATRIX*COLUMNS_MATRIX); i ++){
        *(matrix + i) = reinterpret_cast<int>(rand()%10);
    }
    std::cout << *(matrix+12000) << std::endl;

    applyLambdaToElementMatrix<int, double>(matrix, carre, ROWS_MATRIX, COLUMNS_MATRIX);
    std::cout << "Stop..." << std::flush;
    for (unsigned int i = 0; i < (ROWS_MATRIX*COLUMNS_MATRIX); i ++){
        *(matrix + i) = carre(*(matrix + i));
    }
    std::cout << *(matrix+120000) << std::endl;
    // std::cout << std::to_string(carre) << std::endl;
    delete [] matrix;
    return EXIT_SUCESS;
}
