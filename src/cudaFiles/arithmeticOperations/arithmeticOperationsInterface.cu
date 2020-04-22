#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>       /* ceil */
#include <functional>

#include "arithmeticOperationsKernel.cuh"

// It would be ideal to transfert data while executing kernel device operations
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define THREAD_PER_BLOCK 32

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
__host__ void applyLambdaToElementMatrix(T* a, std::function<F(T&)> lambdaFunction, int ROWS, int COLUMN){
    const size_t SIZE_T = ROWS*COLUMN*sizeof(T);
    const size_t SIZE_F = ROWS*COLUMN*sizeof(F);
    T* d_a;
    F* d_result;

    F* result = new F[ROWS*COLUMN];
    F lambdaGPU = [] __device__  (T& x){return lambdaFunction(x);} // need of –expt-extended-lambda
    
    gpuErrchk(cudaMalloc((void**)&d_a, SIZE_T));
    gpuErrchk(cudaMalloc((void**)&d_result, SIZE_F));
    
    gpuErrchk(cudaMemcpy(d_a, a, SIZE_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_result, result, SIZE_F, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);

    applyLambdaToElementMatrixGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_result, ROWS, COLUMN);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(result, d_result, SIZE_T, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_result));

    std::cout << *(result + 20000) << std::endl;

}

int main(void){
    const int ROWS_MATRIX = 4096;
    const int COLUMNS_MATRIX = 4096;
    int* matrix = new int[ROWS_MATRIX*COLUMNS_MATRIX];
    auto carre = [](int &x){return pow(x,2);}

    for (unsigned int i = 0; i < ROWS_MATRIX * COLUMNS_MATRIX; i ++){
        *(matrix + i) = reinterpret_cast<int>(rand()%10);
    }
    applyLambdaToElementMatrix<int, int>(matrix, carre, ROWS_MATRIX, COLUMNS_MATRIX);
    delete [] matrix;
    return 0;
}
