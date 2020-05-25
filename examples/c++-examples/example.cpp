/*
Compiled with the following instruction
g++ -Wall -L/usr/local/gpuMatrix-1.0.0/lib -I/usr/local/gpuMatrix-1.0.0/include -o example example.cpp -lgpumatrix
*/

#include <iostream>
#include <cstdlib>
#include "GpuMatrix.hpp"

int main(void){
    matrixGPU_init(false);

    GpuMatrix<double> matrixA = GpuMatrix<double>(1200, 1200);
    GpuMatrix<double> matrixB = GpuMatrix<double>(1200, 1200);

    for (unsigned int i = 0; i<(matrixA.ROWS*matrixA.COLUMNS); i++){
        *(matrixA.data + i) = rand()%10;
        *(matrixB.data + i) = rand()%10;
    }

    matrixA.matrixGPU_print(10, 10);
    
    GpuMatrix<double> matrixC = matrixA.dot(matrixB);
    GpuMatrix<double> matrixD = matrixA.transpose();
    double minMatrixA = matrixA.minGpuMatrix();
    std::cout << "Minimum of matrix A is " << minMatrixA << std::endl;

    matrixC.matrixGPU_print(10, 10);
    //matrixD.matrixGPU_print(10, 10);

    matrixA.freeMatrixGPU();
    matrixB.freeMatrixGPU();
    return 0;
}