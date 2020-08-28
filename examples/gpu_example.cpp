/*
Compiled with the following instruction
g++ -Wall -L/usr/local/gpuMatrix-1.0.0/lib -I/usr/local/gpuMatrix-1.0.0/include -o example example.cpp -lgpumatrix
*/

#include <iostream>
#include <cstdlib>
#include <chrono>
#include "GpuMatrix.hpp"


int main(void){
    matrixGPU_init(false);

    GpuMatrix<double> matrixA = GpuMatrix<double>(10, 10);
    GpuMatrix<double> matrixB = GpuMatrix<double>(300, 300);

    for (unsigned int i = 0; i<(matrixA.ROWS*matrixA.COLUMNS); i++){
        *(matrixA.data + i) = rand()%10;
        *(matrixB.data + i) = rand()%10;
    }
    matrixA.matrixGPU_print(10, 10);
    matrixA.data[5]=-3;
    std::cout << matrixA.minGpuMatrix() << std::endl;
    matrixA.matrixGPU_print(10, 10);
    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    GpuMatrix<double> matrixC = matrixA.dot(matrixB);
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for GPU dot operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;

    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Sum operation" << std::endl;
    begin_measure = std::chrono::steady_clock::now();
    GpuMatrix<double> matrixD = matrixA + matrixB;
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for GPU sum operation = " << std::chrono::duration_cast<std::chrono::microseconds>(end_measure - begin_measure).count() << "[us]" << std::endl;

    double minMatrixA = matrixA.minGpuMatrix();
    std::cout << "Minimum of matrix A is " << minMatrixA << std::endl;

    matrixA.matrixGPU_print(10, 10);
    matrixA.freeMatrixGPU();
    matrixB.freeMatrixGPU();
    matrixD.freeMatrixGPU();

    return 0;
}