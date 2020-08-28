/*
Compiled with the following instruction
g++ -Wall -L/usr/local/gpuMatrix-1.0.0/lib -I/usr/local/gpuMatrix-1.0.0/include -o gpu_example gpu_example.cpp -lgpumatrix
*/

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <string>
#include "GpuMatrix.hpp"


int main(void){
    matrixGPU_init(true);

    const int COLUMNS = 300;
    const int ROWS = 300;
    GpuMatrix<double> matrixA = GpuMatrix<double>(ROWS, COLUMNS);
    GpuMatrix<double> matrixB = GpuMatrix<double>(ROWS, COLUMNS);

    for (unsigned int i = 0; i<(matrixA.ROWS*matrixA.COLUMNS); i++){
        *(matrixA.data + i) = rand()%10;
        *(matrixB.data + i) = rand()%10;
    }
    matrixA.matrixGPU_print(10, 10);
    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    GpuMatrix<double> matrixC = matrixA.dot(matrixB);
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for GPU dot operation = " << std::chrono::duration_cast<std::chrono::microseconds>(end_measure - begin_measure).count() << "[us]" << std::endl;

    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Sum operation" << std::endl;
    begin_measure = std::chrono::steady_clock::now();
    GpuMatrix<double> matrixD = matrixA + matrixB;
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for GPU sum operation = " << std::chrono::duration_cast<std::chrono::microseconds>(end_measure - begin_measure).count() << "[us]" << std::endl;

    begin_measure = std::chrono::steady_clock::now();
    GpuMatrix<double> matrixE = matrixA*matrixB;
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for product full matrix on GPU is" << std::chrono::duration_cast<std::chrono::microseconds>(end_measure - begin_measure).count() <<  "[us]" << std::endl;


    begin_measure = std::chrono::steady_clock::now();
    double minMatrixA = matrixA.minGpuMatrix();
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for minimum on GPU" << std::chrono::duration_cast<std::chrono::microseconds>(end_measure - begin_measure).count() << " [us]" << std::endl;

    begin_measure = std::chrono::steady_clock::now();
    double maxMatrixD = matrixB.maxGpuMatrix();
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for maximum on GPU is " << std::chrono::duration_cast<std::chrono::microseconds>(end_measure - begin_measure).count() <<  "[us]" << std::endl;


    begin_measure = std::chrono::steady_clock::now();
    double maxMatrixE = matrixE.meanGpuMatrix();
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for mean on GPU is " << std::chrono::duration_cast<std::chrono::microseconds>(end_measure - begin_measure).count() <<  "[us]" << std::endl;


    matrixA.freeMatrixGPU();
    matrixB.freeMatrixGPU();
    return 0;
}