#include <iostream>

#include "GpuMatrix.h"

int main(void){
    matrixGPU_init(false);

    GpuMatrix<double> matrixA = GpuMatrix<double>(1200, 1200);
    GpuMatrix<double> matrixB = GpuMatrix<double>(1200, 1200);

    for (unsigned int i = 0; i<(matrixA.ROWS*matrixA.COLUMNS); i++){
        *(matrixA.data + i) = 2;
        *(matrixB.data + i) = 3;
    }

    matrixA.data[10000] = 1;
    std::cout << "..." << std::endl;
    GpuMatrix<double> matrixC = matrixA * matrixB;
    double min = matrixC.minGpuMatrix();
    std::cout << "MatA of function is " << matrixA.data[0] << std::endl;
    std::cout << "Min is " << min << std::endl;

    std::cout << "MatC of function is " << matrixC.data[0] << std::endl;

    matrixA.free();
    matrixB.free();
    return 0;
}