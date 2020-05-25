#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <regex>
#include <cuda_runtime.h>
#include <assert.h>

#include "../../GpuMatrix.hpp"
#include "generalInformation.hpp"
// #include <helper_cuda.h>
// #include <helper_functions.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define assertm(exp, msg) assert(((void)msg, exp))

// typedef struct {
//     int warpsSize;
//     int maxThreadsPerBlock;
//     size_t totalGlobalMem;
//     int maxThreadsDim[3];
//     int maxGridSize[3];
//     int maxThreadsPerMultiProcessor;

// } KernelConfiguration;
cudaDeviceProp deviceProps;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
      std::cerr << "Error during configuration : " << cudaGetErrorName(code) << "\nDefinition error : " << cudaGetErrorString(code) << std::endl;
      if (abort) { exit(code); }
    }
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    assertm(pipe, "popen() failed");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

void sofwareVerification(void){
    std::cout << "Currently processing to the device verification and configuration for the GPU operations " << std::endl;
    // #if !defined(nvcc)
    // #error "nvcc is not defined as an environnement variable, please do it "
    // #endif
    const char* cmd = "nvcc --version";
    std::string result = exec(cmd);
    std::regex getVersion("\\sV(.*)");
    std::smatch version;
    // std::cout << version[1].str() << std::endl;
    std::regex_search(result, version, getVersion);
    float numericVersion{};
    try {
        numericVersion = std::stof(version[1].str());
    }
    catch (std::out_of_range& e){
        std::cerr << "Error regarding the access to cuda software please make sure nvcc is defined as environnement variable" << std::endl;
        exit(0);
    }
    if (numericVersion <= 8) {
        std::cerr << "Your cuda version is obsolete please upgrade it to at least version 8.0" << std::endl;
        exit(0);
    }
    std::cout << "Cuda sotware identified ++++++++++++++++++\n" << result << std::endl;

}

void matrixGPU_init(bool verbose = false){
    if (!(verbose)){
        std::cout.setstate(std::ios_base::failbit);
    }
    sofwareVerification();
    int* device = new int;
    gpuErrchk(cudaGetDeviceCount(device));
    if (*device > 1) {
        std::cout << "Warning this library only uses one GPU, wait for an update..." << std::endl;
    }
    gpuErrchk(cudaGetDevice(device));
    gpuErrchk(cudaGetDeviceProperties(&deviceProps, *device));
    std::cout << "CUDA device " << deviceProps.name << std::endl;
    delete device;
    if (!(verbose)){
        std::cout.clear();
    }

}

template <typename T>
void GpuMatrix<T>::matrixGPU_print(unsigned int rows, unsigned int columns, bool isTop, bool isLeft){
    assertm((this->ROWS >= rows && this->COLUMNS >= columns), "error : rows and columns in arguments are higher than matrix size ");
    std::cout << "Matrix dimensions are [" << this->ROWS << "," << this->COLUMNS << "] - displaying a "<< rows << "x" << columns << " insight //" << std::endl;
    switch (isTop){
        case true:
            if (isLeft) {
                // in this case we'll print the rows first rows and the columns first columns
                for (unsigned int i = 0; i<rows; i++){
                    for (unsigned int j = 0; j<columns; j++){
                        std::cout.width(5); std::cout << this->data[i*this->COLUMNS+j] << " " << std::flush;
                    }
                    std::cout << std::endl;
                }
            }
            else {
                // in this case we'll print the rows first rows and the columns last columns
                for (unsigned int i = 1; i<rows; i++){
                    for (unsigned int j = (this->COLUMNS-columns); j<this->COLUMNS; j++){
                        std::cout.width(5); std::cout << this->data[i*this->COLUMNS+j] << "  " << std::flush;
                }
                std::cout << std::endl;

            }
            }
            break;
        case false:
            if (isLeft){
                for (unsigned int i = this->ROWS-rows; i<this->ROWS; i++){
                    for (unsigned int j = 0; j<columns; j++){
                        std::cout.width(5); std::cout << this->data[i*this->COLUMNS+j] << " " << std::flush;
                    }
                    std::cout << std::endl;
                }
            }
            else {
                for (unsigned int i = this->ROWS-rows; i<this->ROWS; i++){
                    for (unsigned int j = (this->COLUMNS-columns); j<this->COLUMNS; j++){
                        std::cout.width(5); std::cout << this->data[i*this->COLUMNS+j] << " " << std::flush;
                    }
                    std::cout << std::endl;
            }
            }
            break;
            }
        printf("\n\n");
}
    

