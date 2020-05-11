#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <regex>
#include <cuda_runtime.h>
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
extern cudaDeviceProp deviceProps;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
      std::cerr << "Error during configuration : " << cudaGetErrorName(code) << "\nDefinition error : " << cudaGetErrorString(code) << std::endl;
      if (abort) { exit(code); }
    }
}

__host__ std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

__host__ void sofwareVerification(void){
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
    }
    std::cout << "Cuda sotware identified ++++++++++++++++++\n" << result << std::endl;

}
__host__ void initialisation(void){
    sofwareVerification();
    int* device = new int;
    gpuErrchk(cudaGetDeviceCount(device));
    if (*device > 1) {
        std::cout << "Warning this library only uses one GPU, wait for an update..." << std::endl;
    }
    gpuErrchk(cudaGetDevice(device));
    gpuErrchk(cudaGetDeviceProperties(&deviceProps, *device));
    std::cout << "CUDA device " << deviceProps.name << std::endl;


}

template <typename T>
struct Matrix {
    size_t ROWS;
    size_t COLUMNS;
    T* data;
};

template <typename T> 
__host__ void gpuPrint(Matrix<T> a, unsigned int rows, unsigned int columns, bool isTop = true, bool isLeft = true){
    assertm((a.ROWS >= rows && a.COLUMNS >= columns), "error : rows and columns in arguments are higher than matrix size ");
    std::cout << "Matrix dimensions are [" << a.ROWS << "," << a.COLUMNS << "] - displaying a "<< rows << "x" << columns << " insight //" << std::endl;
    switch (isTop){
        case true:
            if (isLeft) {
                // in this case we'll print the rows first rows and the columns first columns
                for (unsigned int i = 0; i<rows; i++){
                    for (unsigned int j = 0; j<columns; j++){
                        std::cout.width(5); std::cout << a.data[i*a.COLUMNS+j] << " " << std::flush;
                    }
                    std::cout << std::endl;
                }
            }
            else {
                // in this case we'll print the rows first rows and the columns last columns
                for (unsigned int i = 1; i<rows; i++){
                    std::cout << std::endl;
                    for (unsigned int j = (a.COLUMNS-columns); j<a.COLUMNS; j++){
                        std::cout << a.data[i*a.COLUMNS+j] << "  " << std::flush;
                }
            }
            break;
        case false:
            if (isLeft){
                for (unsigned int i = a.ROWS-rows; i<a.ROWS; i++){
                    std::cout << std::endl;
                    for (unsigned int j = 0; j<columns; j++){
                        std::cout << a.data[i*a.COLUMNS+j] << " " << std::flush;
                    }
                }
            }
            else {
                for (unsigned int i = a.ROWS-rows; i<a.ROWS; i++){
                    for (unsigned int j = (a.COLUMNS-columns); j<a.COLUMNS; j++){
                        std::cout.width(20); std::cout << a.data[i*a.COLUMNS+j] << " " << std::flush;
                    }
                    std::cout << std::endl;
            }
            break;
            }
        }
    }
}
