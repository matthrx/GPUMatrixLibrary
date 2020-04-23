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

// typedef struct {
//     int warpsSize;
//     int maxThreadsPerBlock;
//     size_t totalGlobalMem;
//     int maxThreadsDim[3];
//     int maxGridSize[3];
//     int maxThreadsPerMultiProcessor;

// } KernelConfiguration;

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
    cudaDeviceProp deviceProps;
    gpuErrchk(cudaGetDeviceProperties(&deviceProps, *device));
    std::cout << "CUDA device " << deviceProps.name << std::endl;
    cudaDeviceProp* kernelConfig = &deviceProps;
}

int main(int argc, char *argv[]){
    initialisation();
    return 0;
}