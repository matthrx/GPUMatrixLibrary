// Using a BLACK & LAPACK optimization
#include <armadillo>
#include <chrono>
#include <iostream>

int main(void){
    const int ROWS = 300;
    const int COLUMNS = 300;

    arma::mat A = arma::randu(ROWS, COLUMNS);
    arma::mat B = arma::randu(ROWS, COLUMNS);
    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    arma::mat C = A*B;
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU dot operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;

    return 0;
}