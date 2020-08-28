// Using a BLACK & LAPACK optimization (through Armadillo)
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


    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    arma::mat D = A+B;
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU sum operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    arma::mat C = arma::inv(A);
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU inverse operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    double C = A.min();
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU min operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    double D = B.max();
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU max operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    std::chrono::steady_clock::time_point begin_measure = std::chrono::steady_clock::now();
    auto E = arma::mean(C);
    std::chrono::steady_clock::time_point end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU mean operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    return 0;
}