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


    begin_measure = std::chrono::steady_clock::now();
    arma::mat D = A+B;
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU sum operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    begin_measure = std::chrono::steady_clock::now();
    arma::mat E = arma::inv(A);
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU inverse operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    begin_measure = std::chrono::steady_clock::now();
    double F = A.min();
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU min operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    begin_measure = std::chrono::steady_clock::now();
    double G = B.max();
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU max operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    begin_measure = std::chrono::steady_clock::now();
    auto H = arma::mean(C);
    end_measure = std::chrono::steady_clock::now();
    std::cout << "Time for CPU mean operation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_measure - begin_measure).count() << "[ms]" << std::endl;


    return 0;
}