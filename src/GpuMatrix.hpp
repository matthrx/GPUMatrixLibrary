#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__
#define assertm(exp, msg) assert(((void)msg, exp))

/*all includes */

#include <string>
#include <iostream>

void matrixGPU_init(bool=false);

template <typename T > 
class GpuMatrix {
public:
    unsigned int ROWS;
    unsigned int COLUMNS;
    T* data;

public:
    GpuMatrix(unsigned int, unsigned int);
    GpuMatrix(unsigned int, unsigned int, T*);
    ~GpuMatrix(){};
    void freeMatrixGPU(void);
    T minGpuMatrix(void);
    T maxGpuMatrix(void);
    T meanGpuMatrix(void);
    
    void matrixGPU_print(unsigned int, unsigned int, bool=false, bool=false);

    GpuMatrix<T> transpose(void);
    GpuMatrix<T> dot(GpuMatrix<T>);
    
    GpuMatrix<T> add(GpuMatrix<T>, GpuMatrix<T>);
    GpuMatrix<T> substract(GpuMatrix<T>, GpuMatrix<T>);
    GpuMatrix<T> multiply(GpuMatrix<T>, GpuMatrix<T>);
    GpuMatrix<T> scalarMultiply(T, GpuMatrix<T>);
    GpuMatrix<T> divide(GpuMatrix<T>, GpuMatrix<T>);
    
    static GpuMatrix<double> inverse(GpuMatrix<double>);
    static GpuMatrix<float> inverse(GpuMatrix<float>); 

    static std::tuple<double*, double*, double*> getEigenValues(GpuMatrix<double>, bool=false, bool=false);
    static std::tuple<float*, float*, float*> getEigenValues(GpuMatrix<float>, bool=false, bool=false);

    static std::tuple<double*, double*, double*> getSingularValueDecomposition(GpuMatrix<double>);
    static std::tuple<float*, float*, float*> getSingularValueDecomposition(GpuMatrix<float>);
    

    inline GpuMatrix<T> operator+(GpuMatrix<T>& a){
        return add(*this, a);
    }

    inline GpuMatrix<T> operator-(GpuMatrix<T>& a){
        return substract(*this, a);
    }

    inline GpuMatrix<T> operator*(GpuMatrix<T>& a){
        return multiply(*this, a);
    }

    inline GpuMatrix<T> operator*(T& a){
        return scalarMultiply(a, *this);
    }
    
    // GpuMatrix<T> inverse(void) {
    //     assertm((std::is_same<T, float>::value || std::is_same<T, double>::value), "Error : pleease cast it to double or float");
    //     if (std::is_same<T, float>::value) return inverseMatrixNormalPrecision(*this);
    //     else return GpuMatrix<T>::inverseMatrixHighPrecision(*this);
    // }



    inline GpuMatrix<T> operator/(GpuMatrix<T>& a){
        return divide(*this, a);
    }

};


// template class GpuMatrix<unsigned short int>;
// template class GpuMatrix<signed short int>;
// template class GpuMatrix<unsigned int>;
// template class GpuMatrix<signed int>;
// template class GpuMatrix<unsigned long int>;
// template class GpuMatrix<signed long int>;
// template class GpuMatrix<unsigned long long int>;
// template class GpuMatrix<signed long long int>;

template class GpuMatrix<int>;
template class GpuMatrix<float>;
template class GpuMatrix<double>;
// template class GpuMatrix<long double>;



#endif 