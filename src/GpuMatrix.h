#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__
/*all includes */

#include <string>

void matrixGPU_init(bool);

template <typename T> class GpuMatrix {
public:
    unsigned int ROWS;
    unsigned int COLUMNS;
    T* data;

public:
    GpuMatrix(unsigned int rows, unsigned int colums);
    GpuMatrix(unsigned int rows, unsigned int columns, T* data);
    ~GpuMatrix(){};
    void free(void);
    T minGpuMatrix(void);
    T maxGpuMatrix(void);
    T meanGpuMatrix(void);
    static GpuMatrix<T> dot(GpuMatrix<T> a, GpuMatrix<T> b);
    
    void matrixGPU_print(unsigned int rows, unsigned int columns);
    void matrixGPU_print(unsigned int rows, unsigned int columns, bool);

    void matrixGPU_print(unsigned int rows, unsigned int columns, bool, bool);
    GpuMatrix<T> transpose(void);
    static GpuMatrix<T> inverseMatrixHighPrecision(GpuMatrix<T> a);
    static GpuMatrix<T> inverseMatrixNormalPrecision(GpuMatrix<T> a);
    
    GpuMatrix<T> add(GpuMatrix<T>, GpuMatrix<T>);
    GpuMatrix<T> substract(GpuMatrix<T>, GpuMatrix<T>);
    GpuMatrix<T> multiply(GpuMatrix<T>, GpuMatrix<T>);
    GpuMatrix<T> scalarMultiply(T, GpuMatrix<T>);
    GpuMatrix<T> divide(GpuMatrix<T>, GpuMatrix<T>);

    std::tuple<float*, float*, float*> getEigenValuesNormalPrecisionNormalMatrix(void);
    std::tuple<double*, double*, double*> getEigenValuesHighPrecisionNormalMatrix(void);
    std::tuple<float*, float*, float*> getEigenValuesNormalPrecisionNormalMatrix(bool);
    std::tuple<double*, double*, double*> getEigenValuesHighPrecisionNormalMatrix(bool);
    std::tuple<float*, float*, float*> getEigenValuesNormalPrecisionNormalMatrix(bool, bool);
    std::tuple<double*, double*, double*> getEigenValuesHighPrecisionNormalMatrix(bool, bool);
    std::tuple<double*, double*, double*> getSingularValueDecompositionHighPrecision(void);
    std::tuple<float*, float*, float*> getSingularValueDecompositionNormalPrecision(void);
    

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