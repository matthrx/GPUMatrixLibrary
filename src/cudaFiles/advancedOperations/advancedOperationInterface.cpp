#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"

#define EXIT_SUCESS 0
#define assertm(exp, msg) assert(((void)msg, exp))

typedef struct {
    size_t ROWS;
    size_t COLUMNS;
    double* data;
} Matrix;

template <typename T>
bool isDoublePointer(T *t){return false;}
bool isDoublePointer(double *t){return true;}

template<typename T>
bool isFloatPointer(T *t){return false;}
bool isFloatPointer(float *t) {return true;}


Matrix inverseMatrixHighPrecision(Matrix m) {
    //It uses the LU decomposition with partial pivoting and row interchanges
    assertm(isDoublePointer(m.data), "Types compatible, expecting a double for the data of matrix");
    magma_init();
    double gpu_time,*da, *dataToReturn, *dwork;
    // double *a; //matrix host side
    magma_int_t blockSize;
    magma_int_t dev = 0;
    magma_int_t err = 0; 
    magma_int_t info = 0;
    magma_int_t fullSize = m.ROWS*m.COLUMNS;
    magma_int_t *pivot = new magma_int_t[m.ROWS];

    magma_queue_t queue = NULL;
    magma_queue_create(dev, &queue);

    blockSize = m.ROWS * magma_get_dgetri_nb(m.COLUMNS);

    err = magma_dmalloc_cpu(&dataToReturn, fullSize);
    err = magma_dmalloc(&da, fullSize);
    err = magma_dmalloc(&dwork, blockSize);
  
    // magma_dprint(m.ROWS, m.COLUMNS, m.data, 8);

    magma_dsetmatrix(m.ROWS, m.COLUMNS, m.data, m.ROWS, da, m.COLUMNS, queue); // put matrix data in da for device (GPU) -- da*X = Iusing LU factorization
    // magmablas_dlacpy(MagmaFull, m.ROWS, m.COLUMNS, da, m.ROWS, dr, m.COLUMNS, queue); // put da in dr (copy of data) to test A*A⁽-1)
    // gpu_time = magma_sync_wtime(NULL); // debug is 

    magma_dgetrf_gpu(m.ROWS, m.COLUMNS, da, m.ROWS, pivot, &info);
    magma_dgetri_gpu(m.ROWS, da, m.COLUMNS, pivot, dwork, blockSize, &info); // inverse is in da
    // magma_dgemm(MagmaNoTrans, MagmaNoTrans, m.COLUMNS, m.ROWS, m.COLUMNS, 1, da, m.ROWS, dr, m.COLUMNS, 0, dtest, m.ROWS, queue);
    // gpu_time = magma_sync_wtime(NULL) - gpu_time;
    // std::cout << "Time taken for the operation : " << gpu_time << " sec" << std::endl; 

    magma_dgetmatrix(m.ROWS, m.COLUMNS, da, m.ROWS, dataToReturn, m.COLUMNS, queue);
    // magma_dprint(m.ROWS, m.COLUMNS, m.data, 8);
    // free(a);
    free(pivot);
    magma_free(dataToReturn);
    magma_free(da);
    magma_queue_destroy(queue);                     //  destroy  queuemagma_finalize ();    
    magma_finalize();


    return Matrix {
        m.ROWS, m.COLUMNS, dataToReturn
    };
}

std::tuple<double*,double*> eigenValuesHighPrecision(Matrix m, bool isSymetric=false) {
    assertm(isDoublePointer(m.data), "Types compatible, expecting a double for the data of matrix");
    magma_init();
    magma_queue_t queue = NULL;
    magma_int_t blockSize;
    magma_int_t dev = 0;
    magma_int_t err = 0; 
    magma_int_t info = 0;
    magma_int_t partialSize = m.ROWS;
    magma_int_t fullSize = partialSize*partialSize;

    double *h_work;
    magma_int_t lwork, liwork; //liwork is size of iwork

    double *dr, *r; // m.data will be in dr 
    double *v1;// vectors of eigenvalues only real part (not complex)
    magma_queue_create(dev, &queue);
    // blockSize = m.ROWS * magma_get_dgetri_nb(m.COLUMNS);

    err = magma_dmalloc_cpu(&r, fullSize);
    // err = magma_dmalloc_cpu(&toReturn, fullSize);
    err = magma_dmalloc_cpu(&v1, partialSize);
    err = magma_dmalloc(&dr, fullSize);

    double aux_work[1]; // defining workspace for computation
    magma_int_t aux_iwork[1];
    if (!(isSymetric))
        magma_dsyevd_gpu(MagmaNoVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);
    else 
        magma_dsyevd_gpu(MagmaVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);

    lwork = (magma_int_t) aux_work[0];
    liwork = aux_iwork[0];
    magma_int_t* iwork = new magma_int_t[liwork];
    magma_dmalloc_cpu(&h_work, lwork);

    magma_dsetmatrix(partialSize, partialSize, m.data, partialSize, dr, partialSize, queue);
    double gpu_time = magma_sync_wtime(NULL);
    if (!(isSymetric))
        magma_dsyevd_gpu(MagmaNoVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, h_work, lwork, iwork, liwork, &info);
    else 
        magma_dsyevd_gpu(MagmaVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, h_work, lwork, iwork, liwork, &info);
    // In v1 we have the eigenvalues and if it's symetric we'll have the orthogonal eigenvectors in r
    free(r);
    free(v1);
    free(iwork);
    free(h_work);
    magma_free(dr);
    magma_queue_destroy(queue);                     //  destroy  queuemagma_finalize (); 
    magma_finalize();

    return std::make_tuple(v1, r); 

}

std::tuple<float*,float*> eigenValuesNormalPrecision(Matrix m, bool isSymetric=false) {
    assertm(isFloatPointer(m.data), "Types compatible, expecting a float for the data of matrix");
    magma_init();
    magma_queue_t queue = NULL;
    magma_int_t blockSize;
    magma_int_t dev = 0;
    magma_int_t err = 0; 
    magma_int_t info = 0;
    magma_int_t partialSize = m.ROWS;
    magma_int_t fullSize = partialSize*partialSize;

    float *h_work;
    magma_int_t lwork, liwork; //liwork is size of iwork

    float *dr, *r; // m.data will be in dr 
    float *v1; // vectors of eigenvalues only real part (not complex)
    magma_queue_create(dev, &queue);
    // blockSize = m.ROWS * magma_get_dgetri_nb(m.COLUMNS);

    err = magma_smalloc_cpu(&r, fullSize);
    // err = magma_dmalloc_cpu(&toReturn, fullSize);
    err = magma_smalloc_cpu(&v1, partialSize);
    err = magma_smalloc(&dr, fullSize);

    float aux_work[1]; // defining workspace for computation
    magma_int_t aux_iwork[1];
    if (!(isSymetric))
        magma_ssyevd_gpu(MagmaNoVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);
    else 
        magma_ssyevd_gpu(MagmaVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);

    lwork = (magma_int_t) aux_work[0];
    liwork = aux_iwork[0];
    magma_int_t* iwork = new magma_int_t[liwork];
    magma_smalloc_cpu(&h_work, lwork);

    magma_ssetmatrix(partialSize, partialSize, reinterpret_cast<float*>(m.data), partialSize, dr, partialSize, queue);
    if (!(isSymetric))
        magma_ssyevd_gpu(MagmaNoVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, h_work, lwork, iwork, liwork, &info);
    else 
        magma_ssyevd_gpu(MagmaVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, h_work, lwork, iwork, liwork, &info);

    free(r);
    free(v1);
    free(iwork);
    free(h_work);
    magma_free(dr);
    magma_queue_destroy(queue);                     //  destroy  queuemagma_finalize (); 
    magma_finalize();

    return std::make_pair(v1, r);

}



int main(void){
    std::cout << "Working..." << std::endl;
    Matrix matrix = Matrix{10000, 10000, new double[10000*10000]};
      
    for (unsigned int i = 0; i < matrix.ROWS*matrix.COLUMNS; i++)
        *(matrix.data + i) = rand()%100;
    // inverseMatrix(matrix);
    eigenValuesHighPrecision(matrix);
    return EXIT_SUCESS;
}