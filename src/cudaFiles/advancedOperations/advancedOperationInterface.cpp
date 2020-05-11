#include <iostream>
#include <tuple>
#include <cstdlib>
#include <cuda.h>
#include  <cuda_runtime.h>
#include "magma_v2.h"
#include "magma_lapack.h"

#include "../../GPUOperations.h"

#define EXIT_SUCESS 0
#define assertm(exp, msg) assert(((void)msg, exp))
#define max(a,b) (((a) > (b)) ? (a) : (b)) 
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define carre(x) (x*x)

template <typename T>
struct Matrix {
    size_t ROWS;
    size_t COLUMNS;
    T* data;
}; 

template <typename T>
bool isDoublePointer(T *t){return false;}
bool isDoublePointer(double *t){return true;}

template<typename T>
bool isFloatPointer(T *t){return false;}
bool isFloatPointer(float *t) {return true;}

template <typename T>
Matrix<T> inverseMatrixHighPrecision(Matrix<T> m) {
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


    return Matrix<double> {
        m.ROWS, m.COLUMNS, dataToReturn
    };
}

template <typename T>
Matrix<T> inverseMatrixNormalPrecision(Matrix<T> m) {
    //It uses the LU decomposition with partial pivoting and row interchanges
    assertm(isFloatPointer(m.data), "Types compatible, expecting a float for the data of matrix");
    magma_init();
    float gpu_time,*da, *dataToReturn, *dwork;
    // double *a; //matrix host side
    magma_int_t blockSize;
    magma_int_t dev = 0;
    magma_int_t err = 0; 
    magma_int_t info = 0;
    magma_int_t fullSize = m.ROWS*m.COLUMNS;
    magma_int_t *pivot = new magma_int_t[m.ROWS];

    magma_queue_t queue = NULL;
    magma_queue_create(dev, &queue);

    blockSize = m.ROWS * magma_get_sgetri_nb(m.COLUMNS);

    err = magma_smalloc_cpu(&dataToReturn, fullSize);
    err = magma_smalloc(&da, fullSize);
    err = magma_smalloc(&dwork, blockSize);
  
    // magma_dprint(m.ROWS, m.COLUMNS, m.data, 8);

    magma_ssetmatrix(m.ROWS, m.COLUMNS, m.data, m.ROWS, da, m.COLUMNS, queue); // put matrix data in da for device (GPU) -- da*X = Iusing LU factorization
    // magmablas_dlacpy(MagmaFull, m.ROWS, m.COLUMNS, da, m.ROWS, dr, m.COLUMNS, queue); // put da in dr (copy of data) to test A*A⁽-1)
    // gpu_time = magma_sync_wtime(NULL); // debug is 

    magma_sgetrf_gpu(m.ROWS, m.COLUMNS, da, m.ROWS, pivot, &info);
    magma_sgetri_gpu(m.ROWS, da, m.COLUMNS, pivot, dwork, blockSize, &info); // inverse is in da
    // magma_dgemm(MagmaNoTrans, MagmaNoTrans, m.COLUMNS, m.ROWS, m.COLUMNS, 1, da, m.ROWS, dr, m.COLUMNS, 0, dtest, m.ROWS, queue);
    // gpu_time = magma_sync_wtime(NULL) - gpu_time;
    // std::cout << "Time taken for the operation : " << gpu_time << " sec" << std::endl; 

    magma_sgetmatrix(m.ROWS, m.COLUMNS, da, m.ROWS, dataToReturn, m.COLUMNS, queue);
    // magma_dprint(m.ROWS, m.COLUMNS, m.data, 8);
    // free(a);
    free(pivot);
    magma_free(da);
    magma_queue_destroy(queue);                     //  destroy  queuemagma_finalize ();    
    magma_finalize();


    Matrix<T> toReturn;
    toReturn.ROWS = a.ROWS;
    toReturn.COLUMNS = a.COLUMNS;
    toReturn.data = dataToReturn;
    return toReturn;
}
// std::tuple<double*,double*> 
template <typename T>
std::tuple<double*, double*, double*, double*> eigenValuesHighPrecisionNormalMatrix(Matrix<T> m, bool leftEigenvector = false, bool rightEigenvector = false) {
    /*
    Ax=λx, where x is right eigenvector, while in yA=λy, y is left eigenvector.
    If computed, the left eigenvectors arestored in columns of an array el and the right eigenvectors in columns of er. 
    */
    assertm(isDoublePointer(m.data), "Types compatible, expecting a double for the data of matrix");
    magma_init();
    magma_queue_t queue = NULL;
    magma_int_t blockSize;
    magma_int_t dev = 0;
    magma_int_t err = 0; 
    magma_int_t info = 0;
    magma_int_t partialSize = m.ROWS;
    magma_int_t fullSize = partialSize*partialSize;

    double *hwork;
    magma_int_t lwork; //liwork is size of iwork

    double *r, *a; // m.data will be in r
    double *er, *el; // left and right eigenvectors of 
    double *vr1, *vi1; //host memory only real and imaginary
    magma_queue_create(dev, &queue);
    blockSize = magma_get_dgehrd_nb(partialSize);
    lwork = partialSize*(2+blockSize);
    lwork = max(lwork , partialSize*(5+2*partialSize));

    magma_dmalloc_cpu (&vr1 ,partialSize);             
    magma_dmalloc_cpu (&vi1 ,partialSize);              
    magma_dmalloc_pinned (&r, fullSize);
    magma_dmalloc_pinned (&el ,fullSize);
    magma_dmalloc_pinned (&er ,fullSize);   
    magma_dmalloc_pinned (&hwork ,lwork);
    // lapackf77_dlarnv (&ione ,ISEED ,&fullSize ,a);
    lapackf77_dlacpy(MagmaFullStr, &partialSize, &partialSize, m.data, &partialSize, r, &partialSize);
    // std::cout << *(r + m.COLUMNS) << std::endl;
    if (leftEigenvector && rightEigenvector)
        magma_dgeev(MagmaVec, MagmaVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    else if (!(leftEigenvector) && rightEigenvector)
        magma_dgeev(MagmaNoVec, MagmaVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    else if (leftEigenvector && !(rightEigenvector))
        magma_dgeev(MagmaVec, MagmaNoVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    else 
        magma_dgeev(MagmaNoVec, MagmaNoVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    free(vr1);   
    free(vi1);
    magma_free_pinned(hwork);
    magma_free_pinned(er);
    magma_free_pinned(el);
    magma_free_pinned(r);

    magma_queue_destroy(queue);                    
    magma_finalize();
    return std::tuple<double*, double*, double*, double*>(vr1, vi1, el, er); 

}

template <typename T>
std::tuple<float*, float*, float*, float*> eigenValuesNormalPrecisionNormalMatrix(Matrix<T> m, bool leftEigenvector = false, bool rightEigenvector = false) {
    /*
    Ax=λx, where x is right eigenvector, while in yA=λy, y is left eigenvector. (eigenvectors are normalized)
    If computed, the left eigenvectors arestored in columns of an array el and the right eigenvectors in columns of er. 
    */
    assertm(isFloatPointer(m.data), "Types compatible, expecting a double for the data of matrix");
    magma_init();
    magma_queue_t queue = NULL;
    magma_int_t blockSize;
    magma_int_t dev = 0;
    magma_int_t err = 0; 
    magma_int_t info = 0;
    magma_int_t partialSize = m.ROWS;
    magma_int_t fullSize = partialSize*partialSize;

    float *hwork;
    magma_int_t lwork; //liwork is size of iwork

    float *r, *a; // m.data will be in r
    float *er, *el; // left and right eigenvectors of 
    float *vr1, *vi1; //host memory only real and imaginary
    magma_queue_create(dev, &queue);
    blockSize = magma_get_sgehrd_nb(partialSize);
    lwork = partialSize*(2+blockSize);
    lwork = max(lwork , partialSize*(5+2*partialSize));

    magma_smalloc_cpu (&vr1 ,partialSize);             
    magma_smalloc_cpu (&vi1 ,partialSize);             
    magma_smalloc_pinned (&r, fullSize);
    magma_smalloc_pinned (&el ,fullSize);
    magma_smalloc_pinned (&er ,fullSize);   
    magma_smalloc_pinned (&hwork ,lwork);
    lapackf77_slacpy(MagmaFullStr, &partialSize, &partialSize, m.data, &partialSize, r, &partialSize);
    // std::cout << *(r + m.COLUMNS) << std::endl;
    if (leftEigenvector && rightEigenvector)
        magma_sgeev(MagmaVec, MagmaVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    else if (!(leftEigenvector) && rightEigenvector)
        magma_sgeev(MagmaNoVec, MagmaVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    else if (leftEigenvector && !(rightEigenvector))
        magma_sgeev(MagmaVec, MagmaNoVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    else 
        magma_sgeev(MagmaNoVec, MagmaNoVec, partialSize, r, partialSize, vr1, vi1, el, partialSize, er, partialSize, hwork, lwork, &info);
    free(vr1);   
    free(vi1);
    magma_free_pinned(hwork);
    magma_free_pinned(er);
    magma_free_pinned(el);
    magma_free_pinned(r);

    magma_queue_destroy(queue);                    
    magma_finalize();
    return std::tuple<float*, float*, float*, float*>(vr1, vi1, el, er); 

}
template <typename T>
std::tuple<float*, float*> eigenValuesNormalPrecisionSparseMatrix(Matrix<T> m, bool needEigenvectors = false) {
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
    // magma_ssetmatrix(m.ROWS, m.COLUMNS, m.data, m.ROWS, da, m.COLUMNS, queue)
    float aux_work[1]; // defining workspace for computation
    magma_int_t aux_iwork[1];
    if (!(needEigenvectors))
        magma_ssyevd_gpu(MagmaNoVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);
    else 
        magma_ssyevd_gpu(MagmaVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);

    lwork = (magma_int_t) aux_work[0];
    liwork = aux_iwork[0];
    magma_int_t* iwork = new magma_int_t[liwork];
    magma_smalloc_cpu(&h_work, lwork);

    magma_ssetmatrix(partialSize, partialSize, reinterpret_cast<float*>(m.data), partialSize, dr, partialSize, queue);
    if (!(needEigenvectors))
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

    return std::tuple<float*, float*>(v1, r);

}

template <typename T>
std::tuple<double*,double*> eigenValuesHighPrecisionSparseMatrix(Matrix<T> m, bool needEigenvectors=false) {
    assertm(isDoublePointer(m.data), "Types compatible, expecting a float for the data of matrix");
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
    double *v1; // vectors of eigenvalues only real part (not complex)
    magma_queue_create(dev, &queue);
    // blockSize = m.ROWS * magma_get_dgetri_nb(m.COLUMNS);

    err = magma_dmalloc_cpu(&r, fullSize);
    // err = magma_dmalloc_cpu(&toReturn, fullSize);
    err = magma_dmalloc_cpu(&v1, partialSize);
    err = magma_dmalloc(&dr, fullSize);

    double aux_work[1]; // defining workspace for computation
    magma_int_t aux_iwork[1];
    if (!(needEigenvectors))
        magma_dsyevd_gpu(MagmaNoVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);
    else 
        magma_dsyevd_gpu(MagmaVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, aux_work, -1, aux_iwork, -1, &info);

    lwork = (magma_int_t) aux_work[0];
    liwork = aux_iwork[0];
    magma_int_t* iwork = new magma_int_t[liwork];
    magma_dmalloc_cpu(&h_work, lwork);

    magma_dsetmatrix(partialSize, partialSize, reinterpret_cast<float*>(m.data), partialSize, dr, partialSize, queue);
    if (!(needEigenvectors))
        magma_dsyevd_gpu(MagmaNoVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, h_work, lwork, iwork, liwork, &info);
    else 
        magma_dsyevd_gpu(MagmaVec, MagmaLower, partialSize, dr, partialSize, v1, r, partialSize, h_work, lwork, iwork, liwork, &info);

    free(r);
    free(v1);
    free(iwork);
    free(h_work);
    magma_free(dr);
    magma_queue_destroy(queue);                     //  destroy  queuemagma_finalize (); 
    magma_finalize();

    return std::tuple<double*, double*>(v1, r);

}


template <typename T>
std::tuple<double*, double*, double*> singularValueDecompositionHighPrecision(Matrix<T> m) {
    /*
    Singular value decomposition  (SVD) A=u * σ * transpose(v).
    dim(A)=[m,n] - dim(u)=[m,m] - dim(σ)=[m,n] - dim(T)=[n,n]
    Function description : u is orthogonal matrix, σ is m,n matrix dimensional full of zeo except for min(m,n) diagonal a,d 
    finally v
    */
    magma_init();
    double *r; // where matrix.data will be
    double *u, *vtrans;
    double *v1; //vectors of values
    magma_int_t info, lwork;
    double work[1];
    double *hwork;

    magma_dmalloc_pinned(&r, m.ROWS*m.COLUMNS);
    magma_dmalloc_cpu(&u, m.ROWS*m.ROWS);
    magma_dmalloc_cpu(&vtrans, m.COLUMNS*m.COLUMNS);
    magma_dmalloc_cpu(&v1, min(m.ROWS, m.COLUMNS));

    magma_int_t blockSize =  magma_get_dgesvd_nb(m.ROWS, m.COLUMNS);
    lwork = carre(min(m.ROWS, m.COLUMNS)) + 2*min(m.ROWS, m.COLUMNS)*(1 + blockSize);
    magma_dmalloc_pinned(&hwork, lwork);
    lapackf77_dlacpy(MagmaFullStr, &(m.ROWS), &(m.COLUMNS), m.data, &(m.ROWS), r, &(m.COLUMNS));
    magma_dgesvd(MagmaNoVec, MagmaNoVec, m.ROWS, m.COLUMNS, r, m.ROWS, v1, u, m.ROWS, vtrans, m.COLUMNS, hwork, lwork, &info);

    free(u);
    free(vtrans);
    free(v1);
    magma_free_pinned(r);
    magma_free_pinned(hwork);

    return std::tuple<double*, double*, double*>(u, vtrans, v1);

    }

template <typename T>
std::tuple<float*, float*, float*> singularValueDecompositionNormalPrecision(Matrix<T> m) {
    /*
    Singular value decomposition  (SVD) A=u * σ * transpose(v).
    dim(A)=[m,n] - dim(u)=[m,m] - dim(σ)=[m,n] - dim(T)=[n,n]
    Function description : u is orthogonal matrix, σ is m,n matrix dimensional full of zeo except for min(m,n) diagonal a,d 
    finally v
    */
    magma_init();
    float *r; // where matrix.data will be
    float *u, *vtrans;
    float *v1; //vectors of values
    magma_int_t info, lwork;
    float work[1];
    float *hwork;

    magma_smalloc_pinned(&r, m.ROWS*m.COLUMNS);
    magma_smalloc_cpu(&u, m.ROWS*m.ROWS);
    magma_smalloc_cpu(&vtrans, m.COLUMNS*m.COLUMNS);
    magma_smalloc_cpu(&v1, min(m.ROWS, m.COLUMNS));

    magma_int_t blockSize =  magma_get_sgesvd_nb(m.ROWS, m.COLUMNS);
    lwork = carre(min(m.ROWS, m.COLUMNS)) + 2*min(m.ROWS, m.COLUMNS)*(1 + blockSize);
    magma_smalloc_pinned(&hwork, lwork);
    lapackf77_slacpy(MagmaFullStr, &(m.ROWS), &(m.COLUMNS), m.data, &(m.ROWS), r, &(m.COLUMNS));
    magma_sgesvd(MagmaNoVec, MagmaNoVec, m.ROWS, m.COLUMNS, r, m.ROWS, v1, u, m.ROWS, vtrans, m.COLUMNS, hwork, lwork, &info);

    free(u);
    free(vtrans);
    free(v1);
    magma_free_pinned(r);
    magma_free_pinned(hwork);

    return std::tuple<float*, float*, float*>(u, vtrans, v1);

    }
