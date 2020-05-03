#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"

#define EXIT_SUCESS 0

typedef struct {
    size_t ROWS;
    size_t COLUMNS;
    double* data;
} Matrix;

void inverseMatrix(Matrix m) {
    //It uses the LU decomposition with partial pivoting and row interchanges

    magma_init();
    double gpu_time, *dwork, *da;
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

    // err = magma_dmalloc_cpu(&a, fullSize);
    err = magma_dmalloc(&da, fullSize);
    err = magma_dmalloc(&dwork, blockSize);
  
    magma_dprint(m.ROWS, m.COLUMNS, m.data, 8);

    magma_dsetmatrix(m.ROWS, m.COLUMNS, m.data, m.ROWS, da, m.COLUMNS, queue); // put matrix data in da for device (GPU) -- da*X = Iusing LU factorization
    // magmablas_dlacpy(MagmaFull, m.ROWS, m.COLUMNS, da, m.ROWS, dr, m.COLUMNS, queue); // put da in dr (copy of data) to test A*A⁽-1)
    gpu_time = magma_sync_wtime(NULL); // debug is 

    magma_dgetrf_gpu(m.ROWS, m.COLUMNS, da, m.ROWS, pivot, &info);
    magma_dgetri_gpu(m.ROWS, da, m.COLUMNS, pivot, dwork, blockSize, &info); // inverse is in da
    // magma_dgemm(MagmaNoTrans, MagmaNoTrans, m.COLUMNS, m.ROWS, m.COLUMNS, 1, da, m.ROWS, dr, m.COLUMNS, 0, dtest, m.ROWS, queue);
    // gpu_time = magma_sync_wtime(NULL) - gpu_time;
    // std::cout << "Time taken for the operation : " << gpu_time << " sec" << std::endl; 

    magma_dgetmatrix(m.ROWS, m.COLUMNS, da, m.ROWS, m.data, m.COLUMNS, queue);
    magma_dprint(m.ROWS, m.COLUMNS, m.data, 8);
    // free(a);
    free(pivot);
    magma_free(da);
    magma_queue_destroy(queue);                     //  destroy  queuemagma_finalize ();    
    magma_finalize();
}

int main(void){
    std::cout << "Working..." << std::endl;
    Matrix matrix = Matrix{8, 8, new double[8*8]};
      
    for (unsigned int i = 0; i < matrix.ROWS*matrix.COLUMNS; i++)
        *(matrix.data + i) = rand()%100;
    inverseMatrix(matrix);
    return EXIT_SUCESS;
}