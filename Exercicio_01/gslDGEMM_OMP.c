#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
// #include <gsl/gsl_rstat.h>
#include <omp.h>

#define MSIZE 4096

void printMatrix(gsl_matrix *A);

int main(int argc, char **argv)
{

    int i, j, k;

    gsl_matrix *A, *B, *C;
    double start, stop;
    double t1, t2, t3;
    start = omp_get_wtime();
    // Alocando matrizes quadradas
    A = gsl_matrix_alloc(MSIZE, MSIZE);
    B = gsl_matrix_alloc(MSIZE, MSIZE);
    C = gsl_matrix_alloc(MSIZE, MSIZE);
    // inicializando as matrizes
    for (i = 0; i < MSIZE; i++)
    {
        for (j = 0; j < MSIZE; j++)
        {
            // os elementos de A guardam valores -1
            gsl_matrix_set(A, i, j, -1.0);
            // A matriz B é a identidade
            gsl_matrix_set(B, i, j, 0.0);
            // os elementos de C guardam valores 1
            gsl_matrix_set(C, i, j, 1.0);
        }
        gsl_matrix_set(B, i, i, 1.0);
    }
    double alpha = 2.0, beta = 2.0;
    stop = omp_get_wtime();
    t1 = stop - start;

    printf("Prinit inicial das matrizes\n");
    printf("Matriz A\n");
    printMatrix(A);
    printf("Matriz B\n");
    printMatrix(B);
    printf("Matriz C\n");
    printMatrix(C);
    start = omp_get_wtime();
    #pragma omp parallel num_threads(2)
    {
        int meuId, quantProc;
        meuId = omp_get_thread_num();
        quantProc = omp_get_num_threads();
        gsl_matrix_view Alv, Clv;
        int nrows = A->size1/quantProc;
        int ncols = A->size2;
        int start_row = meuId * nrows;
        Alv = gsl_matrix_submatrix(A, start_row, 0, nrows, ncols);
        gsl_matrix *Al = &Alv.matrix;        
        ncols = C->size2;
        Clv = gsl_matrix_submatrix(C, start_row, 0, nrows, ncols);
        gsl_matrix *Cl = &Clv.matrix;
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, alpha, Al, B, beta, Cl);
    }
    stop = omp_get_wtime();
    t2 = stop - start;

    start = omp_get_wtime();
    printf("Resultado\n");
    printf("Matriz C\n");
    printMatrix(C);
    stop = omp_get_wtime();
    t3 = stop - start;

    printf("Tempo de inicializacao: %lf\n", t1);
    printf("Tempo de multiplicacao: %lf\n", t2);
    printf("Tempo de print: %lf\n", t3);
    printf("Tempo total: %lf\n", t1 + t2 + t3);
    
    gsl_matrix_free(A);
    gsl_matrix_free(C);
    gsl_matrix_free(B);

    return 0;
}

void printMatrix(gsl_matrix *A)
{
    int i, j;
    double *ptr;
    size_t lda;

    if (A == NULL)
    {
        printf("Matriz nula\n");
        return;
    }
    ptr = A->data;
    lda = A->tda;
    if(A->size1 <= 8 && A->size2 <= 8)
    {
        for (i = 0; i < A->size1; i++)
        {
            for (j = 0; j < A->size2; j++)
                printf("%.1lf ", ptr[i * lda + j]);
    
            printf("\n");
        }
    }else{
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < 4; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf(" ... ");
            for (j = A->size2 - 4; j < A->size2; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf("\n");
        }
        printf(" ... \n");
        for(i = A->size1 - 4; i < A->size1; i++)
        {
            for (j = 0; j < 4; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf(" ... ");
            for (j = A->size2 - 4; j < A->size2; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf("\n");
        }
    }
}
