#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MSIZE 8

int main()
{
    double *A = (double* ) malloc(MSIZE*MSIZE*sizeof(double));
    double *B = (double* ) malloc(MSIZE*MSIZE*sizeof(double));
    double *C = (double* ) malloc(MSIZE*MSIZE*sizeof(double));

    for (int i = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++)
        {
            // os elementos de A guardam valores -1
            A[i*MSIZE + j] = -1.0;
            // A matriz B é a identidade
            B[i*MSIZE + j] = 0.0;
            // os elementos de C guardam valores 1
            C[i*MSIZE + j] = 1.0;
        }
        B[i*MSIZE + i] = 1.0;
    }

    double alpha = 2.0, beta = 2.0;

    for (int i = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++)
            printf("%.1lf ", C[i * MSIZE + j]);
    
        printf("\n");
    }

    #pragma omp parallel num_threads(MSIZE*MSIZE)
    {
        int i, j, k;
        register double cvalue = 0.0;
        int minhaThr = omp_get_thread_num();
        i = minhaThr / MSIZE;
        j = minhaThr % MSIZE;
        for(k = 0; k < MSIZE; k++)
            cvalue += A[i*MSIZE + k] * B[k*MSIZE + j];
        C[i*MSIZE + j] = alpha * cvalue + beta * C[i*MSIZE + j];

    }

    for (int i = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++)
            printf("%.1lf ", C[i * MSIZE + j]);
    
        printf("\n");
    }

    free(A);
    free(B);
    free(C);


}