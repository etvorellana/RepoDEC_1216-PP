#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void)
{
    double X[8], Y[8], Z[8];

    for(int i = 0; i < 8; i++)
    {
        X[i] = i;
        Y[i] = i * 2;
    }

    double start, stop;
    start = omp_get_wtime();
    for(int i = 0; i < 8; i++)
    {
        Z[i] = X[i] + Y[i];
    }
    stop = omp_get_wtime();
    printf("Tempo sequencial: %f\n", stop - start);
    
    start = omp_get_wtime();
    #pragma omp parallel num_threads(8)
    {
        int i = omp_get_thread_num();
        Z[i] = X[i] + Y[i];   
    }
    stop = omp_get_wtime();
    printf("Tempo paralelo: %f\n", stop - start);
    return 0;
}