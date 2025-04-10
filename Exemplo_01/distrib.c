#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void)
{
    double X[32], Y[32], Z[32];

    for(int i = 0; i < 32; i++)
    {
        X[i] = i;
        Y[i] = i * 2;
    }

    double start, stop;
    start = omp_get_wtime();
    for(int i = 0; i < 32; i++)
    {
        Z[i] = X[i] + Y[i];
    }
    stop = omp_get_wtime();
    printf("Tempo sequencial: %f\n", stop - start);
    
    start = omp_get_wtime();
    #pragma omp parallel num_threads(32)
    {
        int i = omp_get_thread_num();
        Z[i] = X[i] + Y[i];   
    }
    stop = omp_get_wtime();
    printf("Tempo paralelo: %f\n", stop - start);
    return 0;
}