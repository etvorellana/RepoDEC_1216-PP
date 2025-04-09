#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void)
{
    int n, nth;

    #ifdef _OPENMP //Compilação condicional
        n = omp_get_max_threads();      //função OpenMP
        printf("Número de threads disponíveis: %d\n", n);
        n = omp_get_num_threads();      // Função OpenMP
        nth = omp_get_thread_num(); // Função OpenMP
    #else
        n = 1;
        nth = 0;
    #endif


    printf("Número de threads em execução: %d\n", n);
    printf("Executando na thread: %d\n", nth);
    return 0;

}

// compile with: gcc -fopenmp par_omp_01.c -o main_01
// run with: ./main_01
