#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define TRUE 1
#define FALSE 0

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

    int execParallel = TRUE;
    //execParallel = FALSE;
    int wThreads = 4; // 2;
    #ifdef _OPENMP //Compilação condicional
        omp_set_num_threads(wThreads); // Definindo o número de threads
    #endif
    //#pragma omp parallel if(execParallel) num_threads(2*wThreads)//Diretiva OpenMP
    #pragma omp parallel if(execParallel) //Diretiva OpenMP
    {
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
    }
    
    return 0;

}

// compile with: gcc -fopenmp par_omp_02.c -o main_02
// run with: ./main_02
// compile with: gcc par_omp_02.c -o main_02_
// run with: ./main_02_