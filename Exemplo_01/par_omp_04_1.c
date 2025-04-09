#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define TRUE 1
#define FALSE 0
#define NTHREADS 4
int main(void)
{
    int n, nth;
    
    #pragma omp parallel if(TRUE) num_threads(NTHREADS) //Diretiva OpenMP
    {
        int soma = 0;
        int nth;
        for(int i = 0; i < 16; i++)
            soma+= i;
        #ifdef _OPENMP //Compilação condicional
            nth = omp_get_thread_num(); // Função OpenMP
        #else
            nth = 0;
        #endif 
        printf("Resultado na Thread %d: %d\n", nth, soma);
    }
    return 0;

}

// compile with: gcc -fopenmp par_omp_02.c -o main_02
// run with: ./main_02
// compile with: gcc par_omp_02.c -o main_02_
// run with: ./main_02_