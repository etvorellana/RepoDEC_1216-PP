#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define TRUE 1
#define FALSE 0
#define NTHREADS 4
int main(void)
{
    
    #pragma omp parallel if(TRUE) num_threads(NTHREADS) //Diretiva OpenMP
    {
        int soma = 0;
        int nth, thn;
        #ifdef _OPENMP //Compilação condicional
            nth = omp_get_thread_num(); // Função OpenMP
            thn = omp_get_num_threads(); // Função OpenMP
        #else
            nth = 0;
            thn = 1;
        #endif
        #pragma omp for
        for(int i = 0; i < 16; i++)
            soma+= i;
         
        printf("Resultado na Thread %d: %d\n", nth, soma);
    }
    return 0;

}

// compile with: gcc -fopenmp par_omp_02.c -o main_02
// run with: ./main_02
// compile with: gcc par_omp_02.c -o main_02_
// run with: ./main_02_