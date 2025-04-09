#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#define TRUE 1
#define FALSE 0
#define NTHREADS 4

int main(void)
{
    int size[] = {8, 16, 32, 16, 8, 32, 16, 32, 8, 32, 8, 16, 32, 16, 8, 16};
    
    printf("[ %d, ", size[0]);
    for(int i = 1; i < 15; i++)
        printf("%d, ", size[i]);
    printf("%d ]\n", size[15]);

    #pragma omp parallel if(TRUE) num_threads(NTHREADS) //Diretiva OpenMP
    {
        int soma = 0;
        int mySize[16];
        int cont = 0;
        int nth, thn;
        #ifdef _OPENMP //Compilação condicional
            nth = omp_get_thread_num(); // Função OpenMP
            thn = omp_get_num_threads(); // Função OpenMP
        #else
            nth = 0;
            thn = 1;
        #endif
        #pragma omp for
        for(int i = 0; i < 16; i++){
            int n = size[i];
            size[i] = nth;
            mySize[cont++] = n;
            for(int j = 0; j < n; j++){
                soma += j;
                sleep(1);
            }
        }
        #pragma omp critical
        {
            for(int i = 0; i < cont; i++)
                printf("Thread %d pegou o tamanho do vetor: %d\n", nth, mySize[i]);
            printf("Resultado na Thread %d: %d\n", nth, soma);
        }
    }

    printf("[ %d, ", size[0]);
    for(int i = 1; i < 15; i++)
        printf("%d, ", size[i]);
    printf("%d ]\n", size[15]);
    return 0;

}

// compile with: gcc -fopenmp par_omp_02.c -o main_02
// run with: ./main_02
// compile with: gcc par_omp_02.c -o main_02_
// run with: ./main_02_