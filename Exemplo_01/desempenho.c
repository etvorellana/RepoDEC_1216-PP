#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void)
{
    double start, stop;
    double ooper = 1.0;
    int cont = 0;
    start = omp_get_wtime();
    do{
        ooper = ooper * 1.00001;
        cont++;
        stop = omp_get_wtime();
    }while(stop - start < 1.0);
    printf("Número de operações: %d\n", cont);
    return 0;
}