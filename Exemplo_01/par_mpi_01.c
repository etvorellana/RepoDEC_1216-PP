#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
    int n, rank, size;

    MPI_Init(&argc, &argv); // Inicializa o ambiente MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtém o número total de processos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtém o ID do processo atual

    n = size; // Número de processos disponíveis
    printf("Número de processos disponíveis: %d\n", n);
    printf("Executando no processo: %d\n", rank);

    MPI_Finalize(); // Finaliza o ambiente MPI
    return 0;
}