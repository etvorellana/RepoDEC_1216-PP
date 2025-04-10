#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char * argv[]){

    int rank, size, orig; char msg[100]; MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank != 0){
        sprintf(msg, "Oi do processo %d!", rank);
        MPI_Send(msg, strlen(msg)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        printf("Oi do processo %d!\n", rank);
        for(orig=1; orig<size; orig++){
            MPI_Recv(msg, 100, MPI_CHAR, orig, 0, MPI_COMM_WORLD, &status);
            printf("%s\n", msg);
        }
    }

    MPI_Finalize();
return 0;
}