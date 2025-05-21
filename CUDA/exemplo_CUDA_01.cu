#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MSIZE 64

__host__ int dgemmCUDA(double alpha, double* A, double* B, double beta, double* C);
__global__ void k_dgemm(double alpha, double* A, double* B, double beta, double* C)

int main(int argc, char **argv)
{
    //Alocar as matrizes A, B e C
    double *A = (double* ) malloc(MSIZE*MSIZE*sizeof(double));
    double *B = (double* ) malloc(MSIZE*MSIZE*sizeof(double));
    double *C = (double* ) malloc(MSIZE*MSIZE*sizeof(double));
    // I/O para inicializar as matrizes
    for (int i = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++)
        {
            // os elementos de A guardam valores -1
            A[i*MSIZE + j] = -1.0;
            // A matriz B é a identidade
            B[i*MSIZE + j] = 0.0;
            // os elementos de C guardam valores 1
            C[i*MSIZE + j] = 1.0;
        }
        B[i*MSIZE + i] = 1.0;
    }
    double alpha = 2.0, beta = 2.0;

    for (int i = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++)
            printf("%.1lf ", C[i * MSIZE + j]);
    
        printf("\n");
    }

    // Calcular o GEMM
    dgemmCUDA(alpha, A, B, beta, C);

    // I/O para guardar os resultados
    for (int i = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++)
            printf("%.1lf ", C[i * MSIZE + j]);
    
        printf("\n");
    }
    // Libera memória alocada
    free(A);
    free(B);
    free(C);
    
}


__host__ int dgemmCUDA(double alpha, double* A, double* B, double beta, double* C)
{
    int tSize = MSIZE * MSIZE * sizeof(double))
    double* Ad, Bd, Cd;

    // Alocar memória para as matrizes no diveice
    if ( cudaSuccess != cudaMalloc((void**)&Ad, tSize))
    {
        printf("Erro alocando A!!\n");
        return 1;
    }
    if ( cudaSuccess != cudaMalloc((void**)&Bd, tSize))
    {
        printf("Erro alocando B!!\n");
        return 1;
    }
    if ( cudaSuccess != cudaMalloc((void**)&Cd, tSize))
    {
        printf("Erro alocando C!!\n");
        return 1;
    }

    // Copiar as matrizes do host para os respectivos espços de memória no divice
    cudaError_t cudaErro;
    cudaErro = cudaMemcpy(Ad, A, tSize, cudaMemcpyHostToDevice);
    if(cudaErro != cudaSuccess)
    {
        printf("Erro copiando A para o device!!\n");
        return 2;
    }
    cudaErro = cudaMemcpy(Bd, B, tSize, cudaMemcpyHostToDevice);
    if(cudaErro != cudaSuccess)
    {
        printf("Erro copiando B para o device!!\n");
        return 2;
    }
    cudaErro = cudaMemcpy(Cd, C, tSize, cudaMemcpyHostToDevice);
    if(cudaErro != cudaSuccess)
    {
        printf("Erro copiando C para o device!!\n");
        return 2;
    }

    // Definir a grade que se deseja utilizar
    dim3 dimGrade(MSIZE/32, MSIZE/32);
    dim3 dimBloco(32, 32);

    // Chamada ao kernel que implementa a GEMM
    k_dgemm<<<dimGrade, dimBloco>>>(alpha, A, B, beta, C);

    // copiar a matriz resultante do device para o espaço de meória do host
    cudaErro = cudaMemcpy(C, Cd, tSize, cudaMemcpyDeviceToHost);
    if(cudaErro != cudaSuccess)
    {
        printf("Erro copiando C para o host!!\n");
        return 3;
    }

    //Liberar a memória no divice
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    return 0;
}

__global__ void k_dgemm(double alpha, double* A, double* B, double beta, double* C)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    double cValue = 0.0;
    for(int k = 0; k < MSIZE; k++)
        cValue += A[i*MSIZE + k] * B[k * MSIZE + j];
    C[i*MSIZE + j] = alpha * cValue + beta * C[i*MSIZE + j];
}