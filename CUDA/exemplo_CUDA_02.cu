#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MSIZE 8192
#define BSIZE 8

__host__ int dgemmCUDA(double alpha, double* A, double* B, double beta, double* C);
__global__ void k_dgemm(double alpha, double* A, double* B, double beta, double* C, int TILES);
void printMatrix(double *A, int n);

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

    printMatrix(C, MSIZE);

    // Calcular o GEMM
    dgemmCUDA(alpha, A, B, beta, C);

    // I/O para guardar os resultados
    printf("________________________________________\n");
    printMatrix(C, MSIZE);

    // Libera memória alocada
    free(A);
    free(B);
    free(C);
    
}


__host__ int dgemmCUDA(double alpha, double* A, double* B, double beta, double* C)
{
    int tSize = MSIZE * MSIZE * sizeof(double);
    double *Ad, *Bd, *Cd;

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

    int TILES = MSIZE/BSIZE;

    dim3 dimGrade(TILES, TILES);
    dim3 dimBloco(BSIZE, BSIZE);

    // Chamada ao kernel que implementa a GEMM
    k_dgemm<<<dimGrade, dimBloco>>>(alpha, Ad, Bd, beta, Cd, TILES);

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

__global__ void k_dgemm(double alpha, double* A, double* B, double beta, double* C, int TILES)
{
    __shared__ double Ads[BSIZE][BSIZE];
    __shared__ double Bds[BSIZE][BSIZE];

    int tj = threadIdx.x;
    int ti = threadIdx.y;
    int bj = blockIdx.x;
    int bi = blockIdx.y;

    int i = bi*BSIZE + ti;
    int j = bj*BSIZE + tj;
    int m, n, k;

    double cValue = 0.0;

    for (m=0, n = 0; m < TILES; m++, n += BSIZE){
        Ads[ti][tj] = A [i*MSIZE + n + tj];
        Bds[ti][tj] = B [(n+ti)*MSIZE + j];
        __syncthreads();
        for (k = 0; k < BSIZE; k++)
            cValue += Ads[ti][k] * Bds[k][tj];
        __syncthreads();
    }

    C[i*MSIZE + j] = alpha * cValue + beta * C[i*MSIZE + j];
}


void printMatrix(double *A, int n)
{
    int i, j;
    double *ptr;
    size_t lda;

    if (A == NULL)
    {
        printf("Matriz nula\n");
        return;
    }
    ptr = A;
    lda = n;
    if(n <= 8)
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
                printf("%.1lf ", ptr[i * lda + j]);
    
            printf("\n");
        }
    }else{
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < 4; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf(" ... ");
            for (j = n - 4; j < n; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf("\n");
        }
        printf(" ... \n");
        for(i = n - 4; i < n; i++)
        {
            for (j = 0; j < 4; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf(" ... ");
            for (j = n - 4; j < n; j++)
                printf("%.1lf ", ptr[i * lda + j]);
            printf("\n");
        }
    }
}