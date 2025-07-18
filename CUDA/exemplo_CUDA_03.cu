/*
 * exampleCUDA-001.cu
 *
 * Copyright 2017 Esbel Tomas Valero Orellana <evalero@ninjapad>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 *
 *
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cublas_v2.h>

#define SIZE 8192
#define BSIZE 16

__host__ void dgemmCUDA( const char TA, const char TB,
			const int M, const int N, const int K,
			const double a, double *A, const int lda,
			double *B, const int ldb, const double b,
			double *C, const int ldc
			);

int main(int argc, char **argv)
{
	int i,j,k;

	int matSize = 32;
	double start, stop, dt, Dt;
	double gflop;

	int M , K, N;
	char ta, tb;
	const double alpha = 2.0; 
    const double beta = 2.0;
	int lda, ldb, ldc;

	FILE *desemp;

	if(argc > 1)
		desemp = fopen(argv[1], "w");
	else
		desemp = fopen("./desempenho.dat", "w");

	ta = tb = 'N';
	
	while (matSize <= SIZE){

		double *A = (double*) malloc( matSize * matSize * sizeof(double) );
		double *B = (double*) malloc( matSize * matSize * sizeof(double) );
		double *C = (double*) malloc( matSize * matSize * sizeof(double) );
		Dt = 0.0;

		M = K = N = matSize;
		lda = ldb = ldc = matSize;
		for(k=0; k<3; k++){
			for (i = 0; i < matSize; i++) {
				for (j = 0; j < matSize; j++) {
					//A[i*matSize+j] = (double)(rand()%3 - 1);
					A[i*matSize+j] = -1.0;
					//B[i*matSize+j] = (double)(rand()%9 - 4);
					B[i*matSize+j] = 0.0;
					//C[i*matSize+j] = ((double)rand()) / RAND_MAX;
					C[i*matSize+j] = 1.0;
				}
                B[i*matSize+i] = 1.0; // B é a matriz identidade
			}

			start = omp_get_wtime();
			dgemmCUDA(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
			stop = omp_get_wtime();

			dt = stop - start;
			if(!k)
				Dt = dt;
			else if(Dt > dt)
				Dt = dt;
		}

		gflop = 2.0*matSize*matSize*matSize*0.000000001;
		printf("%f \n", C[0*ldc+0]);
		printf("%f \n", C[(matSize-1)*ldc+matSize-1]);
		printf("Result %d Ok!\n time: best - %f \n Size in GFLOP: %f\n Perfrmance: %f GFLOPS\n", matSize, Dt, gflop, gflop/Dt);
		fprintf(desemp, "%d\t%.12lf\t%.12lf\t%.12lf\n", matSize, Dt, gflop, gflop/Dt);
		fflush(stdout);
		fflush(desemp);
		free(A);
		free(B);
		free(C);
		matSize += 32;
	}
	fclose(desemp);
	return 0;
}



void dgemmCUDA( const char TA, const char TB,
			const int M, const int N, const int K,
			const double a, double *A, const int lda,
			double *B, const int ldb, const double b,
			double *C, const int ldc
			)
{
	int tSize = M * N * sizeof(double);
	double *Ad, *Bd, *Cd;

	// Alocar memória para as matrizes no device
	if ( cudaSuccess != cudaMalloc( (void**)&Ad, tSize ))
		printf( "Erro alocando A!\n" );
	if ( cudaSuccess != cudaMalloc( (void**)&Bd, tSize ))
		printf( "Erro alocando B!\n" );
	if ( cudaSuccess != cudaMalloc( (void**)&Cd, tSize ))
		printf( "Erro alocando C!\n" );
	// Copiar as matrizes para os respectivos espaços de memória no device
	cudaError_t cudaErro;
	cudaErro = cudaMemcpy(Ad, A, tSize, cudaMemcpyHostToDevice);
	if ( cudaSuccess != cudaErro)
		printf( "Erro copiando A!\n" );
	cudaErro = cudaMemcpy(Bd, B, tSize, cudaMemcpyHostToDevice);
	if ( cudaSuccess != cudaErro)
		printf( "Erro copiando B!\n" );
	cudaErro = cudaMemcpy(Cd, C, tSize, cudaMemcpyHostToDevice);
	if ( cudaSuccess != cudaErro)
		printf( "Erro copiando C!\n" );

	cublasHandle_t handle;
    cublasStatus_t cublasErro;
	cublasCreate(&handle);
    /*
    cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
    */
	cublasErro = cublasDgemm(  handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                M, N, K, &a, Ad, lda, Bd, ldb, &b, Cd, ldc);
    if ( CUBLAS_STATUS_SUCCESS != cublasErro)
        printf( "Erro no cublasDgemm!\n" );

	cublasDestroy(handle);


	// Copiar a matriz resultante do divece para o host
	cudaErro = cudaMemcpy(C, Cd, tSize, cudaMemcpyDeviceToHost);
	if ( cudaSuccess != cudaErro)
		printf( "Erro recuperando C!\n" );
	// Liberar a memória no device
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);

	return;
}
