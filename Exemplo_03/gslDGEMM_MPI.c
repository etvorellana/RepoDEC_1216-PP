#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
// #include <gsl/gsl_rstat.h>
#include <omp.h>
#include <mpi.h>

#define MSIZE 4096

void printMatrix(gsl_matrix *A);

int main(int argc, char **argv)
{

  int i, j, k;
  int meuId, quantProc;

  int matSize, tlSize, tgSize, trSize;
  gsl_matrix *A, *B, *C;
  gsl_matrix *Al, *Cl;
  double start, stop;
  double t1, t2, t3;
  MPI_Status status;

  start = omp_get_wtime();
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &quantProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &meuId);

  matSize = MSIZE;
  tgSize = matSize * matSize;   // quantidade de elementos da matriz
  tlSize = matSize / quantProc; // quantidade de linhas da matriz local
  trSize = tlSize * matSize;    // quantidade de elementos da matrriz local

  // Entendendo matrizes no gsl
  // Alocando matrizes quadradas
  if (meuId == 0)
  {
    A = gsl_matrix_alloc(matSize, matSize);
    C = gsl_matrix_alloc(matSize, matSize);
  }

  B = gsl_matrix_alloc(matSize, matSize);

  if (meuId == 0)
  {
    for (i = 0; i < MSIZE; i++)
    {
        for (j = 0; j < MSIZE; j++)
        {
            // os elementos de A guardam valores -1
            gsl_matrix_set(A, i, j, -1.0);
            // A matriz B é a identidade
            gsl_matrix_set(B, i, j, 0.0);
            // os elementos de C guardam valores 1
            gsl_matrix_set(C, i, j, 1.0);
        }
        gsl_matrix_set(B, i, i, 1.0);
    }
  }

  Al = gsl_matrix_alloc(tlSize, matSize);
  Cl = gsl_matrix_alloc(tlSize, matSize);

  MPI_Bcast(B->data, tgSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double *ptrOrg, *ptrDst;
  if (meuId == 0)
  {
    ptrOrg = A->data;
  }
  ptrDst = Al->data;

  MPI_Scatter(ptrOrg, trSize, MPI_DOUBLE, ptrDst, trSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (meuId == 0)
  {
    ptrOrg = C->data;
  }
  ptrDst = Cl->data;

  MPI_Scatter(ptrOrg, trSize, MPI_DOUBLE, ptrDst, trSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double alpha = 2.0, beta = 2.0;
  stop = omp_get_wtime();
  t1 = stop - start;
  if (meuId == 0)
  {
    printf("Prinit inicial das matrizes\n");
    printf("Matriz A\n");
    printMatrix(A);
    printf("Matriz B\n");
    printMatrix(B);
    printf("Matriz C\n");
    printMatrix(C);
  }
  start = omp_get_wtime();
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Al, B, 1.0, Cl);

  if (meuId == 0)
  {
    ptrDst = C->data;
  }
  ptrOrg = Cl->data;

  MPI_Gather(ptrOrg, trSize, MPI_DOUBLE, ptrDst, trSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  stop = omp_get_wtime();
  t2 = stop - start;

  start = omp_get_wtime();
  if (meuId == 0)
  {
    printf("Resultado\n");
    printf("Matriz C\n");
    printMatrix(C);
    stop = omp_get_wtime();
    t3 = stop - start;
    printf("Tempo de inicializacao: %lf\n", t1);
    printf("Tempo de multiplicacao: %lf\n", t2);
    printf("Tempo de print: %lf\n", t3);
    printf("Tempo total: %lf\n", t1 + t2 + t3);
    gsl_matrix_free(A);
    gsl_matrix_free(C);
  }

  gsl_matrix_free(B);
  gsl_matrix_free(Al);
  gsl_matrix_free(Cl);

  MPI_Finalize();
  return 0;
}

void printMatrix(gsl_matrix *A)
{
  int i, j;
  double *ptr;
  size_t lda;

  if (A == NULL)
  {
    printf("Matriz nula\n");
    return;
  }
  ptr = A->data;
  lda = A->tda;
  if (A->size1 <= 8 && A->size2 <= 8)
  {
    for (i = 0; i < A->size1; i++)
    {
      for (j = 0; j < A->size2; j++)
        printf("%.1lf ", ptr[i * lda + j]);

      printf("\n");
    }
  }
  else
  {
    for (i = 0; i < 4; i++)
    {
      for (j = 0; j < 4; j++)
        printf("%.1lf ", ptr[i * lda + j]);
      printf(" ... ");
      for (j = A->size2 - 4; j < A->size2; j++)
        printf("%.1lf ", ptr[i * lda + j]);
      printf("\n");
    }
    printf(" ... \n");
    for (i = A->size1 - 4; i < A->size1; i++)
    {
      for (j = 0; j < 4; j++)
        printf("%.1lf ", ptr[i * lda + j]);
      printf(" ... ");
      for (j = A->size2 - 4; j < A->size2; j++)
        printf("%.1lf ", ptr[i * lda + j]);
      printf("\n");
    }
  }
}
