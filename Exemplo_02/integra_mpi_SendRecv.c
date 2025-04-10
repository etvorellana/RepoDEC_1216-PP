
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>


double integraTrap(double a, double b, int n);
double f(double x);
double int_f(double x);

int main(int argc, char * argv[])
{

    double integral = 0.0, a, b;
    double start, stop;
    int n;
    /*
    MPI_Init(
        int* argc,
        char*** argv)
    */
    MPI_Init(&argc, &argv);
    
    /*
    MPI_Comm_size(
        MPI_Comm communicator,
        int* size)
    */
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
    MPI_Comm_rank(
        MPI_Comm communicator,
        int* rank)
    */
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*
    MPI_Send(
        void* data,
        int count,
        MPI_Datatype datatype,
        int destination,
        int tag,
        MPI_Comm communicator)
    */

    if(rank == 0)
    {
        FILE *fp;
        // Leitura dos parâmetros de entrada
        fp = fopen("parametros.csv", "r");
        if (fp == NULL)
        {
            printf("Erro ao abrir o arquivo de parâmetros.\n");
            return 1;
        }
        // Lê os parâmetros a, b e n do arquivo
        if (fscanf(fp, "%lf,%lf,%d", &a, &b, &n) != 3)
        {
            printf("Erro ao ler os parâmetros do arquivo.\n");
            fclose(fp);
            return 1;
        }
        fclose(fp);
        start = omp_get_wtime();
        // Enviando os parâmetros para os outros ranks
        for(int dest = 1; dest < size; dest++)
        {
            MPI_Send(&a, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&b, 1, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&n, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
        }
    }
    /*
    MPI_Recv(
        void* data,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm communicator,
        MPI_Status* status)
    */
    else
    {
        MPI_Status status;
        MPI_Recv(&a, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&n, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    }
    
    // calculando o parametros para o processamento
    double local_a, local_b, h;
    h = (b - a)/size;
    local_a = a + (rank * h);
    local_b = local_a + h;
    
    // calcular a integral
    integral = integraTrap(local_a, local_b, n/size);
    if(rank == 0)
    {
        double integral_par;
        MPI_Status status;
        for(int orig = 1; orig < size; orig++)
        {
            MPI_Recv(&integral_par, 1, MPI_DOUBLE, orig, 0, MPI_COMM_WORLD, &status);
            integral += integral_par;
        }
        // saída
        double stop = omp_get_wtime();
        printf("Com n=%d trapezios, a estimativa da\n", n);
        printf("integral de %.2f a %.2f e: %.6f\n", a, b, integral);
        double integralR = int_f(b) - int_f(a);
        double eAbs = fabs(integral - integralR);
        double eRel = eAbs/integralR;
        printf("O valor exato é de %.12f, Erel %.12f, Eabs %.12f\n", integralR, eRel, eAbs);
        printf("Tempo: %.12f\n", stop - start);
    }
    else
    {
        MPI_Send(&integral, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize(); 
    return 0;
}

double integraTrap(double a, double b, int n)
{   
    double x = a, integral = 0;
    double h = (b - a) / n;
    integral = (f(a) + f(b)) * 0.5;

    for (int i = 1; i < n; i++)
    {
        x += h;
        integral += f(x);
    }
    integral *= h;
    return integral;
}

//5x^4 + 4x^3 + 3x^2 + 2x + 1
double f(double x)
{
    double fun = 0.0;
    for(int i = 0; i < 5; i++)
    {
        fun += (i+1.0)*pow(x, i);
    }
    //return 5 * x * x * x * x + 4 * x * x * x + 3 * x * x + 2 * x + 1;
    return fun;
}

//x^5 + x^4 + x^3 + x^2 + x
double int_f(double x)
{
    double fun = 0.0;
    for(int i = 1; i < 6; i++)
    {
        fun += pow(x, i);
    }   
    //return x * x * x * x * x + x * x * x * x + x * x * x + x * x + x;
    return fun;
}