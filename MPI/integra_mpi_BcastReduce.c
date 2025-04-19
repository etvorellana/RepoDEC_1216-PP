
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>


double integraTrap(double a, double b, int n);
double f(double x);
double int_f(double x);

int main(int argc, char *argv[])
{

    double integral = 0.0, a = 0.0, b = 1.0;
    int n = 512;

    MPI_Init(NULL, NULL);
    
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // so para garantir vamos zerar o valor em todos os nós
    a = 1.0; 
    b = 0.0; 
    n = 0;
    if(rank == 0)
    {
        // a, b e n são parametros de entrada
        // entra de parametros, por exemplo de um arquivo
        // ...
        a = 0.0, b = 2048.0;
        n = 2048;
    }
    else
    {
        // Nos outros nós vamos colocar lixo nos parâmetros
        a = size;
        b = rank;
        n = rank;
    }

    /*
    MPI_Bcast(
        void* data,
        int count,
        MPI_Datatype datatype,
        int root,
        MPI_Comm communicator)
    */
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // calculando o parametros para o processamento
    double local_a, local_b, h;
    int local_n = n/size; // trapezios no intervalo de cada processo
    h = (b - a)/size;     // tamanho do intervalo 
    local_a = a + (rank * h); // início do intervaalo
    local_b = local_a + h; //final do intervalo
    
    // calcular a integral
    integral = integraTrap(local_a, local_b, n/size);
    double integral_tot;

    /*
    MPI_Reduce(
        void* send_data,
        void* recv_data,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        int root,
        MPI_Comm communicator)
    */
    MPI_Reduce(&integral, &integral_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // saída
    
    if(rank == 0)
    {
        printf("Com n=%d trapezios, a estimativa da\n", n);
        printf("integral de %.2f a %.2f e: %.6f\n", a, b, integral_tot);
        double integralR = int_f(b);
        double eAbs = fabs(integral_tot - integralR);
        double eRel = eAbs/integralR;
        printf("O valor exato é de %.12f, Erel %.12f, Eabs %.12f\n", integralR, eRel, eAbs);
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

double f(double x)
{
    return 5 * x * x * x * x + 4 * x * x * x + 3 * x * x + 2 * x;
}

double int_f(double x)
{
    return x * x * x * x * x + x * x * x * x + x * x * x + x * x;
}