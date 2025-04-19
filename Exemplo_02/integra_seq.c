
#include <stdio.h>
#include <math.h>
#include <omp.h>

double integraTrap(double a, double b, int n);
double f(double x);
double int_f(double x);

int main(int argc, char *argv[])
{
    FILE *fp;
    double integral = 0.0, a, b;
    int n;
    // Leitura dos parâmetros de entrada
    double start = omp_get_wtime();
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
    double stop = omp_get_wtime();
    double t1 = stop - start;
    // calcular a integral
    integral = integraTrap(a, b, n);
    // saída
    stop = omp_get_wtime();
    double t2 = stop - start;
    printf("Com n=%d trapezios, a estimativa é\n", n);
    printf("integral de %.2f a %.2f e: %.12f\n", a, b, integral);
    //printf("f(%.12f) = %.12f, f(%.12f) = %.12f\n", a, b, int_f(a), int_f(b));
    double integralR = int_f(b) - int_f(a);
    double eAbs = fabs(integral - integralR);
    double eRel = eAbs/integralR;
    printf("O valor exato é de %.12f, Erel %.12f, Eabs %.12f\n", integralR, eRel, eAbs);
    stop = omp_get_wtime();
    double t3 = stop - start;
    printf("Tempo de leitura: %.12f\n", t1);
    printf("Tempo até o cálculo: %.12f\n", t2);
    printf("Tempo de cálculo: %.12f\n", t2 - t1);
    printf("Tempo total: %.12f\n", t3);
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

