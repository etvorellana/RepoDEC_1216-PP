#include <stdio.h>

// Definimos uma macro para exportar a função em sistemas Windows.
// Em sistemas Unix/Linux/macOS, isso não é estritamente necessário para funções C simples,
// mas é uma boa prática para portabilidade se você estiver construindo uma DLL/shared library.
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/* Como compilar
    gcc -shared -fPIC -o c4p_lib.so c4p.c
*/


// A função que será chamada do Python
EXPORT double calculate_stuff(double *ptr1, double *ptr2, int a, int b) {
    // Verifica se os ponteiros não são nulos antes de desreferenciá-los
    if (ptr1 == NULL || ptr2 == NULL) {
        fprintf(stderr, "Erro: Ponteiro nulo passado para calculate_stuff.\n");
        return 0.0; // Ou algum valor de erro apropriado
    }

    double result = (*ptr1 + *ptr2) * (double)(a + b);
    return result;
}

// A função que será chamada do Python, recebendo ponteiros para arrays e seus tamanhos
// double *arr1: Ponteiro para o início do primeiro array NumPy
// int size1: Tamanho do primeiro array
// double *arr2: Ponteiro para o início do segundo array NumPy
// int size2: Tamanho do segundo array
// int a, int b: Os dois inteiros adicionais
EXPORT double process_numpy_arrays(double *arr1, int size1, double *arr2, int size2, int a, int b) {
    if (arr1 == NULL || arr2 == NULL) {
        fprintf(stderr, "Erro: Ponteiro de array nulo passado para process_numpy_arrays.\n");
        return 0.0; // Ou algum valor de erro apropriado
    }

    // Exemplo de como acessar os elementos dos arrays
    double sum_arr1 = 0.0;
    for (int i = 0; i < size1; ++i) {
        sum_arr1 += arr1[i];
    }

    double sum_arr2 = 0.0;
    for (int i = 0; i < size2; ++i) {
        sum_arr2 += arr2[i];
    }

    // Exemplo de cálculo: soma dos elementos de ambos os arrays multiplicada pela soma dos inteiros
    double result = (sum_arr1 + sum_arr2) * (double)(a + b);

    // Você também pode modificar os arrays in-place se a flag 'WRITEABLE' for definida no Python
    // Por exemplo, dobrar o primeiro elemento de cada array:
    if (size1 > 0) {
        arr1[0] *= 2.0;
    }
    if (size2 > 0) {
        arr2[0] *= 2.0;
    }

    return result;
}

