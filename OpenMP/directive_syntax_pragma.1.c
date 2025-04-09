/*
* @@name:	directive_syntax_pragma.1
* @@type:	C
* @@operation:	run
* @@expect:	success
* @@version:	pre_omp_3.0
*/
#include <libiomp/omp.h>
#include <stdio.h>
#define NT 4


int main(){
    #pragma omp parallel for num_threads(NT)                 // PRAG 1
    for(int i=0; i<NT; i++) 
        printf("thrd no %d\n", omp_get_thread_num());

    #pragma omp parallel for num_threads(NT)                // PRAG 2
    for(int i=0; i<NT; i++) 
        printf("thrd no %d\n", omp_get_thread_num());

    #pragma omp parallel num_threads(NT)                     // PRAG 3-4
    #pragma omp for
    for(int i=0; i<NT; i++) 
        printf("thrd no %d\n", omp_get_thread_num());

    #pragma omp parallel num_threads(NT)                     // PRAG 5
    {
       int no = omp_get_thread_num();
       if (no%2) 
            printf("thrd no %d is Odd \n",no);
       else 
            printf("thrd no %d is Even\n",no);

       #pragma omp for                                    
       for(int i=0; i<NT; i++) 
        printf("thrd no %d\n", omp_get_thread_num());
    }
}
/*
      repeated 4 times, any order
      OUTPUT: thrd no 0
      OUTPUT: thrd no 1
      OUTPUT: thrd no 2
      OUTPUT: thrd no 3
      
      any order
      OUTPUT: thrd no 0 is Even
      OUTPUT: thrd no 2 is Even
      OUTPUT: thrd no 1 is Odd
      OUTPUT: thrd no 3 is Odd
*/
