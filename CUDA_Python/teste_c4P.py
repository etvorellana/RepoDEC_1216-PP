import ctypes
import os
import numpy as np
import numpy.ctypeslib as npct

# Determina o nome da biblioteca com base no sistema operacional
if os.name == 'posix':  # Linux ou macOS
    lib_name = 'c4p_lib.so'
elif os.name == 'nt':  # Windows
    lib_name = 'c4p_lib.dll'
else:
    raise RuntimeError("Sistema operacional não suportado")

# Carrega a biblioteca C
try:
    c4p_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), lib_name))
except OSError as e:
    print(f"Erro ao carregar a biblioteca C: {e}")
    print("Certifique-se de que a biblioteca (c4p_lib.so ou c4p_lib.dll) está no mesmo diretório do script Python.")
    exit(1)

# Define os tipos de argumento e o tipo de retorno da função C
# 'double *ptr1' -> ctypes.POINTER(ctypes.c_double)
# 'double *ptr2' -> ctypes.POINTER(ctypes.c_double)
# 'int a' -> ctypes.c_int
# 'int b' -> ctypes.c_int
c4p_lib.calculate_stuff.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
# 'double' -> ctypes.c_double
c4p_lib.calculate_stuff.restype = ctypes.c_double

# Cria variáveis Python que serão passadas por referência
# Para passar um double por referência, você precisa de um objeto c_double
# e então obter um ponteiro para ele usando ctypes.byref() ou ctypes.addressof()
# Ou, mais comumente, criar uma instância de c_double e passar seu endereço.
# Aqui, usaremos ctypes.byref() para objetos c_double.
val1_c = ctypes.c_double(10.5)
val2_c = ctypes.c_double(20.3)
int_a = 5
int_b = 3

# Chama a função C
# Passamos ctypes.byref(val1_c) e ctypes.byref(val2_c) para obter os ponteiros
# para os objetos c_double.
result = c4p_lib.calculate_stuff(
    ctypes.byref(val1_c),
    ctypes.byref(val2_c),
    int_a,
    int_b
)

print(f"Valor 1 (Python): {val1_c.value}")
print(f"Valor 2 (Python): {val2_c.value}")
print(f"Inteiro A (Python): {int_a}")
print(f"Inteiro B (Python): {int_b}")
print(f"Resultado da função C: {result}")

# Exemplo com valores diferentes
val3_c = ctypes.c_double(1.0)
val4_c = ctypes.c_double(2.0)
int_c = 10
int_d = 2

result2 = c4p_lib.calculate_stuff(
    ctypes.byref(val3_c),
    ctypes.byref(val4_c),
    int_c,
    int_d
)
print(f"\nResultado da segunda chamada: {result2}")

# Exemplo passando ponteiros nulos (se você quisesse testar o tratamento de erro em C)
# Embora ctypes não tenha um equivalente direto para um "ponteiro nulo",
# passar None pode funcionar em alguns casos, mas a verificação em C é mais robusta.
# A melhor forma de testar ponteiro nulo é se a função C espera um ponteiro e você passa um tipo inválido.
# Para este caso, a verificação `if (ptr1 == NULL || ptr2 == NULL)` na função C é o que importa.

# Define os tipos de argumento e o tipo de retorno da função C
# Usamos npct.ndpointer para descrever arrays NumPy
# dtype: tipo de dados esperado (e.g., np.float64 para double em C)
# ndim: número de dimensões (1 para arrays unidimensionais)
# flags: 'C_CONTIGUOUS' (garante que os dados são armazenados de forma contígua em C-order)
#        'WRITEABLE' (permite que a função C modifique o array)
c4p_lib.process_numpy_arrays.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS, WRITEABLE'), # array 1
    ctypes.c_int,                                                              # size 1
    npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS, WRITEABLE'), # array 2
    ctypes.c_int,                                                              # size 2
    ctypes.c_int,
    ctypes.c_int
]
c4p_lib.process_numpy_arrays.restype = ctypes.c_double

# Cria dois arrays NumPy
array1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
array2 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
int_a = 2
int_b = 4

print(f"Array 1 original (Python): {array1}")
print(f"Array 2 original (Python): {array2}")

# Chama a função C, passando os arrays e seus tamanhos
result = c4p_lib.process_numpy_arrays(
    array1,
    array1.size, # array.size retorna o número total de elementos
    array2,
    array2.size,
    int_a,
    int_b
)

print(f"\nResultado da função C: {result}")
print(f"Array 1 após chamada C (Python): {array1}") # Note que o primeiro elemento foi modificado!
print(f"Array 2 após chamada C (Python): {array2}") # Note que o primeiro elemento foi modificado!

# Exemplo com arrays vazios para testar a robustez da função C
print("\n--- Teste com arrays vazios ---")
empty_array1 = np.array([], dtype=np.float64)
empty_array2 = np.array([], dtype=np.float64)

result_empty = c4p_lib.process_numpy_arrays(
    empty_array1,
    empty_array1.size,
    empty_array2,
    empty_array2.size,
    int_a,
    int_b
)
print(f"Resultado com arrays vazios: {result_empty}")

# Exemplo com arrays de dimensões diferentes (para arrays 1D, não faz diferença, mas para 2D sim)
print("\n--- Teste com arrays de tamanhos diferentes ---")
arr_small = np.array([1.0], dtype=np.float64)
arr_large = np.arange(10, dtype=np.float64) # Array de 0 a 9

result_mixed = c4p_lib.process_numpy_arrays(
    arr_small,
    arr_small.size,
    arr_large,
    arr_large.size,
    1,
    1
)
print(f"Array pequeno após chamada C (Python): {arr_small}")
print(f"Array grande após chamada C (Python): {arr_large}")
print(f"Resultado com arrays de tamanhos diferentes: {result_mixed}")
