import os
import csv
import sympy as sp


x = sp.Symbol('x')
r = sp.Symbol('r')


def laguerre(j, x):
    f = sp.exp(-x) * x**j
    for n in range(j):
        f = sp.diff(f, x)

    f = sp.exp(x) * f

    globals()[f'L_{j}'] = f

    return sp.simplify(globals()[f'L_{j}'])


def laguerre_associated(n, l, Z):
    j = n-l-1
    k = 2*l+1

    f = laguerre(j+k, x)

    for i in range(k):
        f = sp.diff(f, x)

    globals()[f'L_{j}^{k}'] = (-1)**k * f
    return globals()[f'L_{j}^{k}'].subs(x, 2*r*Z/n)


def radial_wavefunction(n, l, Z):
    """
    Calcula la función de onda radial para un estado dado (n, l) en un átomo con número atómico Z.
    
    Parámetros:
        n (int): Número cuántico principal.
        l (int): Número cuántico azimutal.
        Z (float): Número atómico (número de protones en el núcleo).
    
    Retorna:
        La función de onda radial.
    """
    
    # Polinomio de Laguerre generalizado
    laguerre_poly = laguerre_associated(n, l, Z)
    
    # Factor de normalización
    normalization_factor = sp.sqrt((2*Z/n)**3 * sp.factorial(n-l-1) / (2*n*sp.factorial(n+l)**3))
    
    # Exponencial
    exponential = sp.exp(-Z*r/n)
    
    # Función de onda radial    
    return normalization_factor * exponential * (2*r*Z/n)**l * laguerre_poly


def save_to_csv(folder, name, data):
    alphas = data[0]
    energies = data[1]

    if not os.path.exists(folder):
        os.makedirs(folder)

    nombre_archivo = f'{name}.csv'
    ruta_archivo = os.path.join(folder, nombre_archivo)

    with open(ruta_archivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['alpha', 'energy'])
        for x, y in zip(alphas, energies):
            escritor_csv.writerow([x, y])
