import os
import csv
import sympy as sp


x = sp.Symbol('x')
x_1 = sp.Symbol('x_1')
x_2 = sp.Symbol('x_2')


def well_wavefunction(n, L, x = sp.Symbol('x')):
    """
    Calcula la función de onda para un estado dado (n) en un pozo de potencial infinito.
    
    Parámetros:
        n (int): Estado cuántico del electrón.
    
    Retorna:
        La función de onda.
    """
    
    # Factor de normalización
    normalization_factor = sp.sqrt(2/L)
    
    # Función de onda radial
    well_wavefunction = normalization_factor * sp.sin(n*sp.pi*x/L)
    
    return well_wavefunction


def save_to_csv(folder, name, data):
    v_0 = data[0]
    energies = data[1]

    if not os.path.exists(folder):
        os.makedirs(folder)

    nombre_archivo = f'{name}.csv'
    ruta_archivo = os.path.join(folder, nombre_archivo)

    with open(ruta_archivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['v_0', 'energy'])
        for x, y in zip(v_0, energies):
            escritor_csv.writerow([x, y])
