import os
import sys

folder = folder = os.path.dirname(__file__).split('\\')[-1]
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

import sympy as sp
import numpy as np
import potential_well_functions


N_list = [2, 3, 4, 5, 6, 7, 8]
L = 1


# Wavefunction Solutions
n_max = 8
x = sp.Symbol('x')

for n in range(1, n_max+1):
    for l in range(n):
        globals()[f'phi_{n}'] = potential_well_functions.well_wavefunction(n, L)
        globals()[f'dd_phi_{n}'] = sp.diff(globals()[f'phi_{n}'], x, x)


# Numerical solutions
alphas = np.linspace(0, 2, 200)

for N in N_list:
    energies = []

    for alpha in alphas:
        h_pq = []
        for i in range(1, N+1):
            h_pq.append([])
            for j in range(1, N+1):
                globals()[f'h_{i}{j}'] = -1/2 * sp.integrate(globals()[f'u_{i}0']*globals()[f'dd_u_{j}0'], (r, 0, sp.oo)) \
                                        - sp.integrate(globals()[f'u_{i}0']*globals()[f'u_{j}0'] * sp.exp(-alpha*r)/r, (r, 0, sp.oo))
                h_pq[i-1].append(globals()[f'h_{i}{j}'])
        
        h_pq_matrix = sp.Matrix(h_pq)
        eigenvalue = np.min(list(h_pq_matrix.eigenvals().keys()))
        energies.append(eigenvalue)

    yukawa_functions.save_to_csv(f'{path}/{folder}/N={N}/results_{Z}', f'N={N}', [alphas, energies])