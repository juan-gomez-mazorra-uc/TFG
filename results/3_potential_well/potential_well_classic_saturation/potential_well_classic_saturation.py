import os
import sys

folder = folder = os.path.dirname(__file__).split('/')[-1]
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

import numpy as np
import sympy as sp
import pandas as pd
import potential_well_functions


L = 1
# N_list = [2, 3, 4]
N_list = [3]
V_0_list = np.linspace(0, 100, 200)


# Define the wave function |psi>
def psi(n, x):
    return sp.sqrt(2 / L) * sp.sin(n * sp.pi * x / L)


# Verify normalization
x = sp.symbols('x')
n = sp.symbols('n', integer=True, positive=True)
psi_expr = sp.sqrt(2 / L) * sp.sin(n * sp.pi * x / L)
normalization = sp.integrate(psi_expr**2, (x, 0, L))
# print(normalization)


for N in N_list:
    # Initialize N_max and matrices
    phi = [0 for _ in range(N**2)]
    H_ij = [[0 for _ in range(N**2)] for _ in range(N**2)]

    # Define the basis functions |phi>
    y = sp.symbols('y')
    for i in range(1, N + 1):
        phi[N * (i - 1) + i - 1] = psi(i, x) * psi(i, y)
        for j in range(i + 1, N + 1):
            phi[N * (i - 1) + j - 1] = (psi(i, x) * psi(j, y) + psi(j, x) * psi(i, y)) / sp.sqrt(2)
        for j in range(1, i):
            phi[N * (i - 1) + j - 1] = (psi(i, x) * psi(j, y) - psi(j, x) * psi(i, y)) / sp.sqrt(2)

    # Define the kinetic energy operator K
    K = [-(sp.diff(phi_i, x, x) + sp.diff(phi_i, y, y)) / 2 for phi_i in phi]

    # Populate the Hamiltonian matrix H_ij
    v0 = sp.symbols('v0')
    for i in range(N**2):
        for j in range(i + 1):
            H_ij[i][j] = sp.integrate(phi[i] * K[j], (x, 0, L), (y, 0, L))
            H_ij[i][j] += v0 * sp.integrate((phi[i] * phi[j]).subs(y, x), (x, 0, L))
            H_ij[j][i] = H_ij[i][j]

    H_ij_matrix = sp.Matrix(H_ij)


    eigenvalues = [[] for _ in range(N**2)]

    for V_0 in V_0_list:
        H_ij_matrix_numeric = H_ij_matrix.subs({v0: V_0}).evalf()
        eigen_info = H_ij_matrix_numeric.eigenvects()
        for eig in eigen_info:
            eigenvalue = eig[0]
            eigenvector = eig[2][0]
            # eigenvectors.append(eigenvector)

            max_eigenvector = abs(eigenvector[0, 0])
            max_index = 0

            for i in range(1, len(eigenvector)):
                if abs(eigenvector[i, 0]) > max_eigenvector:
                    max_eigenvector = abs(eigenvector[i, 0])
                    max_index = i

            eigenvalues[max_index].append(eigenvalue)


    data = {'v0': V_0_list}
    for i, eigenvalue in enumerate(eigenvalues):
        if i <= 3:
            data[f'energy_{i}'] = eigenvalue

    if not os.path.exists(f'{path}/{folder}/results_{L} (V large)'):
        os.makedirs(f'{path}/{folder}/results_{L} (V large)')

    df = pd.DataFrame(data)
    df.to_csv(f'{path}/{folder}/results_{L} (V large)/N={N}.csv', index=False)
