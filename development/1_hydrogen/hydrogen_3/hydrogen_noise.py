import os
import functions
import numpy as np
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

"""
GroundStateEigensolver
"""

solver = GroundStateEigensolver(
        JordanWignerMapper(),
        NumPyMinimumEigensolver(),
    )


"""
Barrido distancias
"""

distances = np.linspace(0.25, 4, 2)
shots_list = [16, 128, 1024, 16384]

for shots in shots_list:
    print(shots)
    energies = []
    energies_vqe = []

    vqe = functions.vqe_algorithm(shots, maxiter=100, seed=100)
    for distance in distances:
        print(distance)
        hamiltonian, result, result_vqe = functions.vqe_distance(solver, vqe, distance)

        energies.append(result.groundenergy + hamiltonian.nuclear_repulsion_energy)
        energies_vqe.append(result_vqe.eigenvalue.real + hamiltonian.nuclear_repulsion_energy)

    
    path_csv = os.path.join('results', f'shots={shots}.csv')
    path_plt = os.path.join('results', f'shots={shots}.png')
    
    functions.save_csv(distances, energies, energies_vqe, path_csv)
    functions.save_plt(distances, energies, energies_vqe, path_plt)
