import os
import sys

folder = folder = os.path.dirname(__file__).split('/')[-1]
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

import json
import sympy as sp
import numpy as np
import potential_well_functions
from qiskit_algorithms import VQE
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


optimizer_list = [COBYLA]
optimizer_list_str = ['COBYLA']
N_list = [2]
shots_list = [128, 16384]
L = 1

# Save an IBM Quantum account and set it as your default account.
with open(f'credentials.json', 'r') as file:
    credentials = json.load(file)
    api_token = credentials.get('api_token')
QiskitRuntimeService.save_account(channel="ibm_quantum", token=api_token, set_as_default=True, overwrite=True)
 
# Load saved credentials
service = QiskitRuntimeService()
# backends = ['ibm_sherbrooke', 'ibm_brisbane', 'ibm_osaka', 'ibm_kyoto']
backends = ['ibm_brisbane', 'ibm_osaka', 'ibm_kyoto']


def hamiltonian_interaction(L, N = 3, v_0 = 1):
    h_pq = []

    for i in range(1, N+1):
        h_pq.append([])
        for j in range(1, N+1):
            globals()[f'h_{i}{j}'] = -1/2 * sp.integrate(globals()[f'phi_{i}']*globals()[f'dd_phi_{j}'], (x, 0, L))
            h_pq[i-1].append(globals()[f'h_{i}{j}'])

    size = len(h_pq)
    h1_a = np.array(h_pq, dtype=float)
    
    h_ijkl = np.zeros((size, size, size, size))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    globals()[f'h_{i}{j}{k}{l}'] = v_0*\
                                                   globals()[f'phi_{i+1}_1']*globals()[f'phi_{j+1}_2']*\
                                                   globals()[f'phi_{k+1}_2']*globals()[f'phi_{l+1}_1']

                    h_ijkl[i][j][k][l] = sp.integrate(globals()[f'h_{i}{j}{k}{l}'], (x_1, 0, L))

    h2_aa = np.array(h_ijkl, dtype=float)

    hamiltonian = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa)

    return hamiltonian


# Wavefunction Solutions
n_max = 8
x = sp.Symbol('x')
x_1 = sp.Symbol('x_1')

for n in range(1, n_max+1):
    for l in range(n):
        globals()[f'phi_{n}'] = potential_well_functions.well_wavefunction(n, L)
        globals()[f'dd_phi_{n}'] = sp.diff(globals()[f'phi_{n}'], x, x)

for n in range(1, n_max+1):
    for l in range(n):
        globals()[f'phi_{n}_1'] = potential_well_functions.well_wavefunction(n, L, sp.Symbol('x_1'))
        globals()[f'phi_{n}_2'] = potential_well_functions.well_wavefunction(n, L, sp.Symbol('x_1'))


# VQE
V_0_list = np.linspace(0, 10, 200)

for N in N_list:
    for i, optimizer in enumerate(optimizer_list):
        for shots in shots_list:
            for backend_name in backends:
                energies = []

                hamiltonian = hamiltonian_interaction(L, N, 1)

                mapper = JordanWignerMapper()
                fermionic_op = hamiltonian.second_q_op()
                qubit_op = mapper.map(fermionic_op)

                num_spatial_orbitals = int(fermionic_op.num_spin_orbitals/2)
                # The tuple of the number of alpha- and beta-spin particles
                num_particles = (1, 1)

                ansatz = UCCSD(
                    num_spatial_orbitals,
                    num_particles,
                    mapper,
                    initial_state=HartreeFock(
                        num_spatial_orbitals,
                        num_particles,
                        mapper,
                    ),
                )

                seed = 170
                algorithm_globals.random_seed = seed

                backend = service.backend(backend_name)
                noise_model = NoiseModel.from_backend(backend)

                noisy_estimator = AerEstimator(
                    backend_options={
                    "method": "density_matrix",
                    "noise_model": noise_model,
                },
                    run_options={"seed": seed, "shots": shots},
                    transpile_options={"seed_transpiler": seed},
                )

                vqe_solver = VQE(noisy_estimator, ansatz, optimizer())
                vqe_solver.initial_point = np.zeros(ansatz.num_parameters)

                for V_0 in V_0_list:
                    hamiltonian = hamiltonian_interaction(L, N, V_0)

                    mapper = JordanWignerMapper()
                    fermionic_op = hamiltonian.second_q_op()
                    qubit_op = mapper.map(fermionic_op)
                    
                    eigenvalue = vqe_solver.compute_minimum_eigenvalue(operator=qubit_op).eigenvalue
                    energies.append(eigenvalue)
                    print(eigenvalue)

                potential_well_functions.save_to_csv(f'{path}/{folder}/results_{optimizer_list_str[i]}_{shots}', f'backend={backend_name}', [V_0_list, energies])
