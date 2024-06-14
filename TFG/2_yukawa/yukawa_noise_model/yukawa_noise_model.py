import os
import sys

folder = folder = os.path.dirname(__file__).split('\\')[-1]
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

import json
import sympy as sp
import numpy as np
import yukawa_functions
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
Z = 0.75

# Save an IBM Quantum account and set it as your default account.
with open(f'credentials.json', 'r') as file:
    credentials = json.load(file)
    api_token = credentials.get('api_token')
QiskitRuntimeService.save_account(channel="ibm_quantum", token=api_token, set_as_default=True, overwrite=True)
 
# Load saved credentials
service = QiskitRuntimeService()
# backends = ['ibm_sherbrooke', 'ibm_brisbane', 'ibm_osaka', 'ibm_kyoto']
backends = ['ibm_osaka', 'ibm_kyoto']


def hamiltonian_alpha(alpha = 0, N = 3, l = 0):
    h_pq = []

    for i in range(1, N+1):
        h_pq.append([])
        for j in range(1, N+1):
            globals()[f'h_{i}{j}'] = -1/2 * sp.integrate(globals()[f'u_{i}0']*globals()[f'dd_u_{j}0'], (r, 0, sp.oo)) \
                                    - sp.integrate(globals()[f'u_{i}0']*globals()[f'u_{j}0'] * sp.exp(-alpha*r)/r, (r, 0, sp.oo))
            h_pq[i-1].append(globals()[f'h_{i}{j}'])

    size = len(h_pq)
    h1_a = np.array(h_pq, dtype=float)
    h2_aa = np.zeros((size, size, size, size))
    hamiltonian = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa)
    
    return hamiltonian


# Radial Wavefunction Solutions
n_max = 8
r = sp.Symbol('r')

for n in range(1, n_max+1):
    for l in range(n):
        globals()[f'u_{n}{l}'] = r * yukawa_functions.radial_wavefunction(n, l, Z)
        globals()[f'dd_u_{n}{l}'] = sp.diff(globals()[f'u_{n}{l}'], r, r)


# VQE
alphas = np.linspace(0, 2, 200)

for N in N_list:
    for i, optimizer in enumerate(optimizer_list):
        for shots in shots_list:
            for backend_name in backends:
                energies = []

                hamiltonian = hamiltonian_alpha(0, N)

                mapper = JordanWignerMapper()
                fermionic_op = hamiltonian.second_q_op()
                qubit_op = mapper.map(fermionic_op)

                num_spatial_orbitals = int(fermionic_op.num_spin_orbitals/2)
                # The tuple of the number of alpha- and beta-spin particles
                num_particles = (1, 0)

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

                for alpha in alphas:
                    hamiltonian = hamiltonian_alpha(alpha, N)

                    mapper = JordanWignerMapper()
                    fermionic_op = hamiltonian.second_q_op()
                    qubit_op = mapper.map(fermionic_op)
                    
                    eigenvalue = vqe_solver.compute_minimum_eigenvalue(operator=qubit_op).eigenvalue
                    energies.append(eigenvalue)
                    print(eigenvalue)

                yukawa_functions.save_to_csv(f'{path}/{folder}/results_{optimizer_list_str[i]}_{shots}', f'backend={backend_name}', [alphas, energies])
