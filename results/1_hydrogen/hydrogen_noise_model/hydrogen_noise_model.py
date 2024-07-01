import os
import sys

folder = folder = os.path.dirname(__file__).split('/')[-1]
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

import json
import numpy as np
import hydrogen_functions
from qiskit_algorithms import VQE
from qiskit_aer.noise import NoiseModel
from qiskit_nature.units import DistanceUnit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


optimizer_list = [COBYLA]
optimizer_list_str = ['COBYLA']
shots_list = [128, 16384]


# Save an IBM Quantum account and set it as your default account.
with open(f'credentials.json', 'r') as file:
    credentials = json.load(file)
    api_token = credentials.get('api_token')
QiskitRuntimeService.save_account(channel="ibm_quantum", token=api_token, set_as_default=True, overwrite=True)
 
# Load saved credentials
service = QiskitRuntimeService()
backends = ['ibm_sherbrooke', 'ibm_brisbane', 'ibm_osaka', 'ibm_kyoto']


def hamiltonian_distance(distance):
    # build the model:
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()
    
    return problem


solver = GroundStateEigensolver(
    JordanWignerMapper(),
    NumPyMinimumEigensolver(),
    )

driver = PySCFDriver(
    atom=f"H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
    )

problem = driver.run()
mapper = JordanWignerMapper()
fermionic_op = problem.hamiltonian.second_q_op()

num_spatial_orbitals = int(fermionic_op.num_spin_orbitals/2)
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


# VQE
distances = np.linspace(0.25, 2.5, 200)

for i, optimizer in enumerate(optimizer_list):
    for shots in shots_list:
        for backend_name in backends:
            energies = []
            energies_vqe = []

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

            vqe = VQE(noisy_estimator, ansatz, optimizer=optimizer())
            vqe.initial_point = np.zeros(ansatz.num_parameters)
            
            for distance in distances:
                problem = hamiltonian_distance(distance)
                hamiltonian = problem.hamiltonian

                mapper = JordanWignerMapper()
                fermionic_op = problem.hamiltonian.second_q_op()
                qubit_op = mapper.map(fermionic_op)

                result = solver.solve(problem)
                result_vqe = vqe.compute_minimum_eigenvalue(operator=qubit_op)

                energies.append(result.groundenergy + hamiltonian.nuclear_repulsion_energy)
                energies_vqe.append(result_vqe.eigenvalue.real + hamiltonian.nuclear_repulsion_energy)
                print(distance)

            hydrogen_functions.save_to_csv(f'{path}/{folder}/results_{optimizer_list_str[i]}_{shots}', f'backend={backend_name}', [distances, energies, energies_vqe])
