import os
import sys

folder = folder = os.path.dirname(__file__).split('/')[-1]
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

import numpy as np
import hydrogen_functions
from qiskit_algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


optimizer_list = [COBYLA, SLSQP, SPSA]
optimizer_list_str = ['COBYLA', 'SLSQP', 'SPSA']
shots_list = [16, 128, 1024, 16384]
ansatz_str = 'UCCSD_no_initial_point' # 'TwoLocal' 'UCCSD' 'UCCSD_no_initial_point' 


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
    hamiltonian = problem.hamiltonian
    
    return hamiltonian


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

if ansatz_str == 'TwoLocal':
    ansatz = TwoLocal(fermionic_op.num_spin_orbitals, rotation_blocks="ry", entanglement_blocks="cz")

if ansatz_str == 'UCCSD' or 'UCCSD_no_initial_point':
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
distances = np.linspace(0.25, 2.5, 2)

for i, optimizer in enumerate(optimizer_list):
    for shots in shots_list:
        energies = []
        energies_vqe = []

        seed = 170
        algorithm_globals.random_seed = seed

        noiseless_estimator = AerEstimator(
            run_options={"seed": seed, "shots": shots},
            transpile_options={"seed_transpiler": seed},
        )

        vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer())
        if ansatz_str == 'TwoLocal':
            vqe.initial_point = np.zeros(16)
        if ansatz_str == 'UCCSD':
            vqe.initial_point = np.zeros(ansatz.num_parameters)
        
        for distance in distances:
            hamiltonian = hamiltonian_distance(distance)

            mapper = JordanWignerMapper()
            fermionic_op = problem.hamiltonian.second_q_op()
            qubit_op = mapper.map(fermionic_op)

            result = solver.solve(problem)
            result_vqe = vqe.compute_minimum_eigenvalue(operator=qubit_op)

            energies.append(result.groundenergy + hamiltonian.nuclear_repulsion_energy)
            energies_vqe.append(result_vqe.eigenvalue.real + hamiltonian.nuclear_repulsion_energy)
            print(distance)

        hydrogen_functions.save_to_csv(f'{path}/{folder}/hydrogen_aer_{ansatz_str}_aaa/results_{optimizer_list_str[i]}', f'shots={shots}', [distances, energies, energies_vqe])
