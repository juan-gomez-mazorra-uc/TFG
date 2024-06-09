from qiskit_nature.second_q.drivers import PySCFDriver
# Se importa DistanceUnit para especificar las unidades de distancia
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_qubit_op(distance):
    # Se crea el modelo
    driver = PySCFDriver(
      atom=f"H 0 0 0; H 0 0 {distance}",  # Geometría de la molécula de H2
      basis="sto3g",  # Base de funciones STO-3G para representar las funciones de onda
      charge=0,  # Molécula neutra, sin carga total
      spin=0,  # Multiplicidad del espín 1 (singlete, ya que spin = 0)
      unit=DistanceUnit.ANGSTROM  # Las distancias se especifican en Angstroms.
    )

    # El driver devuelve una instancia de ElectronicStructureProblem
    problem = driver.run()

    # Operador Hamiltoniano en su representación de segundo cuantización
    fermionic_op = problem.hamiltonian.second_q_op()

    # Mapper empleado para transformar operadores fermiónicos a operadores de qubits
    mapper = JordanWignerMapper()
 
    # Operador de qubits utilizando el mapper de Jordan-Wigner
    qubit_op = mapper.map(fermionic_op)
	
    return problem, fermionic_op, qubit_op


import numpy as np
from qiskit_algorithms import VQE
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver


def energy(distances, ansatz, optimizer, shots):

    solver = GroundStateEigensolver(
        JordanWignerMapper(),
        NumPyMinimumEigensolver(),
        )

    seed = 170
    algorithm_globals.random_seed = seed

    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": shots},
        transpile_options={"seed_transpiler": seed},
    )

    vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer())
    vqe.initial_point = np.zeros(ansatz.num_parameters)

    energies = []
    energies_vqe = []

    for distance in distances:

        problem, fermionic_op, qubit_op = get_qubit_op(distance)
            
        hamiltonian = problem.hamiltonian

        result = solver.solve(problem)
        result_vqe = vqe.compute_minimum_eigenvalue(operator=qubit_op)

        energies.append(result.groundenergy + hamiltonian.nuclear_repulsion_energy)
        energies_vqe.append(result_vqe.eigenvalue.real + hamiltonian.nuclear_repulsion_energy)

    return (energies, energies_vqe)

print(energy(distances, ansatz, optimizer, shots))



import numpy as np
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

def get_ansatz(mapper=JordanWignerMapper()):

    problem, fermionic_op, qubit_op = get_qubit_op(0.735)

    # define ansatz
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

    return ansatz


distances = np.linspace(0.25, 2.5, 3)
ansatz = get_ansatz(mapper)
optimizer = COBYLA
shots = 1024

energy(distances, ansatz, optimizer, shots)
