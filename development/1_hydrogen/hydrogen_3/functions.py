import os
import csv
import matplotlib.pyplot as plt
from qiskit_algorithms import VQE
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

def vqe_algorithm(shots, maxiter, seed=170):
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    mapper = JordanWignerMapper()

    fermionic_op = problem.hamiltonian.second_q_op()

    # define ansatz and optimizer
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

    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": shots},
        transpile_options={"seed_transpiler": seed},
    )

    vqe = VQE(noiseless_estimator, ansatz, optimizer=COBYLA(maxiter=maxiter))

    return vqe


def vqe_distance(solver, vqe, distance):
    # build the model:
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    # it'll return an instance of ElectronicStructureProblem:
    problem = driver.run()

    mapper = JordanWignerMapper()
    fermionic_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)

    hamiltonian = problem.hamiltonian
    result = solver.solve(problem)

    """
    vqe.estimator = noisy_estimator
    """

    result_vqe = vqe.compute_minimum_eigenvalue(operator=qubit_op)
    
    return (hamiltonian, result, result_vqe)



def save_csv(distances, energies, energies_vqe, path):

    folder = path.split('/')[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    

    with open(path, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['distance', 'energy', 'energy_vqe'])
        for x, y, z in zip(distances, energies, energies_vqe):
            escritor_csv.writerow([x, y, z])


def save_plt(distances, energies, energies_vqe, path):

    folder = path.split('/')[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.plot(distances, energies, marker='', linestyle='-', label='Exact solution')
    plt.plot(distances, energies_vqe, marker='', linestyle='-', label='UCCSD Ansatz')
    plt.xlabel('Bond length/Å')
    plt.ylabel('Energy/Hartree')
    plt.title('Hydrogen Molecule')
    plt.legend()
    plt.savefig(path)
    plt.show()
