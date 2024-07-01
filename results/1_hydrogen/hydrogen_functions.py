import os
import csv


def save_to_csv(folder, name, data):
    distances = data[0]
    energies = data[1]
    energies_vqe = data[2]

    if not os.path.exists(folder):
        os.makedirs(folder)

    nombre_archivo = f'{name}.csv'
    ruta_archivo = os.path.join(folder, nombre_archivo)
    
    with open(ruta_archivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['distance', 'energy', 'energy_vqe'])
        for x, y, z in zip(distances, energies, energies_vqe):
            escritor_csv.writerow([x, y, z])
