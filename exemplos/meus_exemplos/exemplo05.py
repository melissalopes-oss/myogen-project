#Exemplo 5 do MyoGen - Surface Motor Unit Action Potentials (MUAPs)
#Simula o "eco elétrico" (MUAP) que cada unidade motora produz quando
#dispara, captado por uma grade de eletrodos HD-sEMG na pele.

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

from myogen import simulator
from myogen.utils.neo import signal_to_grid

plt.style.use("fivethirtyeight")

sampling_frequency = 2048.0 * pq.Hz


# Load Muscle Model

save_path = Path("./results")
muscle: simulator.Muscle = joblib.load(save_path / "muscle_model.pkl")


# Create Surface EMG Model

electrode_array_monopolar = simulator.SurfaceElectrodeArray(
    num_rows=3,
    num_cols=3,
    inter_electrode_distances__mm=5 * pq.mm,
    electrode_radius__mm=5 * pq.mm,
    differentiation_mode="monopolar",
    bending_radius__mm=muscle.radius__mm + muscle.skin_thickness__mm + muscle.fat_thickness__mm,
)

surface_emg = simulator.SurfaceEMG(
    muscle_model=muscle,
    electrode_arrays=[electrode_array_monopolar],
    sampling_frequency__Hz=sampling_frequency,
    sampling_points_in_t_and_z_domains=256,
    sampling_points_in_theta_domain=32,
    MUs_to_simulate=[0, 1, 2, 3],  # Simula as unidades motoras 0, 1, 2 e 3
)

# Simulate MUAPs

muaps = surface_emg.simulate_muaps()

print("MUAP simulation completed!")
first_signal = muaps.groups[0].segments[0].analogsignals[0]
grid_shape = first_signal.annotations["grid_shape"]
print(f"Generated MUAPs shape: {first_signal.shape} (stored as 2D for NWB compatibility)")
print(f"  - {len(muaps.groups[0].segments)} total motor units in muscle")
print(f"  - {len(muscle.resulting_number_of_innervated_fibers)} MUs in muscle model")
print(f"  - {grid_shape[0]} rows × {grid_shape[1]} columns electrode grid")
print(f"  - {first_signal.shape[0]} time samples")

# Verifica quantas fibras existem nas últimas unidades motoras do músculo
print("\nFiber counts for simulated MUs:")
for mu_idx in range(95, min(100, len(muscle.resulting_number_of_innervated_fibers))):
    n_fibers = muscle.resulting_number_of_innervated_fibers[mu_idx]
    print(f"  MU {mu_idx}: {n_fibers} fibers")

# Salva os resultados
joblib.dump(surface_emg, save_path / "surface_emg.pkl")

# Plot MUAPs
# Plota o MUAP de cada unidade motora simulada, mostrando a forma
# de onda captada em cada um dos 25 eletrodos da grade (5x5).

for muap_index in range(len(muaps.groups[0].segments)):
    if np.mean(muaps.groups[0].segments[muap_index].analogsignals[0]) == 0:
        continue

    # Extrai os dados do MUAP
    muap_signal = muaps.groups[0].segments[muap_index].analogsignals[0]

    # Converte para formato de grade 3D (tempo, linhas, colunas)
    muap_data_mV = signal_to_grid(muap_signal)
    muap_data_uV = signal_to_grid(muap_signal.rescale(pq.uV))

    # Mostra diagnóstico de amplitude
    print(f"\nMUAP {muap_index} amplitude range:")
    print(f"\tMin: {np.min(muap_data_mV):.6f} mV ({np.min(muap_data_uV):.2f} uV)")
    print(f"\tMax: {np.max(muap_data_mV):.6f} mV ({np.max(muap_data_uV):.2f} uV)")
    print(f"\tPeak-to-peak: {np.ptp(muap_data_mV):.6f} mV ({np.ptp(muap_data_uV):.2f} uV)")

    # Pega as dimensões da grade
    n_rows, n_cols = muap_signal.annotations["grid_shape"]

    # Cria a grade de subplots (um gráfico por eletrodo)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), sharex=True, sharey=True
    )
    fig.suptitle(f"MUAP {muap_index} ({np.ptp(muap_data_mV):.1f} mV peak-to-peak)")

    time_vector = muap_signal.times.rescale(pq.ms).magnitude

    # Plota a forma de onda de cada eletrodo
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].plot(
                time_vector,
                muap_data_mV[:, row, col],
                linewidth=1.5,
            )
            axs[row, col].grid(False)
            if row == n_rows - 1 and col == n_cols // 2:
                axs[row, col].set_xticks([0, 7.5, 15])
                axs[row, col].set_xlabel("Time (ms)")
            if col == 0 and row == n_rows // 2:
                axs[row, col].set_ylabel("Amplitude (mV)")

    plt.tight_layout()
    plt.show()

print("\n[DONE] Simulação dos MUAPs completa!")