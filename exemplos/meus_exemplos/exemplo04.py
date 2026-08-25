
#Exemplo 4 do MyoGen - Muscle Model

#cria modelo físico do músculo:distribui as fibras musculares e define o território (área de inervação) de cada unidade motora.
#este exemplo depende do exemplo 1 (precisa do thresholds.pkl)

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

from myogen import simulator
from myogen.utils.plotting.muscle import plot_innervation_areas_2d, plot_mf_centers

plt.style.use("fivethirtyeight")

# Load recruitment thresholds (gerado no exemplo 1)
save_path = Path("./results")

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

# Define muscle parameters
muscle_radius = 4.9 * pq.mm  # Muscle radius in mm
mean_fiber_length = 32 * pq.mm  # Mean fiber length in mm
fiber_length_variation = 3 * pq.mm  # Fiber length variation in mm
fiber_density = 400 * pq.mm**-2  # Fiber density in fibers per mm²

# Define simulation parameters
max_innervation_ratio = 1 / 4  # Maximum motor unit territory size
grid_resolution = 256  # Spatial resolution for muscle discretization

muscle = simulator.Muscle(
    recruitment_thresholds=recruitment_thresholds,
    radius_bone__mm=1.0 * pq.mm,
    fiber_density__fibers_per_mm2=fiber_density,
    fat_thickness__mm=10 * pq.mm,
    autorun=True,
)

# Salva o modelo do músculo para não precisar rodar de novo depois
joblib.dump(muscle, save_path / "muscle_model.pkl")

# Exibe estatísticas do músculo
total_fibers = sum(muscle.resulting_number_of_innervated_fibers)

print("Muscle model statistics:")
print(f"\tTotal muscle fibers: {total_fibers}")
print(f"\tMean fibers per MU: {total_fibers / len(recruitment_thresholds):.1f}")
print(f"\tMuscle cross-sectional area: {np.pi * muscle_radius**2:.1f}")

plt.figure(figsize=(6, 6))

plot_mf_centers(muscle, ax=plt.gca())
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.axis("equal")
plt.grid(False)

plt.tight_layout()
plt.show()


# ============================================================
# Visualize Motor Unit Innervation Areas
# ============================================================
# Mostra a organização espacial das unidades motoras e suas áreas
# de inervação (território) — ou seja, quais fibras pertencem a
# qual unidade motora, calculado a partir dos limiares de
# recrutamento e da densidade de fibras.

plt.figure(figsize=(6, 6))

plot_innervation_areas_2d(muscle, ax=plt.gca())
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.axis("equal")
plt.grid(False)

plt.tight_layout()
plt.show()

print("\n[DONE] Modelo do músculo criado com sucesso!")