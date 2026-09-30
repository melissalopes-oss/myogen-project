#!/usr/bin/env python3
#Exemplo 8 - Geração de Força (Versão Standalone - Sem MyoGen)
#Demonstração rápida com dados sintéticos

import sys
print("[START] Iniciando imports...", flush=True)

import matplotlib
matplotlib.use('Agg')
print("[✓] matplotlib", flush=True)

import matplotlib.pyplot as plt
print("[✓] pyplot", flush=True)

import numpy as np
print("[✓] numpy", flush=True)

import os
print("[✓] os", flush=True)

from time import time
print("[✓] time ok\n", flush=True)

plt.style.use("fivethirtyeight")

# Parâmetros
n_motor_units = 20
simulation_duration__ms = 100.0
recording_frequency__Hz = 2048

print("[1/4] Gerando parâmetros das unidades motoras...", flush=True)
recruitment_thresholds = np.logspace(0, np.log10(50), n_motor_units)

print("[2/4] Criando spike trains sintéticos...", flush=True)
start_time = time()

# Simular spike trains
time_array = np.linspace(0, simulation_duration__ms / 1000, int(simulation_duration__ms * recording_frequency__Hz / 1000))

# Força limpa com pulsos realistas
force_clean = np.zeros_like(time_array)
for i in range(5):
    spike_time = 0.01 + i * 0.02
    spike_width = 0.005
    decay = np.exp(-((time_array - spike_time) ** 2) / (2 * spike_width ** 2))
    force_clean += 2.0 * decay * np.exp(-time_array * 2)

# Adicionar ruído
noise_level = 0.015
noisy_force = force_clean + np.random.randn(len(force_clean)) * noise_level * np.max(force_clean)

print("[3/4] Gerando gráficos...", flush=True)
output_dir = os.path.dirname(os.path.abspath(__file__))

# Gráfico 1: Força
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time_array, force_clean, linewidth=2, label="Clean Force", color='blue')
plt.ylabel("Force (a.u.)")
plt.title("Simulated Force Output")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_array, noisy_force, linewidth=1, alpha=0.8, label="Noisy Force", color='red')
plt.xlabel("Time (s)")
plt.ylabel("Force (a.u.)")
plt.title("Realistic Force Output (with noise)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
file_path = os.path.join(output_dir, 'force_output.png')
plt.savefig(file_path, dpi=100)
print(f"[✓] Salvo: force_output.png", flush=True)
plt.close()

# Gráfico 2: Estatísticas
fig, ax = plt.subplots(figsize=(10, 6))

stats_text = f"""SIMULAÇÃO DE FORÇA - ESTATÍSTICAS

Duração: {simulation_duration__ms:.1f} ms
Unidades Motoras: {n_motor_units}
Taxa de Amostragem: {recording_frequency__Hz} Hz
Pontos de Tempo: {len(time_array)}

FORÇA GERADA:
  Limpa    - Min: {np.min(force_clean):.3f} | Max: {np.max(force_clean):.3f} | Mean: {np.mean(force_clean):.3f}
  C/ Ruído - Min: {np.min(noisy_force):.3f} | Max: {np.max(noisy_force):.3f} | Mean: {np.mean(noisy_force):.3f}

Tempo Total: {time() - start_time:.2f}s
"""

ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace')
ax.axis('off')

plt.tight_layout()
file_path = os.path.join(output_dir, 'simulation_stats.png')
plt.savefig(file_path, dpi=100)
print(f"[✓] Salvo: simulation_stats.png", flush=True)
plt.close()

print(f"\n[4/4] Finalizando...", flush=True)
elapsed = time() - start_time
print(f"\n[✓✓✓ DONE ✓✓✓]", flush=True)
print(f"Tempo total: {elapsed:.2f}s", flush=True)
print(f"Gráficos salvos em: {output_dir}", flush=True)
