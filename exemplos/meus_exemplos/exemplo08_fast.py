
#Exemplo 8 do MyoGen - Force Generation (Versão Rápida)
#Versão simplificada para demonstração com dados sintéticos

import sys
print("[START] Iniciando imports...", flush=True)

import matplotlib
matplotlib.use('Agg')
print("[IMPORT] matplotlib ok", flush=True)

import matplotlib.pyplot as plt
print("[IMPORT] pyplot ok", flush=True)

import numpy as np
print("[IMPORT] numpy ok", flush=True)

import quantities as pq
print("[IMPORT] quantities ok", flush=True)

import os
print("[IMPORT] os ok", flush=True)

from time import time
print("[IMPORT] time ok", flush=True)

# Simular módulo myogen se não estiver disponível
print("[CHECK] Tentando importar MyoGen...", flush=True)
MYOGEN_AVAILABLE = False
try:
    print("[IMPORT] Carregando simulator...", flush=True)
    from myogen import simulator
    print("[IMPORT] Carregando ForceModel...", flush=True)
    from myogen.simulator.core.force.force_model import ForceModel
    print("[IMPORT] MyoGen importado com sucesso!", flush=True)
    MYOGEN_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] MyoGen não disponível: {e}", flush=True)
    MYOGEN_AVAILABLE = False

plt.style.use("fivethirtyeight")

# Define Parameters
n_motor_units = 20
recruitment_range = 50
recording_frequency__Hz = 2048 * pq.Hz
longest_duration_rise_time__ms = 90.0 * pq.ms
contraction_time_range = 3

# Simulação curta
simulation_duration__ms = 100.0  # 100 ms
timestep__ms = 0.05 * pq.ms
t_points = int(simulation_duration__ms / timestep__ms.magnitude)

print("[1/5] Inicializando simulação...")
start_time = time()

# Generate Recruitment Thresholds
print("[2/5] Gerando limiares de recrutamento...")
if MYOGEN_AVAILABLE:
    recruitment_thresholds, _ = simulator.RecruitmentThresholds(
        N=n_motor_units,
        recruitment_range__ratio=recruitment_range,
        mode="combined",
        deluca__slope=5,
    )
    
    # Create Force Model
    print("[3/5] Criando modelo de força...")
    force_model = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=recording_frequency__Hz,
        longest_duration_rise_time__ms=longest_duration_rise_time__ms,
        contraction_time_range_factor=contraction_time_range,
    )
    
    print(f"\tMotor units: {force_model._number_of_neurons}")
    print(f"\tRecruitment ratio: {force_model._recruitment_ratio:.1f}")
else:
    # Dados sintéticos para teste rápido
    recruitment_thresholds = np.logspace(0, np.log10(recruitment_range), n_motor_units)

# Generate synthetic spike trains and force output
print("[4/5] Gerando spike trains e força...")

# Simular spike trains (dados sintéticos)
time_array = np.linspace(0, simulation_duration__ms / 1000, int(simulation_duration__ms * 20.48))  # 2048 Hz

# Força sintética com padrão realista
force_clean = np.zeros_like(time_array)
for i in range(5):  # Alguns pulsos de força
    spike_time = 0.01 + i * 0.02
    spike_width = 0.005
    decay = np.exp(-((time_array - spike_time) ** 2) / (2 * spike_width ** 2))
    force_clean += 2.0 * decay * np.exp(-time_array * 2)

# Adicionar ruído
noise_level = 0.015
noisy_force = force_clean + np.random.randn(len(force_clean)) * noise_level * np.max(force_clean)

# Visualização
print("[5/5] Gerando gráficos...")
output_dir = os.path.dirname(os.path.abspath(__file__))

# Figura 1: Força limpa vs com ruído
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
plt.savefig(os.path.join(output_dir, 'force_output.png'), dpi=100)
print(f"[SAVED] force_output.png")
plt.close()

# Figura 2: Estatísticas
fig, ax = plt.subplots(figsize=(10, 6))

stats_text = f"""
SIMULAÇÃO DE FORÇA - ESTATÍSTICAS

Duração da simulação: {simulation_duration__ms:.1f} ms
Número de unidades motoras: {n_motor_units}
Taxa de amostragem: 2048 Hz
Timestep: 0.05 ms

FORÇA GERADA:
  Mínimo (limpo): {np.min(force_clean):.3f} a.u.
  Máximo (limpo): {np.max(force_clean):.3f} a.u.
  Média (limpo): {np.mean(force_clean):.3f} a.u.
  
  Mínimo (com ruído): {np.min(noisy_force):.3f} a.u.
  Máximo (com ruído): {np.max(noisy_force):.3f} a.u.
  Média (com ruído): {np.mean(noisy_force):.3f} a.u.

Tempo de execução: {time() - start_time:.2f}s
"""

ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace')
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'simulation_stats.png'), dpi=100)
print(f"[SAVED] simulation_stats.png")
plt.close()

print(f"\n[✓ DONE] Simulação completada em {time() - start_time:.2f}s")
print(f"[✓] Gráficos salvos em: {output_dir}")
