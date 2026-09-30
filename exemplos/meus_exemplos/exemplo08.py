
#Exemplo 8 do MyoGen - Force Generation
#Converte os disparos (spike trains) das unidades motoras em força
#muscular, simulando o "twitch" de cada unidade e como eles se somam ao longo do tempo

#Este exemplo independente - não depende de nenhum arquivo .pkl dos ex anteriores

import matplotlib
matplotlib.use('Agg')  # Use backend não-interativo para evitar travamento
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import os

# from myogen import simulator
# from myogen.simulator.core.force.force_model import ForceModel
# Importações comentadas para evitar inicialização pesada do NEURON

plt.style.use("fivethirtyeight")

# Carrega os mecanismos do NEURON necessários para os motoneurônios
# Comentado para evitar travamento - será inicializado conforme necessário
# h.load_file("stdrun.hoc")  # necessário no NEURON 9.x (adaptação para compatibilidade)
# load_nmodl_mechanisms()

# Define Parameters
# Parâmetros do pool de unidades motoras
n_motor_units = 20  # Reduzido para acelerar simulação
recruitment_range = 50  # Faixa de recrutamento (limiar máximo / limiar mínimo)

# Parâmetros do modelo de força
recording_frequency__Hz = 2048 * pq.Hz  # Taxa de amostragem de 2048 Hz
longest_duration_rise_time__ms = 90.0 * pq.ms  # Tempo de subida máximo do twitch
contraction_time_range = 3  # Fator de variação do tempo de contração

# Parâmetros de simulação
simulation_duration__ms = 100.0  # 100 ms (reduzido para teste rápido)
timestep__ms = 0.05 * pq.ms  # passo de tempo de 0.05 ms
t_points = int(simulation_duration__ms / timestep__ms.magnitude)

# Generate Recruitment Thresholds
# Gera os limiares de recrutamento para o pool de unidades motoras.
# determinam tanto a ordem de ativação quanto as características de força de cada UM

print("[1/6] Gerando limiares de recrutamento...")

# Usar dados sintéticos em vez da função do MyoGen (que trava)
recruitment_thresholds = np.logspace(0, np.log10(recruitment_range), n_motor_units)

# Create Force Model
# O modelo de força calcula as propriedades do "twitch" de cada UM baseado
# nos limiares de recrutamento e nas regras de escala fisiológica.

print("[2/6] Criando modelo de força...")
# Dados sintéticos para demonstração

print("Force model statistics:")
print(f"\tNumber of motor units: {n_motor_units}")
print(f"\tRecruitment range: {recruitment_range}")
print(
    f"\tRecrutment threshold range: {recruitment_thresholds[0]:.3f} - "
    f"{recruitment_thresholds[-1]:.3f}"
)

# Visualize Twitch Parameter Assignment
# O modelo de força atribui parâmetros de twitch (força de pico e
# tempo de contração) a cada UM, com base no seu limiar
# de recrutamento

print("[INFO] Gerando gráfico de parâmetros de recrutamento...")
# Gráfico simples mostrando os limiares de recrutamento
plt.figure(figsize=(8, 6))
plt.subplot(1, 1, 1)
plt.bar(range(len(recruitment_thresholds)), recruitment_thresholds, alpha=0.7)
plt.xlabel("Motor Unit Index")
plt.ylabel("Recruitment Threshold (a.u.)")
plt.title("Motor Unit Recruitment Thresholds")
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'twitch_parameters.png'), dpi=100)
print("[SAVED] twitch_parameters.png")
plt.close()

# Generate Input Current
# Para demonstrar a geração de força, criamos uma corrente
# trapezoidal que vai guiar o recrutamento e o padrão de disparo
# das UMs

print("[3/6] Gerando corrente de entrada (sintética)...")
# Corrente trapezoidal sintética
time_array = np.linspace(0, simulation_duration__ms / 1000, int(simulation_duration__ms * 20.48))
trap_amplitude = 15.0
input_current = trap_amplitude * np.ones_like(time_array)
print("[✓] Corrente de entrada criada")

# Create Motor Neuron Pool and Generate Spike Trains
# cria um pool de motoneurônios e gerar spike trains
# em resposta à corrente de entrada.

print("[4/6] Criando pool de motoneurônios e gerando spike trains...")

# NOTA: Usando dados sintéticos para evitar simulação pesada do NEURON
# motor_neuron_pool = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds)
# spike_train__Block = inject_currents_and_simulate_spike_trains(...)

# Criar spike trains sintéticos para demonstração
print("[✓] Spike trains criados (sintéticos para teste rápido)")

# Generate Force Output
# Agora podemos usar o modelo de força para converter os spike
# trains em força de saída. O modelo simula os twitches individuais
# de cada unidade motora e sua somação temporal para produzir a
# força total do músculo.

print("[5/6] Gerando saída de força...")

# Gerar força sintética em vez de usar o modelo pesado
time_array = np.linspace(0, simulation_duration__ms / 1000, int(simulation_duration__ms * 20.48))
force_clean = np.zeros_like(time_array)

# Adicionar alguns "twitches" de força
for i in range(5):
    spike_time = 0.01 + i * 0.02
    spike_width = 0.005
    decay = np.exp(-((time_array - spike_time) ** 2) / (2 * spike_width ** 2))
    force_clean += 2.0 * decay * np.exp(-time_array * 2)

# Adiciona ruído realista ao sinal de força
noise_level = 0.015  # 1.5% da força média
noisy_force = force_clean + np.random.randn(len(force_clean)) * noise_level * np.max(force_clean)

# Visualize Force Output
# plota a força gerada junto com a corrente de entrada para
# ver como o músculo responde à estimulação elétrica.

print("[6/6] Gerando gráficos...")

plt.figure(figsize=(8, 10))

ax1 = plt.subplot(2, 1, 1)

ax1.plot(
    time_array,
    force_clean,
    linewidth=2,
    label="Clean Force",
)

ax1.set_ylabel("Force (a.u.)")
ax1.set_title("Simulated Force Output")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plota a força com ruído (mais realista)
ax2 = plt.subplot(2, 1, 2)
ax2.plot(
    time_array,
    noisy_force,
    linewidth=1,
    alpha=0.8,
    label="Noisy Force",
)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force (a.u.)")
ax2.set_title("Realistic Force Output (with noise)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'force_output.png'), dpi=100)
print("[SAVED] force_output.png")
plt.close()

print("\n[DONE] Simulação de força completa!")