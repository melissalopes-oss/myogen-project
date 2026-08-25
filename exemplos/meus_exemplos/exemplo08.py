
#Exemplo 8 do MyoGen - Force Generation
#Converte os disparos (spike trains) das unidades motoras em força
#muscular, simulando o "twitch" de cada unidade e como eles se somam ao longo do tempo

#Este exemplo independente - não depende de nenhum arquivo .pkl dos ex anteriores

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from neuron import h

from myogen import simulator
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron.populations import AlphaMN__Pool
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.neuron.inject_currents_into_populations import (
    inject_currents_and_simulate_spike_trains,
)
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.plotting.force import plot_twitch_parameter_assignment, plot_twitches

plt.style.use("fivethirtyeight")

# Carrega os mecanismos do NEURON necessários para os motoneurônios
h.load_file("stdrun.hoc")  # necessário no NEURON 9.x (adaptação para compatibilidade)
load_nmodl_mechanisms()

# Define Parameters
# Parâmetros do pool de unidades motoras
n_motor_units = 50
recruitment_range = 50  # Faixa de recrutamento (limiar máximo / limiar mínimo)

# Parâmetros do modelo de força
recording_frequency__Hz = 2048 * pq.Hz  # Taxa de amostragem de 2048 Hz
longest_duration_rise_time__ms = 90.0 * pq.ms  # Tempo de subida máximo do twitch
contraction_time_range = 3  # Fator de variação do tempo de contração

# Parâmetros de simulação
simulation_duration__ms = 10000.0  # 10 segundos
timestep__ms = 0.05 * pq.ms  # passo de tempo de 0.05 ms
t_points = int(simulation_duration__ms / timestep__ms.magnitude)

# Generate Recruitment Thresholds
# Gera os limiares de recrutamento para o pool de unidades motoras.
# determinam tanto a ordem de ativação quanto as características de força de cada UM

recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    mode="combined",
    deluca__slope=5,
)

# Create Force Model
# O modelo de força calcula as propriedades do "twitch" de cada UM baseado
# nos limiares de recrutamento e nas regras de escala fisiológica.

force_model = ForceModel(
    recruitment_thresholds=recruitment_thresholds,
    recording_frequency__Hz=recording_frequency__Hz,
    longest_duration_rise_time__ms=longest_duration_rise_time__ms,
    contraction_time_range_factor=contraction_time_range,
)

print("Force model statistics:")
print(f"\tNumber of motor units: {force_model._number_of_neurons}")
print(f"\tRecruitment ratio: {force_model._recruitment_ratio:.1f}")
print(
    f"\tPeak force range: {force_model.peak_twitch_forces__unitless[0]:.3f} - "
    f"{force_model.peak_twitch_forces__unitless[-1]:.3f}"
)
print(
    f"\tContraction time range: {force_model.contraction_times__samples[0]:.1f} - "
    f"{force_model.contraction_times__samples[-1]:.1f} samples"
)

# Visualize Twitch Parameter Assignment
# O modelo de força atribui parâmetros de twitch (força de pico e
# tempo de contração) a cada UM, com base no seu limiar
# de recrutamento

plt.figure(figsize=(8, 12))

ax1 = plt.subplot(2, 1, 1)
plot_twitch_parameter_assignment(
    force_model, ax1, [10, 20, 40], flip_x=True, apply_default_formatting=True
)
ax1.set_title("Twitch Parameter Assignment")

ax2 = plt.subplot(2, 1, 2)
plot_twitches(force_model, ax2, [10, 20, 40], apply_default_formatting=True)
ax2.set_title("Motor Unit Twitches")

plt.tight_layout()
plt.show()

# Generate Input Current
# Para demonstrar a geração de força, criamos uma corrente
# trapezoidal que vai guiar o recrutamento e o padrão de disparo
# das UMs

trap_amplitude = 15.0 * pq.nA  # Amplitude de pico
trap_rise_time = 5000.0 * pq.ms  # Duração da subida (ms)
trap_plateau_time = 8000.0 * pq.ms  # Duração do platô (ms)
trap_fall_time = 3000.0 * pq.ms  # Duração da descida (ms)
trap_offset = 5.0 * pq.nA  # Corrente de base
trap_delay = 0.0 * pq.ms  # Atraso inicial (ms)

input_current__AnalogSignal = create_trapezoid_current(
    n_pools=1,
    t_points=t_points,
    timestep__ms=timestep__ms,
    amplitudes__nA=[trap_amplitude],
    rise_times__ms=[trap_rise_time],
    plateau_times__ms=[trap_plateau_time],
    fall_times__ms=[trap_fall_time],
    offsets__nA=[trap_offset],
    delays__ms=[trap_delay],
)

# Create Motor Neuron Pool and Generate Spike Trains
# cria um pool de motoneurônios e gerar spike trains
# em resposta à corrente de entrada.

motor_neuron_pool = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds)

# Gera spike trains usando a função utilitária
spike_train__Block = inject_currents_and_simulate_spike_trains(
    populations=[motor_neuron_pool],
    input_current__AnalogSignal=input_current__AnalogSignal,
    spike_detection_thresholds__mV=50 * pq.mV,
)

# Generate Force Output
# Agora podemos usar o modelo de força para converter os spike
# trains em força de saída. O modelo simula os twitches individuais
# de cada unidade motora e sua somação temporal para produzir a
# força total do músculo.

force_output = force_model.generate_force(spike_train__Block=spike_train__Block)

# Adiciona ruído realista ao sinal de força
noise_level = 0.015  # 1.5% da força média
noisy_force = force_output.magnitude[:, 0] + np.random.randn(
    len(force_output.magnitude[:, 0])
) * noise_level * np.mean(force_output.magnitude[:, 0])

# Visualize Force Output
# plota a força gerada junto com a corrente de entrada para
# ver como o músculo responde à estimulação elétrica.

plt.figure(figsize=(8, 10))

ax1 = plt.subplot(2, 1, 1)

ax1.plot(
    force_output.times.rescale("s"),
    force_output[:, 0],
    linewidth=2,
    label="Clean Force",
)

ax1.set_ylabel("Force (a.u.)")
ax1.set_title("Simulated Force Output")
ax1.grid(True, alpha=0.3)

# Plota a força com ruído (mais realista)
ax2 = plt.subplot(2, 1, 2)
ax2.plot(
    force_output.times.rescale("s"),
    noisy_force,
    linewidth=1,
    alpha=0.8,
    label="Noisy Force",
)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force (a.u.)")
ax2.set_title("Realistic Force Output (with noise)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n[DONE] Simulação de força completa!")