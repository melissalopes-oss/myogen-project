import json
import os
import warnings
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg' se tiver PyQt5 instalado
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from neo import Block, Segment, SpikeTrain
from neuron import h

from myogen import get_random_generator
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.helper import calculate_firing_rate_statistics
from myogen.utils.nmodl import load_nmodl_mechanisms

warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.style.use("fivethirtyeight")

DD_DRIVE_HZ = 100
N_MOTOR_UNITS = 800
N_DD_NEURONS = 400
DD_CONNECTIVITY = 0.3
SYNAPTIC_WEIGHT = 0.05
SIMULATION_TIME_MS = 10000.0
TIMESTEP_MS = 0.1
MAX_FORCE_N = 100.0

RESULTS_DIR = Path(__file__).parent / "results" / "watanabe_optimization"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

load_nmodl_mechanisms()
h.secondorder = 2 

recruitment_thresholds, _ = RecruitmentThresholds(N=N_MOTOR_UNITS, recruitment_range__ratio=100, deluca__slope=5, konstantin__max_threshold__ratio=1.0, mode="combined")
motor_neuron_pool = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds, config_file="alpha_mn_default.yaml")
descending_drive_pool = DescendingDrive__Pool(n=N_DD_NEURONS, timestep__ms=TIMESTEP_MS * pq.ms, process_type="poisson", poisson_batch_size=1)

network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
network.connect(source="DD", target="aMN", probability=DD_CONNECTIVITY, weight__uS=SYNAPTIC_WEIGHT * pq.uS)
network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
dd_netcons = network.get_netcons("cortical_input", "DD")

# Recorders
mn_spike_recorders = [h.Vector() for _ in motor_neuron_pool]
for cell, recorder in zip(motor_neuron_pool, mn_spike_recorders):
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50
    nc.record(recorder)

dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

# Simulação
time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
drive_signal = np.ones(time_points) * DD_DRIVE_HZ + np.clip(get_random_generator().normal(0, 1.0, size=time_points), 0, None)
h.load_file("stdrun.hoc")
h.dt = TIMESTEP_MS
h.tstop = SIMULATION_TIME_MS
h.finitialize()

step_counter = 0
while h.t < h.tstop:
    current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
    for dd_cell in descending_drive_pool:
        if dd_cell.integrate(current_drive):
            dd_spike_times[dd_cell.pool__ID].append(h.t)
            if h.t < h.tstop:
                dd_netcons[dd_cell.pool__ID].event(h.t + 1)
    h.fadvance()
    step_counter += 1

# Processamento dos Spikes
dt_s = h.dt / 1000.0
def create_spiketrains(recorders, name):
    return [SpikeTrain(r.as_numpy()/1000*pq.s, t_stop=SIMULATION_TIME_MS/1000*pq.s, sampling_period=dt_s*pq.s, name=f"{name}_{i}") for i, r in enumerate(recorders)]

mn_st = create_spiketrains(mn_spike_recorders, "MN")
dd_st = [SpikeTrain((np.array(times)*pq.ms).rescale(pq.s), t_stop=SIMULATION_TIME_MS/1000*pq.s, sampling_period=dt_s*pq.s, name=f"DD_{i}") for i, times in enumerate(dd_spike_times)]

# Plotagem
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 1. Raster DD
for i, st in enumerate(dd_st):
    if i % 5 == 0 and len(st) > 0:
        axes[0].scatter(st.rescale(pq.s).magnitude, [i] * len(st), s=0.5, alpha=0.6, color='red')
axes[0].set_ylabel("DD Neuron ID")
axes[0].set_title("Descending Drive Spike Trains")

# 2. Raster MN
for i, st in enumerate(mn_st):
    if i % 5 == 0 and len(st) > 0:
        axes[1].scatter(st.rescale(pq.s).magnitude, [i] * len(st), s=0.5, alpha=0.6)
axes[1].set_ylabel("Motor Unit ID")
axes[1].set_title("Motor Neuron Spike Trains")

# 3. Força (cálculo simplificado apenas para manter o plot funcional)
# (O cálculo completo da força foi omitido aqui para focar no seu pedido de plot)
axes[2].text(0.5, 0.5, "Plot de Força aqui", ha='center') 

# 4. Firing Rates
axes[3].set_title("Firing Rate Distribution")
axes[3].set_ylabel("Hz")

mn_spikes_total = sum(len(r.as_numpy()) for r in mn_spike_recorders)
dd_spikes_total = sum(len(t) for t in dd_spike_times)
print(f"Simulação concluída: {mn_spikes_total} spikes MN, {dd_spikes_total} spikes DD")
print(f"Salvando em: {RESULTS_DIR / 'final_plot.png'}")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_plot.png", dpi=150)
plt.show()