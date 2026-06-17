import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import quantities as pq
from pathlib import Path
from neuron import h
from neo import SpikeTrain

from myogen import get_random_generator
from myogen.simulator import RecruitmentThresholds
from myogen.simulator import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

# ── Parâmetros ──────────────────────────────────────────────
N_MOTOR_UNITS      = 800
N_DD_NEURONS       = 400
DD_DRIVE_HZ        = 100
DD_CONNECTIVITY    = 0.8
SYNAPTIC_WEIGHT    = 0.5      # uS
SIMULATION_TIME_MS = 10000.0
TIMESTEP_MS        = 0.1

RESULTS_DIR = Path(__file__).parent / "results" / "watanabe_optimization"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ── Setup ────────────────────────────────────────────────────
load_nmodl_mechanisms()
h.secondorder = 2

recruitment_thresholds, _ = RecruitmentThresholds(
    N=N_MOTOR_UNITS, recruitment_range__ratio=100,
    deluca__slope=5, konstantin__max_threshold__ratio=1.0, mode="combined"
)
motor_neuron_pool    = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds)
descending_drive_pool = DescendingDrive__Pool(
    n=N_DD_NEURONS, timestep__ms=TIMESTEP_MS * pq.ms,
    process_type="poisson", poisson_batch_size=1
)

network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
network.connect(source="DD", target="aMN", probability=DD_CONNECTIVITY, weight__uS=SYNAPTIC_WEIGHT * pq.uS)
network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
dd_netcons = network.get_netcons("cortical_input", "DD")

# ── Recorders ────────────────────────────────────────────────
mn_spike_recorders = [h.Vector() for _ in motor_neuron_pool]
for cell, recorder in zip(motor_neuron_pool, mn_spike_recorders):
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50
    nc.record(recorder)
dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

# ── Simulação ────────────────────────────────────────────────
time_points  = int(SIMULATION_TIME_MS / TIMESTEP_MS)
drive_signal = (np.ones(time_points) * DD_DRIVE_HZ
                + np.clip(get_random_generator().normal(0, 1.0, size=time_points), 0, None))

h.load_file("stdrun.hoc")
h.dt    = TIMESTEP_MS
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

mn_spikes_total = sum(len(r.as_numpy()) for r in mn_spike_recorders)
dd_spikes_total = sum(len(t) for t in dd_spike_times)
print(f"Simulação concluída: {mn_spikes_total} spikes MN, {dd_spikes_total} spikes DD")

# ── Spike trains ─────────────────────────────────────────────
dt_s     = h.dt / 1000.0
t_stop_s = SIMULATION_TIME_MS / 1000.0 * pq.s

mn_st = [
    SpikeTrain(r.as_numpy() / 1000 * pq.s, t_stop=t_stop_s,
               sampling_period=dt_s * pq.s, name=f"MN_{i}")
    for i, r in enumerate(mn_spike_recorders)
]
dd_st = [
    SpikeTrain((np.array(times) * pq.ms).rescale(pq.s), t_stop=t_stop_s,
               sampling_period=dt_s * pq.s, name=f"DD_{i}")
    for i, times in enumerate(dd_spike_times)
]

# ── Plot ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Raster DD
dd_times_list = [st.rescale(pq.s).magnitude for st in dd_st if len(st) > 0]
dd_ids = [i for i, st in enumerate(dd_st) if len(st) > 0]
if dd_times_list:
    axes[0].eventplot(dd_times_list, lineoffsets=dd_ids, linelengths=0.8,
                      linewidths=1.0, color='red', alpha=0.6)
axes[0].set_ylabel("DD Neuron ID")
axes[0].set_title("Descending Drive Spike Trains")
axes[0].set_ylim(-0.5, N_DD_NEURONS - 0.5)

# Raster MN
mn_times_list = [st.rescale(pq.s).magnitude for st in mn_st if len(st) > 0]
mn_ids = [i for i, st in enumerate(mn_st) if len(st) > 0]
if mn_times_list:
    axes[1].eventplot(mn_times_list, lineoffsets=mn_ids, linelengths=0.8,
                      linewidths=1.0, alpha=0.7)
axes[1].set_ylabel("Motor Unit ID")
axes[1].set_title("Motor Neuron Spike Trains")
axes[1].set_ylim(-0.5, N_MOTOR_UNITS - 0.5)

# Firing rate médio por janela de 100ms
bin_size_s = 0.1
bins       = np.arange(0, t_stop_s.magnitude, bin_size_s)
mean_rates = []
for b in bins[:-1]:
    rates_in_bin = [
        np.sum((st.magnitude >= b) & (st.magnitude < b + bin_size_s)) / bin_size_s
        for st in mn_st if len(st) > 0
    ]
    mean_rates.append(np.mean(rates_in_bin) if rates_in_bin else 0)

axes[2].plot(bins[:-1], mean_rates, color='green', linewidth=1.5)
axes[2].fill_between(bins[:-1], mean_rates, alpha=0.2, color='green')
axes[2].set_xlabel("Tempo (s)")
axes[2].set_ylabel("Taxa média (Hz)")
axes[2].set_title("Firing Rate Médio dos Motor Neurons")

plt.tight_layout()
out_path = RESULTS_DIR / "final_plot.png"
plt.savefig(out_path, dpi=150)
print(f"Gráfico salvo em: {out_path}")