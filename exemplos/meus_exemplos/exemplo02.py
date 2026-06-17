
# EXEMPLO 02 - GERAÇÃO DE TRENS DE DISPARO COM INJEÇÃO DE CORRENTE

from pathlib import Path
import joblib
import neuron
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt
from neo import Block, Segment, SpikeTrain
from neuron import h
from scipy.ndimage import gaussian_filter1d

from myogen import get_random_generator
from myogen.simulator.neuron.populations import AlphaMN__Pool
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.neuron.inject_currents_into_populations import (
    inject_currents_and_simulate_spike_trains,
    inject_currents_into_populations,
)
from myogen.utils.nmodl import load_nmodl_mechanisms

plt.style.use("fivethirtyeight")
plt.style.use("fivethirtyeight")

h.load_file("stdrun.hoc")
# Carrega as funções de controle de simulação do NEURON (continuerun, etc.)


def mean_firing_rate(spiketrain):
    """Taxa média de disparo de um neo.SpikeTrain."""
    return (len(spiketrain) / (spiketrain.t_stop - spiketrain.t_start)).rescale(pq.Hz)


def rasterplot_rates(spiketrains, filter_function=None):
    """Desenha um raster plot de disparos com histogramas nas margens."""
    if filter_function is not None:
        spiketrains = [st for st in spiketrains if filter_function(st)]

    fig = plt.figure()
    ax = fig.add_axes((0.10, 0.10, 0.62, 0.62))
    axhistx = fig.add_axes((0.10, 0.74, 0.62, 0.16), sharex=ax)
    axhisty = fig.add_axes((0.74, 0.10, 0.16, 0.62), sharey=ax)

    ax.eventplot(
        [st.rescale(pq.s).magnitude for st in spiketrains],
        lineoffsets=np.arange(len(spiketrains)),
        colors="black",
        linelengths=0.8,
        linewidths=0.7,
    )
    rates = [float(mean_firing_rate(st).magnitude) for st in spiketrains]
    axhisty.barh(np.arange(len(spiketrains)), rates, height=0.85, color="C0")
    ax.set_ylim(-1, max(len(spiketrains), 1))
    return ax, axhistx, axhisty


# CRIANDO O POOL DE MOTONEURÔNIOS

load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

n_pools = 1
motor_neuron_pools = [
    AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds) for _ in range(n_pools)
]

# CRIANDO A CORRENTE DE ENTRADA

timestep = 0.05 * pq.ms
simulation_time = 1000 * pq.ms

rise_time_ms = list(get_random_generator().uniform(100, 500, size=n_pools)) * pq.ms
plateau_time_ms = list(get_random_generator().uniform(1000, 2000, size=n_pools)) * pq.ms
fall_time_ms = list(get_random_generator().uniform(1000, 2000, size=n_pools)) * pq.ms

input_current__AnalogSignal = create_trapezoid_current(
    n_pools,
    int(simulation_time / timestep),
    timestep,
    amplitudes__nA=[15.0 * pq.nA] * n_pools,
    rise_times__ms=rise_time_ms,
    plateau_times__ms=plateau_time_ms,
    fall_times__ms=fall_time_ms,
    delays__ms=500.0 * pq.ms,
)

print(
    f"Input current signal shape: {input_current__AnalogSignal.shape}\n"
    f"Class: {input_current__AnalogSignal.__class__}"
)

joblib.dump(input_current__AnalogSignal, save_path / "input_current__AnalogSignal.pkl")

# SIMULAÇÃO MANUAL - PASSO A PASSO

# Passo 1: injeta a corrente em cada neurônio do pool
inject_currents_into_populations(motor_neuron_pools, input_current__AnalogSignal)

# Passo 2: configura a detecção e gravação dos disparos
spike_detection_threshold__mV = 50.0 * pq.mV
simulation_time__ms = input_current__AnalogSignal.t_stop.rescale(pq.ms)

spike_recorders = []

for pool_idx, pool in enumerate(motor_neuron_pools):
    pool_spike_recorders = []

    for cell in pool:
        spike_recorder = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = spike_detection_threshold__mV
        nc.record(spike_recorder)
        pool_spike_recorders.append(spike_recorder)

    spike_recorders.append(pool_spike_recorders)

# Passo 3: inicializa as tensões de membrana e roda a simulação
for pool in motor_neuron_pools:
    for section, voltage in zip(*pool.get_initialization_data()):
        section.v = voltage

h.finitialize()
h.continuerun(float(simulation_time__ms.magnitude))

# Passo 4: converte os dados gravados para o formato neo.Block
spike_train__Block_manual = Block(name="Manual Simulation Results")

for pool_idx, pool_spike_recorders in enumerate(spike_recorders):
    segment = Segment(name=f"Pool {pool_idx}")
    segment.spiketrains = []

    for neuron_idx, spike_recorder in enumerate(pool_spike_recorders):
        spike_times = (spike_recorder.as_numpy() * pq.ms).rescale(pq.s)

        spiketrain = SpikeTrain(
            spike_times,
            t_stop=simulation_time__ms.rescale(pq.s),
            sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
            sampling_period=(h.dt * pq.ms).rescale(pq.s),
            name=str(neuron_idx),
            description=f"Pool {pool_idx}, Neuron {neuron_idx}",
        )
        segment.spiketrains.append(spiketrain)

    spike_train__Block_manual.segments.append(segment)

joblib.dump(spike_train__Block_manual, save_path / "spike_train__Block_manual.pkl")

# SIMULAÇÃO COM FUNÇÃO DE UTILIDADE (MAIS SIMPLES)

spike_train__Block = inject_currents_and_simulate_spike_trains(
    populations=motor_neuron_pools,
    input_current__AnalogSignal=input_current__AnalogSignal,
    spike_detection_thresholds__mV=50 * pq.mV,
)

joblib.dump(spike_train__Block, save_path / "spike_train__Block_utility.pkl")

print("\nComparison of results:")
print(f"Manual approach: {len(spike_train__Block_manual.segments)} segments")
print(f"Utility approach: {len(spike_train__Block.segments)} segments")

for i, (manual_seg, utility_seg) in enumerate(
    zip(spike_train__Block_manual.segments, spike_train__Block.segments)
):
    manual_spikes = sum(len(st) for st in manual_seg.spiketrains)
    utility_spikes = sum(len(st) for st in utility_seg.spiketrains)
    print(f"Pool {i}: Manual={manual_spikes} spikes, Utility={utility_spikes} spikes")

# ESTATÍSTICAS DE TAXA DE DISPARO

firing_rates = [
    np.array(
        [
            mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
            for st__s in spike_train__segment.spiketrains
            if len(st__s) > 1
        ]
    )
    for spike_train__segment in spike_train__Block.segments
]

print("Firing rate statistics:")
for pool_idx, firing_rates_per_pool in enumerate(firing_rates):
    active_neurons = np.sum(firing_rates_per_pool > 0)
    if len(firing_rates_per_pool) > 0 and np.sum(firing_rates_per_pool > 0) > 0:
        mean_rate = np.mean(firing_rates_per_pool[firing_rates_per_pool > 0])
        max_rate = np.max(firing_rates_per_pool)
    else:
        mean_rate = 0.0
        max_rate = 0.0

    print(
        f"  Pool {pool_idx + 1}: {active_neurons}/{len(recruitment_thresholds)} active neurons, "
        f"mean rate: {mean_rate:.1f} Hz, max rate: {max_rate:.1f} Hz"
    )

# VISUALIZAÇÃO DOS TRENS DE DISPARO

spike_train_list = list(spike_train__Block.segments[0].spiketrains)
active_spiketrains = [st for st in spike_train_list if len(st) > 0]

ax, axhistx, axhisty = rasterplot_rates(spike_train_list, filter_function=lambda st: len(st) > 0)
ax.plot(
    input_current__AnalogSignal.times,
    input_current__AnalogSignal.magnitude.T[0]
    / input_current__AnalogSignal.magnitude.T[0].max()
    * len(active_spiketrains),
    color="black",
)

axhisty.set_xlabel("FR (pps)")
axhistx.clear()

if len(active_spiketrains) > 0:
    sampling_period_s = (h.dt * pq.ms).rescale(pq.s).magnitude
    t_start = min(st.t_start for st in active_spiketrains).rescale(pq.s).magnitude
    t_stop = max(st.t_stop for st in active_spiketrains).rescale(pq.s).magnitude
    n_bins = int(round((t_stop - t_start) / sampling_period_s))
    edges = t_start + np.arange(n_bins + 1) * sampling_period_s
    all_spikes = np.concatenate([st.rescale(pq.s).magnitude for st in active_spiketrains])
    all_spikes = all_spikes[(all_spikes >= edges[0]) & (all_spikes < edges[-1])]
    counts, _ = np.histogram(all_spikes, bins=edges)
    rate_hz = counts / sampling_period_s / len(active_spiketrains)
    rate_hz = gaussian_filter1d(rate_hz, sigma=(15e-3) / sampling_period_s, mode="constant")

    axhistx.plot(edges[:-1] + sampling_period_s / 2, rate_hz, linewidth=2)
    axhistx.set_ylabel("FR (pps)")
    axhistx.set_xlim(ax.get_xlim())

ax.set_ylabel("Neuron Index (#)")
ax.set_xlabel("Time (s)")

sns.despine(ax=ax)

fig = plt.gcf()
fig.set_size_inches(12, 6)

gap = 0.025
bottom_margin = 0.03

ax_pos = ax.get_position()
axhistx_pos = axhistx.get_position()
axhisty_pos = axhisty.get_position()

ax.set_position([ax_pos.x0, ax_pos.y0 + bottom_margin, ax_pos.width, ax_pos.height])
axhistx.set_position(
    [axhistx_pos.x0, axhistx_pos.y0 + gap + bottom_margin, axhistx_pos.width, axhistx_pos.height]
)
axhisty.set_position(
    [axhisty_pos.x0 + gap, axhisty_pos.y0 + bottom_margin, axhisty_pos.width, axhisty_pos.height]
)

plt.show()