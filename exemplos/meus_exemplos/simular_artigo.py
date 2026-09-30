import sys
import yaml
import itertools
from pathlib import Path

import joblib
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from myogen import get_random_generator, simulator
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.types import pps

plt.style.use("fivethirtyeight")


def mean_firing_rate(spiketrain):
    return (len(spiketrain) / (spiketrain.t_stop - spiketrain.t_start)).rescale(pq.Hz)


def population_psth(spiketrains, bin_size):
    t_start = min(st.t_start for st in spiketrains).rescale(pq.s).magnitude
    t_stop = max(st.t_stop for st in spiketrains).rescale(pq.s).magnitude
    bs = (bin_size).rescale(pq.s).magnitude
    n_bins = int((t_stop - t_start) / bs)
    edges = t_start + np.arange(n_bins + 1) * bs
    spikes = np.concatenate([st.rescale(pq.s).magnitude for st in spiketrains])
    spikes = spikes[(spikes >= edges[0]) & (spikes < edges[-1])]
    counts, _ = np.histogram(spikes, bins=edges)
    return counts, edges[:-1]


# ============================================================
# --- NOVO: Função para carregar o YAML igual ao seu configurar_simulacao.py ---
# ============================================================
def carregar_parametros(caminho_yaml: str) -> dict:
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rodar_simulacao(caminho_yaml: str):
    parametros = carregar_parametros(caminho_yaml)
    condicao_nome = parametros["condicao"]
    print(f"\n{'='*60}")
    print(f"Iniciando Simulação: {condicao_nome}")
    print(f"{'='*60}\n")

    load_nmodl_mechanisms()
    save_path = Path("./results")
    save_path.mkdir(exist_ok=True)

    # ============================================================
    # --- NOVO: Criação das populações baseadas no YAML do artigo ---
    # ============================================================
    print("1. Configurando populações com base no artigo...")
    
    recruitment_thresholds, _ = simulator.RecruitmentThresholds(
        N=parametros["numero_motoneuronios"],
        recruitment_range__ratio=parametros["razao_forca"],
        mode="combined",
        deluca__slope=5,
    )
    fator_forca = parametros["forca_minima_N"]/recruitment_thresholds[0]
    recruitment_thresholds = recruitment_thresholds * fator_forca
    print(recruitment_thresholds[0])

    motor_neuron_pool = AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        axon_velocities=(
            parametros["velocidade_conducao_minima_m_s"],
            parametros["velocidade_conducao_maxima_m_s"],
        )
    )

    timestep = 0.1 * pq.ms
    h.secondorder = 2  

    descending_drive_pool = DescendingDrive__Pool(
        n=parametros["numero_neuronios_corticoespinhais"], 
        process_type="poisson", 
        timestep__ms=timestep, 
        poisson_batch_size=parametros["numero_neuronios_corticoespinhais"]
    )

    # ============================================================
    # Generate SENOIDAL Drive Pattern (Baseado no ex. 3 modificado)
    # ============================================================
    print("2. Gerando estímulo descendente senoidal...")
    simulation_time = 5000 * pq.ms  # Reduzi para 5s para rodar mais rápido nos testes
    time_points = int(simulation_time / timestep)

    dd_offset__pps = 32.5 * pps
    dd_amplitude__pps = 32.5 * pps
    frequency_hz = 1.0

    time_array = np.linspace(0, simulation_time.magnitude, time_points) * pq.ms
    time_s = time_array.rescale(pq.s).magnitude

    offset_value = dd_offset__pps.magnitude.item()
    amplitude_value = dd_amplitude__pps.magnitude.item()
    sine_drive_raw = offset_value + amplitude_value * np.sin(2 * np.pi * frequency_hz * time_s)
    
    sine_drive_raw = np.clip(sine_drive_raw, 0, None)
    noise_raw = np.clip(get_random_generator().normal(0, 1.0, size=time_points), 0, None)
    sine_drive_raw = sine_drive_raw + noise_raw
    sine_drive = sine_drive_raw * pps

    sine_drive_signal = AnalogSignal(
        signal=sine_drive, sampling_period=timestep.rescale(pq.s)
    )

    # ============================================================
    # Create Network and Connections 
    # ============================================================
    network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
    network.connect(source="DD", target="aMN", probability=0.5, weight__uS=0.15 * pq.uS)
    network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
    dd_netcons = network.get_netcons("cortical_input", "DD")

    # ============================================================
    # Setup Spike Recording 
    # ============================================================
    dd_spike_times = [[] for _ in range(len(descending_drive_pool))]
    mn_spike_recorders = []
    
    for cell in motor_neuron_pool:
        spike_recorder = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(spike_recorder)
        mn_spike_recorders.append(spike_recorder)

    # ============================================================
    # Run Simulation 
    # ============================================================
    print("3. Executando simulação temporal no NEURON...")
    h.load_file("stdrun.hoc")
    
    # --- CORREÇÃO: Converter Quantity para float puro para o NEURON não travar ---
    h.dt = float(timestep.rescale(pq.ms).magnitude)
    h.tstop = float(simulation_time.rescale(pq.ms).magnitude)

    for section, voltage in itertools.chain.from_iterable(
        zip(*pool.get_initialization_data()) for pool in [motor_neuron_pool, descending_drive_pool]
    ):
        section.v = voltage

    h.finitialize()
    total_steps = int(simulation_time / timestep)
    step_counter = 0

    with tqdm(
        total=float(simulation_time),
        desc=f"Running {condicao_nome}",
        unit="ms",
        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} ms [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        while h.t < h.tstop:
            # Pega o valor atual da senoide
            current_drive_qty = sine_drive_signal[min(step_counter, len(sine_drive_signal) - 1)]
            
            # --- CORREÇÃO: Extrair o número puro (float) do Quantity para o integrate() ---
            current_drive_val = float(current_drive_qty.magnitude)

            for dd_cell in descending_drive_pool:
                # Usa o número puro aqui, corrigindo o erro
                if dd_cell.integrate(current_drive_val):
                    dd_spike_times[dd_cell.pool__ID].append(h.t)
                    spike_time = h.t + 1
                    if spike_time < h.tstop:
                        dd_netcons[dd_cell.pool__ID].event(spike_time)

            h.fadvance()
            step_counter += 1
            pbar.update(float(timestep))

    # ============================================================
    # Convert Spike Data to Neo Format 
    # ============================================================
    spike_train_block = Block(name=f"Sinusoidal DD Spike Trains - {condicao_nome}")

    dd_segment = Segment(name="Descending Drive")
    dd_segment.spiketrains = [
        SpikeTrain(
            (np.array(spike_times) * pq.ms).rescale(pq.s),
            t_stop=simulation_time.rescale(pq.s),
            sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
            sampling_period=h.dt * pq.ms,
            name=f"DD_{i}",
        )
        for i, spike_times in enumerate(dd_spike_times)
    ]

    mn_segment = Segment(name="Motor Neurons")
    mn_segment.spiketrains = [
        SpikeTrain(
            (np.array(recorder.as_numpy()) * pq.ms).rescale(pq.s),
            t_stop=simulation_time.rescale(pq.s),
            sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
            sampling_period=h.dt * pq.ms,
            name=f"MN_{i}",
        )
        for i, recorder in enumerate(mn_spike_recorders)
    ]
    spike_train_block.segments.append(mn_segment)
    
    nome_arquivo_pkl = f"sine_dd_spike_trains_{condicao_nome.lower()}.pkl"
    joblib.dump(spike_train_block, save_path / nome_arquivo_pkl)

    # ============================================================
    # Advanced Visualization (Igual ao original, adaptado o Título)
    # ============================================================
    print("\n4. Gerando gráficos...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Simulação Artigo: Condição {condicao_nome}", fontsize=16, fontweight='bold')

    axes[0].plot(time_s, sine_drive_signal, "b-", linewidth=2, label="DD Input (senoide)")
    axes[0].axhline(dd_offset__pps.magnitude.item(), color="r", linestyle="--", alpha=0.7, label="Offset (média)")
    axes[0].set_ylabel("Drive (Hz)")
    axes[0].set_title(f"Descending Drive Senoidal ({frequency_hz} Hz)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    dd_colors = plt.get_cmap("Blues")(np.linspace(0.3, 0.8, len(dd_segment.spiketrains)))
    for i, (spiketrain, color) in enumerate(zip(dd_segment.spiketrains, dd_colors)):
        if len(spiketrain) > 0:
            axes[1].scatter(spiketrain.magnitude, [i] * len(spiketrain), c=[color], s=0.8, alpha=0.8)

    axes[1].set_ylabel("DD Neuron ID")
    axes[1].set_title(f"Descending Drive Population Activity (n={descending_drive_pool.n})")
    axes[1].set_ylim(-1, descending_drive_pool.n)
    axes[1].grid(True, alpha=0.3)

    mn_colors = plt.get_cmap("Reds")(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
    active_mn_count = 0
    for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
        if len(spiketrain) > 0:
            spike_times = spiketrain.rescale(pq.s).magnitude
            axes[2].scatter(spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8)
            active_mn_count += 1

    axes[2].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
    axes[2].set_title(
        f"Motor Neuron Population Activity (n={active_mn_count}/{motor_neuron_pool.n} active)"
    )
    axes[2].set_ylim(-1, motor_neuron_pool.n)
    axes[2].grid(True, alpha=0.3)

    bin_size_ms = 100
    dd_counts, bin_centers_s = population_psth(dd_segment.spiketrains, bin_size_ms * pq.ms)
    dd_rates_binned = dd_counts / (bin_size_ms / 1000.0) / descending_drive_pool.n

    mn_counts, _ = population_psth(mn_segment.spiketrains, bin_size_ms * pq.ms)
    mn_rates_binned = mn_counts / (bin_size_ms / 1000.0) / motor_neuron_pool.n
    
    axes[3].plot(bin_centers_s, dd_rates_binned, "b-", linewidth=2, label="DD Population", alpha=0.8)
    axes[3].plot(bin_centers_s, mn_rates_binned, "r-", linewidth=2, label="MN Population", alpha=0.8)

    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Population Rate (Hz)")
    axes[3].set_title("Population Firing Rates Over Time")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(0, simulation_time.rescale(pq.s).magnitude)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: uv run python simular_artigo.py <arquivo.yaml>")
        print("Exemplo: uv run python simular_artigo.py results/parametros_normal.yaml")
        sys.exit(1)

    caminho = sys.argv[1]
    print(caminho)
    rodar_simulacao(caminho)