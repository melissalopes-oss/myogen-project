"""
Simulação de HD-sEMG (High-Density Surface EMG) a partir dos parâmetros do artigo.
Baseado nos Exemplos 04, 05 e 06 do MyoGen.
"""

import sys
from pathlib import Path
import yaml
import joblib
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

from myogen import simulator
from myogen.utils.neo import signal_to_grid

plt.style.use("fivethirtyeight")


def carregar_parametros(caminho_yaml: str) -> dict:
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simular_hd_semg(caminho_yaml: str):
    parametros = carregar_parametros(caminho_yaml)
    condicao_nome = parametros["condicao"]

    print(f"\n{'='*60}")
    print(f"Simulando HD-sEMG para a Condição: {condicao_nome}")
    print(f"{'='*60}\n")

    save_path = Path("./results")
    save_path.mkdir(exist_ok=True)

    # ============================================================
    # 1. Carregar ou Gerar os Limiares e os Spike Trains
    # ============================================================
    # Razão de recrutamento do artigo (Máximo / Mínimo)
    razao_recrutamento = parametros["forca_maxima_N"] / parametros["forca_minima_N"]

    recruitment_thresholds, _ = simulator.RecruitmentThresholds(
        N=parametros["numero_motoneuronios"],
        recruitment_range__ratio=razao_recrutamento,
        mode="combined",
        deluca__slope=5,
    )

    # Tenta carregar os spike trains que você já salvou no simular_artigo.py
    arquivo_spikes = save_path / f"sine_dd_spike_trains_{condicao_nome.lower()}.pkl"
    if arquivo_spikes.exists():
        print(f"[OK] Carregando spike trains salvos de: {arquivo_spikes.name}")
        spike_train_block = joblib.load(arquivo_spikes)
    else:
        print(f"[AVISO] Arquivo {arquivo_spikes.name} não encontrado.")
        print("Certifique-se de ter rodado o simular_artigo.py antes para gerar os disparos!")
        return

    # ============================================================
    # 2. Modelo Muscular (Baseado no Exemplo 04)
    # ============================================================
    print("\n2. Criando o modelo anatômico do músculo (Muscle)...")
    muscle_radius = 4.9 * pq.mm
    fiber_density = 400 * pq.mm**-2

    muscle = simulator.Muscle(
        recruitment_thresholds=recruitment_thresholds,
        radius_bone__mm=1.0 * pq.mm,
        fiber_density__fibers_per_mm2=fiber_density,
        fat_thickness__mm=3 * pq.mm,
        autorun=True,
    )
    print(f"   Total de fibras musculares geradas: {sum(muscle.resulting_number_of_innervated_fibers)}")

    # ============================================================
    # 3. Grade de Eletrodos de Alta Densidade (HD-sEMG - Exemplo 05)
    # ============================================================
    print("\n3. Configurando a Grade de Eletrodos (HD-sEMG Array)...")
    sampling_frequency = 2048.0 * pq.Hz

    # Grade 5x5 de eletrodos monopolares com distância inter-eletrodo de 5 mm
    electrode_grid = simulator.SurfaceElectrodeArray(
        num_rows=5,
        num_cols=5,
        inter_electrode_distances__mm=5 * pq.mm,
        electrode_radius__mm=1 * pq.mm,
        differentiation_mode="monopolar",
        bending_radius__mm=muscle.radius__mm + muscle.skin_thickness__mm + muscle.fat_thickness__mm,
    )

    # Configura o modelo de superfície para simular os MUAPs
    # Simulamos as unidades motoras ativas para ser rápido e eficiente
    surface_emg = simulator.SurfaceEMG(
        muscle_model=muscle,
        electrode_arrays=[electrode_grid],
        sampling_frequency__Hz=sampling_frequency,
        sampling_points_in_t_and_z_domains=128,  # resolução otimizada
        sampling_points_in_theta_domain=16,
        MUs_to_simulate=list(range(min(20, parametros["numero_motoneuronios"]))),  # primeiras 20 UMs
    )

    print("   Simulando os potenciais de ação na superfície (MUAPs)...")
    surface_emg.simulate_muaps()

    # ============================================================
    # 4. Gerar os Sinais de HD-sEMG na Pele (Exemplo 06)
    # ============================================================
    print("\n4. Convolvendo os MUAPs com os Spike Trains...")
    surface_emg_signals = surface_emg.simulate_surface_emg(spike_train__Block=spike_train_block)

    # Adiciona ruído realista (SNR de 15 dB)
    noisy_emg_block = surface_emg.add_noise(snr__dB=15.0)

    # ============================================================
    # 5. Visualização Multicanal da Grade (High-Density)
    # ============================================================
    print("\n5. Gerando os gráficos dos canais de HD-sEMG...")
    signal_segment = noisy_emg_block.groups[0].segments[0].analogsignals[0]
    emg_grid_3d = signal_to_grid(signal_segment)  # Formato: [tempo, linha, coluna]

    time_s = np.arange(emg_grid_3d.shape[0]) / float(sampling_frequency.magnitude)

    # Plota a matriz 5x5 de canais de eletrodos
    fig, axes = plt.subplots(5, 5, figsize=(14, 10), sharex=True, sharey=False)
    fig.suptitle(f"Sinais HD-sEMG (Grade 5x5) - Condição: {condicao_nome}", fontsize=16, fontweight="bold")

    for r in range(5):
        for c in range(5):
            canal_uV = emg_grid_3d[:, r, c] * 1000.0  # Converte mV para microvolts (uV)
            axes[r, c].plot(time_s, canal_uV, color="C0", linewidth=0.8)
            axes[r, c].set_title(f"Eletrodo [{r},{c}]", fontsize=8)
            axes[r, c].grid(True, alpha=0.2)

    # Ajusta rótulos dos eixos externos
    for ax in axes[-1, :]:
        ax.set_xlabel("Tempo (s)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude (uV)", fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"\n[SUCESSO] Simulação de HD-sEMG para {condicao_nome} concluída com sucesso!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: uv run python simular_hd_semg.py <arquivo.yaml>")
        print("Exemplo: uv run python simular_hd_semg.py results/parametros_normal.yaml")
        sys.exit(1)

    caminho = sys.argv[1]
    simular_hd_semg(caminho)
#uv run python exemplos/meus_exemplos/simular_hd_semg.py results/parametros_normal.yaml
#uv run python exemplos/meus_exemplos/simular_hd_semg.py results/parametros_dpn.yaml