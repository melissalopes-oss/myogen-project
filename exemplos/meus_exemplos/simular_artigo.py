# -*- coding: utf-8 -*-
"""
Helper script to generate spike‑train data for MyoGen HD‑sEMG simulations.
It loads the parameter YAML, builds recruitment thresholds, creates a simple
cortical drive (sine‑modulated Poisson), runs the network to obtain a
neo.Block with spike trains, and saves the block as
`results/sine_dd_spike_trains_<condicao>.pkl`.

Usage:
    uv run python exemplos/meus_exemplos/simular_artigo.py <yaml_path>
Example:
    uv run python exemplos/meus_exemplos/simular_artigo.py results/parametros_normal.yaml
"""

import sys
from pathlib import Path
import yaml
import joblib
import numpy as np
import quantities as pq

from myogen import simulator
from myogen.utils.neo import spike_trains_to_block


def carregar_parametros(caminho_yaml: str) -> dict:
    """Load the YAML file containing simulation parameters."""
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gerar_spike_trains(parametros: dict) -> tuple:
    """Create recruitment thresholds and a simple cortical drive.
    Returns a tuple (spike_train_block, description).
    """
    # Recruitment ratio (max / min) as in the article
    razao = parametros["forca_maxima_N"] / parametros["forca_minima_N"]
    recruitment_thresholds, _ = simulator.RecruitmentThresholds(
        N=parametros["numero_motoneuronios"],
        recruitment_range__ratio=razao,
        mode="combined",
        deluca__slope=5,
    )

    # Simple sinusoidal cortical drive (frequency 2 Hz, amplitude 0.5)
    duration = parametros.get("duracao_simulacao_s", 2.0) * pq.s
    dt = 1.0 / 2000.0 * pq.s  # 2 kHz sampling for the drive
    t = np.arange(0, float(duration), float(dt))
    cortical_rate = 30 + 10 * np.sin(2 * np.pi * 2 * t)  # 2 Hz sine

    # Build a Poisson spike train generator for each motor neuron
    spikes = []
    for _ in range(parametros["numero_motoneuronios"]):
        # Poisson process with time‑varying rate
        probs = cortical_rate * float(dt)
        spikes.append(np.random.rand(len(t)) < probs)

    # Assemble into a neo.Block (compatible with SurfaceEMG)
    # Here we use a minimal helper: create a BinnedSpikeTrain per MU
    from myogen.utils.neo import spike_trains_to_block
    spike_block = spike_trains_to_block(spikes, dt)
    return spike_block


def main(caminho_yaml: str):
    parametros = carregar_parametros(caminho_yaml)
    condicao = parametros.get("condicao", "desconhecida").lower()
    spike_block = gerar_spike_trains(parametros)

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / f"sine_dd_spike_trains_{condicao}.pkl"
    joblib.dump(spike_block, outfile)
    print(f"[OK] Spike‑train block saved to {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python simular_artigo.py <yaml_path>")
        sys.exit(1)
    main(sys.argv[1])