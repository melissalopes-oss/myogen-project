#Exemplo 6 do MyoGen - Surface EMG Signals
#Junta os MUAPs (exemplo 5) com os trens de disparo (exemplo 3) p gerar
#o sinal de EMG de superfície final

from pathlib import Path

import joblib
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt

from myogen import simulator
from myogen.utils.neo import signal_to_grid
from myogen.utils.types import CURRENT__AnalogSignal, SPIKE_TRAIN__Block

plt.style.use("fivethirtyeight")

save_path = Path("./results")

spike_train__Block: SPIKE_TRAIN__Block = joblib.load(save_path / "sine_dd_spike_trains.pkl")
input_current__AnalogSignal: CURRENT__AnalogSignal = joblib.load(
    save_path / "sine_drive_pattern.pkl"
)
surface_emg: simulator.SurfaceEMG = joblib.load(save_path / "surface_emg.pkl")

# Generate Surface EMG
# Aqui é onde acontece a "soma": pega os MUAPs (o "timbre" de cada
# unidade motora) e os spike trains (quando cada uma disparou) e
# junta tudo para formar o sinal de EMG final.

surface_emg_signals = surface_emg.simulate_surface_emg(spike_train__Block=spike_train__Block)

print("Surface EMG simulation completed!")

# Acessa o primeiro grupo (grade de eletrodos) e primeiro segmento (pool)
first_emg_signal = surface_emg_signals.groups[0].segments[0].analogsignals[0]
grid_shape = first_emg_signal.annotations["grid_shape"]
print(f"Generated EMG shape: {first_emg_signal.shape} (stored as 2D for NWB compatibility)")
print(f"  - {first_emg_signal.shape[0]} time samples")
print(f"  - {grid_shape[0]} electrode rows")
print(f"  - {grid_shape[1]} electrode columns")

# Salva o resultado
joblib.dump(surface_emg_signals, save_path / "surface_emg_signals.pkl")

# Visualize Surface EMG Results
# MyoGen é um simulador = resultados s ruído do mundo real
# Por isso adicionamos ruído artificial para deixar o sinal mais pro sinal real de lab

noisy_surface_emg__Block = surface_emg.add_noise(snr__dB=5.0)

plt.rcParams.update({"font.size": 24})

fig, ax = plt.subplots(figsize=(12, 6))

# Pega o sinal de EMG do eletrodo central da grade (linha 2, coluna 2)
noisy_signal = noisy_surface_emg__Block.groups[0].segments[0].analogsignals[0]
emg_grid = signal_to_grid(noisy_signal)  # Converte para formato 3D (tempo, linhas, colunas)
emg_signal = emg_grid[:, 2, 2]
current_signal = input_current__AnalogSignal[:, 0].magnitude

# Normaliza a corrente de entrada entre 0 e 1, para caber no mesmo gráfico
current_normalized = (current_signal - np.min(current_signal)) / (
    np.max(current_signal) - np.min(current_signal)
)

current_normalized *= np.max(emg_signal)  # Escala para o mesmo range de amplitude do EMG

# Plota os dois sinais normalizados no mesmo eixo
ax.plot(
    np.arange(len(emg_signal)) / surface_emg.sampling_frequency__Hz,
    emg_signal,
    linewidth=2,
    label="Surface EMG",
)

ax.plot(
    input_current__AnalogSignal.times.rescale(pq.s).magnitude,
    current_normalized,
    linewidth=2,
    label="Input Current (senoide)",
    alpha=1,
    zorder=4,
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (mV)")
ax.legend()

sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)

plt.title("Surface EMG\nand Input Current (Senoide)")

plt.tight_layout()
plt.show()

print("\n[DONE] Sinal de EMG de superfície gerado com sucesso!")