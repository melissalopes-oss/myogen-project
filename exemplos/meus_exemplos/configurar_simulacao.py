import sys
from pathlib import Path

import quantities as pq
import yaml  

from myogen import simulator
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool


def carregar_parametros(caminho_yaml: str) -> dict:
    """Lê o arquivo YAML e retorna um dicionário com os parâmetros."""
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        parametros = yaml.safe_load(f)
    return parametros


def montar_simulacao(parametros: dict):
    """Usa os parâmetros do YAML para configurar os objetos do MyoGen."""

    print(f"\n{'='*60}")
    print(f"Configurando simulação: {parametros['condicao']}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------
    #  Limiares de recrutamento (usa a razão de força do artigo
    #  como o recruitment_range do MyoGen)
    # ------------------------------------------------------------
    recruitment_thresholds, _ = simulator.RecruitmentThresholds(
        N=parametros["numero_motoneuronios"],
        recruitment_range__ratio=parametros["razao_forca"],
        mode="combined",
        deluca__slope=5,
    )
    print(f"[1] Limiares de recrutamento criados para {parametros['numero_motoneuronios']} unidades motoras")
    print(f"    Razão de recrutamento usada: {parametros['razao_forca']}")

    # ------------------------------------------------------------
    #  Modelo de força (usa o tempo de contração MÁXIMO da
    #  tabela, que corresponde à unidade motora mais lenta)
    # ------------------------------------------------------------
    force_model = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=2048 * pq.Hz,
        longest_duration_rise_time__ms=parametros["tempo_contracao_maximo_ms"] * pq.ms,
        contraction_time_range_factor=parametros["fator_espalhamento_tempo"],
    )
    print(f"[2] Modelo de força criado")
    print(f"    Tempo de contração (unidade mais lenta): {parametros['tempo_contracao_maximo_ms']} ms")
    print(f"    Força de pico (escala do MyoGen, sem unidade): {force_model.peak_twitch_forces__unitless[0]:.4f} - {force_model.peak_twitch_forces__unitless[-1]:.4f}")

    # --------------------------------------------------------
    # Fator de conversão: transforma a escala "sem unidade"
    # do MyoGen (que vai de ~1 até a razão de recrutamento)
    # em Newtons reais, usando os valores do artigo (Tabela 1).
    # Confirmado com o professor: como não estamos usando um
    # modelo de músculo completo (que receberia o sinal
    # normalizado 0-1), multiplicar por um fator fixo é válido.
    # --------------------------------------------------------
    fator_conversao_N = parametros["forca_minima_N"] / force_model.peak_twitch_forces__unitless[0]
    forca_pico_N = force_model.peak_twitch_forces__unitless * fator_conversao_N

    print(f"    Fator de conversão calculado: {fator_conversao_N:.6f} N por unidade MyoGen")
    print(f"    Força de pico em Newtons (primeira UM): {forca_pico_N[0]:.4f} N (artigo: {parametros['forca_minima_N']} N)")
    print(f"    Força de pico em Newtons (última UM): {forca_pico_N[-1]:.4f} N (artigo: {parametros['forca_maxima_N']} N)")

    # ------------------------------------------------------------
    # Pool de motoneurônios (usa a velocidade de condução
    # mínima e máxima da tabela)
    # ------------------------------------------------------------
    motor_neuron_pool = AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        axon_velocities=(
            parametros["velocidade_conducao_minima_m_s"],
            parametros["velocidade_conducao_maxima_m_s"],
        ),
    )
    print(f"[3] Pool de motoneurônios criado")
    print(
        f"    Velocidade de condução: "
        f"{parametros['velocidade_conducao_minima_m_s']}-"
        f"{parametros['velocidade_conducao_maxima_m_s']} m/s"
    )

    # ------------------------------------------------------------
    # Comando descendente (usa o número de neurônios
    # corticoespinhais da tabela)
    # ------------------------------------------------------------
    timestep = 0.05 * pq.ms  # mesmo passo de integração do artigo
    descending_drive_pool = DescendingDrive__Pool(
        n=parametros["numero_neuronios_corticoespinhais"],
        process_type="poisson",
        timestep__ms=timestep,
        poisson_batch_size=parametros["numero_neuronios_corticoespinhais"],
    )
    print(f"[4] Comando descendente criado")
    print(f"    Número de neurônios corticoespinhais: {parametros['numero_neuronios_corticoespinhais']}")

    print(f"\n[OK] Todos os componentes da condição '{parametros['condicao']}' foram criados.")
    print("Próximo passo: conectar esses componentes numa Network (como no exemplo 3)")
    print("e rodar a simulação temporal, igual fizemos nos exemplos anteriores.\n")

    return {
        "recruitment_thresholds": recruitment_thresholds,
        "force_model": force_model,
        "fator_conversao_N": fator_conversao_N,
        "forca_pico_N": forca_pico_N,
        "motor_neuron_pool": motor_neuron_pool,
        "descending_drive_pool": descending_drive_pool,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python configurar_simulacao.py <arquivo.yaml>")
        print("Exemplo: python configurar_simulacao.py parametros_normal.yaml")
        sys.exit(1)

    caminho = sys.argv[1]
    parametros = carregar_parametros(caminho)
    componentes = montar_simulacao(parametros)