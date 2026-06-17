from pathlib import Path
import joblib
from matplotlib import pyplot as plt
from myogen import simulator

plt.style.use("fivethirtyeight")

#PARÂMETROS
n_motor_units = 10      # Número de unidades motoras
recruitment_range = 6  # A maior UM precisa de x mais força que a menor

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

#MODELO FUGLEVAND
rt_fuglevand, _ = simulator.RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    mode="fuglevand"
)

plt.plot(rt_fuglevand * 100, "-o", label="Fuglevand Model")
plt.title("Fuglevand Recruitment Thresholds")
plt.xlabel("Motor Unit (#)")
plt.ylabel("Recruitment Threshold (%)")
plt.tight_layout()
plt.show()

# --- MODELO DE LUCA ---
deluca_results = {
    slope: simulator.RecruitmentThresholds(
        N=n_motor_units,
        recruitment_range__ratio=recruitment_range,
        deluca__slope=slope,
        mode="deluca",
    )[0]
    for slope in [0.001, 5, 25, 50]
}

for s, rt in deluca_results.items():
    plt.plot(rt * 100, "-o", label=f"Slope={s}")
plt.title("De Luca Recruitment Thresholds")
plt.xlabel("Motor Unit (#)")
plt.ylabel("Recruitment Threshold (%)")
plt.legend(framealpha=1.0, edgecolor="none")
plt.tight_layout()
plt.show()

# --- MODELO KONSTANTIN ---
rt_konstantin, _ = simulator.RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    konstantin__max_threshold__ratio=1.0,
    mode="konstantin",
)

plt.plot(rt_konstantin * 100, "-o", label="Konstantin Model")
plt.title("Konstantin Recruitment Thresholds")
plt.xlabel("Motor Unit (#)")
plt.ylabel("Recruitment Threshold (%)")
plt.tight_layout()
plt.show()

# --- MODELO COMBINADO ---
combined_results = {
    slope: simulator.RecruitmentThresholds(
        N=n_motor_units,
        recruitment_range__ratio=recruitment_range,
        deluca__slope=slope,
        konstantin__max_threshold__ratio=1.0,
        mode="combined",
    )[0]
    for slope in [0.001, 5, 25, 50]
}

for s, rt in combined_results.items():
    plt.plot(rt * 100, "-o", label=f"Slope={s}")
plt.title("Combined Recruitment Thresholds")
plt.xlabel("Motor Unit (#)")
plt.ylabel("Recruitment Threshold (%)")
plt.legend(framealpha=1.0, edgecolor="none")
plt.tight_layout()
plt.show()

#SALVAR RESULTADO
joblib.dump(combined_results[5], save_path / "thresholds.pkl")