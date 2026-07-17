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

#SALVAR RESULTADO
joblib.dump(rt_fuglevand, save_path / "thresholds.pkl")