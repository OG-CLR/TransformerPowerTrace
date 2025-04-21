import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configuration ===
data_dir = "Data"  # 🔁 À adapter
function_name = "strncat"              # 🔁 Choisis la fonction à tracer
file_path = os.path.join(data_dir, f"{function_name}_all.csv")

# === Lecture du fichier CSV ===
df = pd.read_csv(file_path, header=None).fillna(0.0)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

# === Tracer les 10 premières traces ===
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.plot(df.iloc[i], label=f"Trace {i+1}")

plt.title(f"10 Traces énergétiques de l'instruction '{function_name}'")
plt.xlabel("Temps (échantillons)")
plt.ylabel("Énergie (u.a.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{function_name}_power_traces.png")

plt.show()
