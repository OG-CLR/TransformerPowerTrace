import torch
from datasets.load_power_traces import PowerTraceDataset
from models.transformer1 import PowerTraceTransformer
from visualization.attention_visualization import plot_attention_colored_trace
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le dataset
dataset = PowerTraceDataset("Data")

# Créer une instance du modèle avec le bon nombre de classes
model = PowerTraceTransformer(num_classes=len(dataset.label_map)).to(device)

# Charger les poids du modèle sauvegardé
model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
model.eval()  # mode évaluation pour désactiver dropout, etc.

# Choisir une classe à afficher
requested_class = "memcopy"  # 🔹 changer ici le nom de la classe
label_map = dataset.label_map

# Trouver l'indice de classe correspondant
label_index = None
for idx, name in label_map.items():
    if name == requested_class:
        label_index = idx
        break

if label_index is None:
    raise ValueError(f"Classe '{requested_class}' introuvable dans le dataset.")

# Initialiser la figure
fig, axs = plt.subplots(5, 3, figsize=(16, 10))
axs = axs.flatten()
num_traces = 15

# Normalisation commune
all_weights = []
all_traces = []

count = 0
for sample_idx, (x, y, mask) in enumerate(dataset):
    if y.item() != label_index:
        continue

    x_in = x.unsqueeze(0).to(device)
    mask_in = mask.unsqueeze(0).to(device)

    attn_weights = model.get_attention_weights(x_in, mask_in)[0]  # (T,)
    trace = x.squeeze().cpu()                                     # (T,)

    all_weights.append(attn_weights)
    all_traces.append(trace)
    count += 1
    if count >= num_traces:
        break

# Trouver le max global pour normaliser la colormap
max_attn = max(w.max() for w in all_weights)
norm = plt.Normalize(vmin=0, vmax=max_attn)

for i in range(num_traces):
    trace = all_traces[i]
    attn_weights = all_weights[i]
    colors = cm.get_cmap("viridis")(norm(attn_weights))

    ax = axs[i]
    for j in range(len(trace) - 1):
        ax.plot([j, j + 1], [trace[j], trace[j + 1]], color=colors[j])
    ax.set_title(f"{requested_class} #{i}")
    ax.set_xticks([])
    ax.set_yticks([])

# Ajouter une barre de couleur commune
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Poids d'attention")

plt.tight_layout(rect=[0, 0, 0.93, 1])  
plt.savefig(f"attention_traces_{requested_class}_all.png")
print(f"✅ Figure globale sauvegardée sous attention_traces_{requested_class}_all.png")
plt.show()
