import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_attention_colored_trace(trace, attn_weights, figsize=(14, 4), cmap="viridis"):
    """
    Affiche une trace de puissance colorée selon les poids d'attention.

    Arguments :
    - trace : (T,) numpy array ou tensor 1D
    - attn_weights : (T,) numpy array ou tensor 1D, valeurs entre 0 et 1 (après softmax)
    - figsize : taille du graphique
    - cmap : colormap matplotlib (ex: viridis, plasma, inferno...)
    """
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    norm = plt.Normalize(vmin=0, vmax=attn_weights.max())
    colors = cm.get_cmap(cmap)(norm(attn_weights))

    plt.figure(figsize=figsize)
    for i in range(len(trace)-1):
        plt.plot([i, i+1], [trace[i], trace[i+1]], color=colors[i])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical")
    cbar.set_label("Attention weight")

    plt.title("Trace colorée selon les poids d'attention")
    plt.xlabel("Temps")
    plt.ylabel("Puissance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
