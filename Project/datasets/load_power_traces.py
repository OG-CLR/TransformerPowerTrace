import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class PowerTraceDataset(Dataset):
    def __init__(self, data_dir, file_suffix="_all.csv", normalize=True):
        self.traces = []
        self.labels = []
        self.label_map = {}
        
        # Lister les fichiers
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(file_suffix)])

        for label_id, filename in enumerate(files):
            function_name = filename.replace(file_suffix, "")
            self.label_map[label_id] = function_name

            print(f"Chargement : {filename} → label {label_id} ({function_name})")

            # Charger le fichier
            df = pd.read_csv(os.path.join(data_dir, filename))
            df = df.select_dtypes(include=[np.number])
            df = df.dropna(axis=1, how='any')
            arr = df.to_numpy(dtype=np.float32)  # shape: (n_traces, T)

            if normalize:
                arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-8)

            # Ajouter chaque trace individuellement
            for trace in arr:
                self.traces.append(torch.tensor(trace, dtype=torch.float32))
                self.labels.append(label_id)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # Retourne (T, 1), label
        return self.traces[idx].unsqueeze(-1), self.labels[idx]


# Fonction pour coller les batches et gérer le padding
def collate_fn(batch):
    sequences, labels = zip(*batch)  # chaque sequence a taille (Ti, 1)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # (B, T_max, 1)
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    attention_mask = torch.arange(padded_seqs.size(1)).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T_max)
    return padded_seqs, torch.tensor(labels), attention_mask

