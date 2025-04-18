import torch
from torch.utils.data import DataLoader
from datasets.load_power_traces import PowerTraceDataset, collate_fn
from models.transformer1 import PowerTraceTransformer
from torch.utils.data import random_split
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger dataset
dataset = PowerTraceDataset("Data")

# Split 90/10
n = len(dataset)
n_train = int(0.9 * n)
n_val = n - n_train
train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Créer le modèle
model = PowerTraceTransformer(num_classes=len(dataset.label_map)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Entraînement

best_acc = 0.0  # meilleur score atteint jusqu'à maintenant

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        out = model(x, mask)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    # 🔍 Boucle de validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out = model(x, mask)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} - Validation accuracy: {acc:.2%}")

    if acc > best_acc:
        best_acc = acc
        save_path = "weights/best_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Nouveau meilleur modèle sauvegardé avec {acc:.2%} de précision !")

