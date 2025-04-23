import torch
from torch.utils.data import DataLoader
from datasets.load_power_traces import PowerTraceDataset, collate_fn
from models.transformer1 import PowerTraceTransformer
from torch.utils.data import random_split
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Configuration du device
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
best_acc = 0.0
train_losses, val_losses = [], []

for epoch in range(4):
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

    train_losses.append(total_loss)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    # Validation
    model.eval()
    correct = total = 0
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out = model(x, mask)
            loss = criterion(out, y)
            val_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = correct / total
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} - Validation accuracy: {acc:.2%}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "weights/best_model.pth")
        print(f"✅ Nouveau meilleur modèle sauvegardé avec {acc:.2%} de précision !")

# Analyse finale
print("\n\n=== Rapport de classification ===")
print(classification_report(all_labels, all_preds, target_names=list(dataset.label_map.values())))

print("\n=== Matrice de confusion ===")
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=list(dataset.label_map.values())).plot()
plt.title("Confusion matrix - Validation")
plt.savefig("confusion_matrix.png")
plt.show()

# Courbe de pertes
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Courbe de perte")
plt.savefig("loss_graph.png")
plt.grid(True)
plt.show()