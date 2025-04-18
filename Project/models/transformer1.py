import torch
import torch.nn as nn

class PowerTraceTransformer(nn.Module):
    def __init__(self, d_model=128, num_classes=11, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # Projette l’entrée 1D (puissance à chaque timestep) vers un vecteur d_model (embedding dimension)
        self.input_proj = nn.Linear(1, d_model)  # Entrée: (B, T, 1) → Sortie: (B, T, d_model)

        # Crée un bloc d’encodeur Transformer avec attention multi-tête, normalisation, MLP
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,       # dimension des vecteurs d’entrée/sortie de chaque couche
            nhead=nhead,           # nombre de têtes d’attention
            dim_feedforward=4*d_model,  # dimension du MLP interne (classique: 4 × d_model)
            dropout=dropout,       # taux de dropout
            batch_first=True       # pour avoir les dimensions (B, T, C) au lieu de (T, B, C)
        )

        # Empile plusieurs couches d’encodeur Transformer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Couche finale de classification → chaque trace est associée à une fonction
        self.classifier = nn.Linear(d_model, num_classes)

        # Attention pooling learnable : pour calculer un score d’importance par timestep
        self.attn_pool = nn.Linear(d_model, 1)

        # Encodage positionnel appris → indique à quelle position temporelle se trouve chaque vecteur
        self.pos_embedding = nn.Embedding(10_000, d_model)  # max longueur temporelle = 10 000

    def forward(self, x, mask):
        # x : (B, T, 1) — batch de traces, chaque trace étant une séquence 1D de puissance
        # mask : (B, T) — booléen, True pour les positions valides, False pour les paddings

        # Applique la projection linéaire sur chaque point de la trace
        x = self.input_proj(x)  # → (B, T, d_model)

        # Récupère la taille de la séquence
        B, T, _ = x.size()

        # Crée un vecteur de positions [0, 1, ..., T-1], broadcasté sur le batch
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        # Ajoute l’encodage de position appris aux vecteurs de puissance encodés
        x = x + self.pos_embedding(positions)  # (B, T, d_model)

        # Crée un masque de padding pour l’attention : True = ignorer (donc ~mask)
        padding_mask = ~mask  # (B, T)

        # Applique les couches Transformer : chaque position regarde toutes les autres
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)

        # Attention pooling :
        # Chaque vecteur temporel x[t] passe dans une couche linéaire → score scalaire d’importance
        attn_scores = self.attn_pool(x).squeeze(-1)  # (B, T)

        # On masque les positions padding avec -∞ pour qu'elles aient une proba nulle
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))  # (B, T)

        # Softmax pour transformer les scores en poids d’attention (somme = 1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        # Applique les poids à chaque vecteur, puis somme pondérée → vecteur global (B, d_model)
        x = (x * attn_weights).sum(dim=1)  # (B, d_model)

        # Enfin, passe dans la couche de classification → (B, num_classes)
        return self.classifier(x)
