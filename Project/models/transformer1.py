import torch
import torch.nn as nn

class PowerTraceTransformer(nn.Module):
    def __init__(self, d_model=128, num_classes=11, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4*d_model, dropout, batch_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.pos_embedding = nn.Embedding(10_000, d_model)  


    def forward(self, x, mask):
        x = self.input_proj(x)
        B, T, _ = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0)  
        x = x + self.pos_embedding(positions) 
        padding_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        mask_float = mask.unsqueeze(-1).float()
        x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        return self.classifier(x)
