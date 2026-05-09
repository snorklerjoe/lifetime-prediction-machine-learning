import torch

class RULModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(RULModel, self).__init__()
        self.cnnconv1 = torch.nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1)
        self.cnnrelu = torch.nn.ReLU()
        self.cnnpool = torch.nn.AdaptiveAvgPool1d(1)  # Global average pooling to get a fixed-size output
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, esw, rth_vt):
        # Ensure rth_vt has batch dimension
        if rth_vt.dim() == 2:
            rth_vt = rth_vt.unsqueeze(0)
            
        # CNN branch for Esw
        esw = esw.unsqueeze(1)  # Add channel dimension
        cnn_out = self.cnnconv1(esw)
        cnn_out = self.cnnrelu(cnn_out)
        cnn_out = self.cnnpool(cnn_out) # shape: [batch, hidden_size, 1]
        
        # Attention expects [batch, L, E]
        cnn_out = cnn_out.transpose(1, 2) # shape: [batch, 1, hidden_size]

        # LSTM branch for Rth and Vt
        lstm_out, _ = self.lstm(rth_vt) # shape: [batch, seq_len, hidden_size]
        
        # Cross attention between CNN and LSTM outputs
        attn_output, _ = self.attention(cnn_out, lstm_out, lstm_out) # out shape: [batch, 1, hidden_size]
        
        # Final RUL prediction
        rul_pred = self.fc(attn_output.squeeze(1)) # out shape: [batch, 1]
        
        return rul_pred

model = RULModel(input_size=2, hidden_size=32, num_layers=2)
esw = torch.randn(1, 1127)
rth_vt = torch.randn(1127, 2)
out = model(esw, rth_vt)
print("Output shape:", out.shape)
