import torch

class RULModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(RULModel, self).__init__()
        self.cnnconv1 = torch.nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1)
        self.cnnrelu = torch.nn.ReLU()
        # Remove cnnpool since it's MaxPool1d(1) and does nothing, or keep it.
        # It's harmless but unnecessary. We'll leave it out or keep it.
        self.cnnpool = torch.nn.MaxPool1d(1)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Add batch_first=True to MultiheadAttention
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, esw, rth_vt):
        # rth_vt in train_model is 2D: [seq_len, 2]. We need it to be 3D: [1, seq_len, 2]
        if rth_vt.dim() == 2:
            rth_vt = rth_vt.unsqueeze(0)
            
        # CNN branch for Esw
        # esw in train_model is 2D: [1, seq_len].
        if esw.dim() == 2:
            esw = esw.unsqueeze(1)  # Add channel dimension: [1, 1, seq_len]
            
        cnn_out = self.cnnconv1(esw) # [1, hidden_size, seq_len]
        cnn_out = self.cnnrelu(cnn_out)
        cnn_out = self.cnnpool(cnn_out) # still [1, hidden_size, seq_len]
        
        # Prepare for attention (batch_first=True expects [batch, seq_len, hidden_size])
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM branch for Rth and Vt
        lstm_out, _ = self.lstm(rth_vt) # [1, seq_len, hidden_size]
        
        # Cross attention between CNN and LSTM outputs
        attn_output, _ = self.attention(cnn_out, lstm_out, lstm_out) # [1, seq_len, hidden_size]
        
        # Final RUL prediction
        rul_pred = self.fc(attn_output).squeeze(-1) # [1, seq_len]
        
        return rul_pred

model = RULModel(input_size=2, hidden_size=32, num_layers=2)
esw = torch.randn(1, 1127)
rth_vt = torch.randn(1127, 2)
rul_pred = model(esw, rth_vt)
print("Output shape:", rul_pred.shape)
