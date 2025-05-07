import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

def generate_series(n=1000):
    t = np.arange(n)
    series = np.sin(0.02 * t) + 0.5 * np.random.randn(n)
    anomalies = np.random.choice(n, size=10, replace=False)
    series[anomalies] += np.random.uniform(5, 10, size=anomalies.shape)
    return pd.DataFrame({'value': series})

class SequenceDataset(Dataset):
    def __init__(self, data, seq_len=30):
        from sklearn.preprocessing import MinMaxScaler
        import torch
        self.scaler = MinMaxScaler()
        vals = self.scaler.fit_transform(data[['value']].values)
        self.sequences = [vals[i:i+seq_len] for i in range(len(vals) - seq_len)]
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i], self.sequences[i]

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=30, n_features=1, embedding_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        self.enc_fc = nn.Linear(64, embedding_dim)
        self.dec_fc = nn.Linear(embedding_dim, 64)
        self.decoder = nn.LSTM(input_size=64, hidden_size=n_features, batch_first=True)
    def forward(self, x):
        out, _ = self.encoder(x)
        emb = self.enc_fc(out[:, -1, :])
        dec_in = self.dec_fc(emb).unsqueeze(1).repeat(1, x.size(1), 1)
        rec, _ = self.decoder(dec_in)
        return rec

def train():
    df = generate_series()
    os.makedirs('models', exist_ok=True)
    df.to_csv('data_series.csv', index=False)

    ds = SequenceDataset(df)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = LSTMAutoencoder()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 31):
        total_loss = 0
        for x, y in loader:
            optim.zero_grad()
            rec = model(x)
            loss = loss_fn(rec, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/30, loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), 'models/model.pth')

    errs = []
    with torch.no_grad():
        for x, y in loader:
            rec = model(x)
            errs.append(((rec - y) ** 2).mean(dim=(1,2)).numpy())
    errs = np.concatenate(errs)
    threshold = errs.mean() + 3 * errs.std()
    import numpy as _np; _np.save('models/threshold.npy', threshold)
    print("Saved model and threshold:", threshold)

if __name__ == '__main__':
    train()
