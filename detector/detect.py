import numpy as np
import pandas as pd
import torch
from train import SequenceDataset, LSTMAutoencoder
from torch.utils.data import DataLoader

def detect():
    df = pd.read_csv('data_series.csv')
    ds = SequenceDataset(df)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    model = LSTMAutoencoder()
    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()

    threshold = np.load('models/threshold.npy')
    errors = []
    with torch.no_grad():
        for x, y in loader:
            rec = model(x)
            errs = ((rec - y)**2).mean(dim=(1,2)).numpy()
            errors.extend(errs)
    df['reconstruction_error'] = [None]*(len(df)-len(errors)) + errors
    df['anomaly'] = df['reconstruction_error'] > threshold
    df.to_csv('anomalies.csv', index=False)
    print("Anomalies saved to anomalies.csv")

if __name__ == '__main__':
    detect()
