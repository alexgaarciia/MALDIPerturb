import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


def make_loader(X, y, batch_size=256, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)), batch_size=batch_size, shuffle=shuffle)

def predict(model, X, device, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], dtype=torch.float32).to(device)
            out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)
