import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Sequence
from functools import partial
import numpy as np
import random

def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CpGPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2):
        super(CpGPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, num_classes=5).float()
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]
        return self.classifier(final_hidden).squeeze()

def main():
    set_seed(13)
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_x, train_y = prepare_data(2048)
    test_x, test_y = prepare_data(512)
    
    train_dataset = DNADataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = CpGPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_model(model, train_loader, criterion, optimizer, EPOCHS, device)
    torch.save(model.state_dict(), 'cpg_model.pth')

class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        yield [random.randint(0, 4) for _ in range(seq_len)]

def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        if seq[i:i+2] == "CG":
            cgs += 1
    return cgs

alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(5))}
int2dna = {i: a for a, i in zip(alphabet, range(5))}

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

def prepare_data(num_samples=100):
    X_dna_seqs = list(rand_sequence(num_samples))
    temp = [''.join(intseq_to_dnaseq(seq)) for seq in X_dna_seqs]
    y_dna_seqs = [count_cpgs(seq) for seq in temp]
    return X_dna_seqs, y_dna_seqs

def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()
