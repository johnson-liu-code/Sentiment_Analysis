import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm

class SparseCoocDataset(Dataset):
    def __init__(self, i, j, counts):
        self.i = i
        self.j = j
        self.counts = counts

    def __len__(self):
        return len(self.i)

    def __getitem__(self, idx):
        return self.i[idx], self.j[idx], self.counts[idx]

class LogBilinearModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

    def forward(self, word_idx, context_idx):
        w = self.word_embeddings(word_idx)
        c = self.context_embeddings(context_idx)
        b_w = self.word_biases(word_idx).squeeze()
        b_c = self.context_biases(context_idx).squeeze()
        dot = torch.sum(w * c, dim=1)
        return dot + b_w + b_c

def weighting_function(x, x_max=100, alpha=0.75):
    wx = (x / x_max) ** alpha
    return torch.where(x < x_max, wx, torch.ones_like(x))

def train_sparse_glove(cooc_sparse, embedding_dim=200, epochs=100, batch_size=256,
                       learning_rate=0.01, x_max=100, alpha=0.75, num_workers=4,
                       training_save_dir='training_logs/', use_gpu=True):
    os.makedirs(training_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    i = torch.tensor(cooc_sparse.row, dtype=torch.long)
    j = torch.tensor(cooc_sparse.col, dtype=torch.long)
    x_ij = torch.tensor(cooc_sparse.data, dtype=torch.float32)

    dataset = SparseCoocDataset(i, j, x_ij)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    vocab_size = cooc_sparse.shape[0]
    model = LogBilinearModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for word_idx, context_idx, count in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            word_idx, context_idx, count = word_idx.to(device), context_idx.to(device), count.to(device)
            optimizer.zero_grad()
            prediction = model(word_idx, context_idx)
            log_count = torch.log(count + 1e-10)
            weight = weighting_function(count, x_max=x_max, alpha=alpha)
            loss = weight * (prediction - log_count) ** 2
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(count)

        avg_loss = total_loss / len(train_dataset)
        print(f"Train Loss (Epoch {epoch + 1}): {avg_loss:.4f}")

    torch.save(model.word_embeddings.weight.cpu().detach(), os.path.join(training_save_dir, 'final_word_vectors.pt'))
