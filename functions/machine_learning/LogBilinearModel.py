import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------- Dataset Definition ----------
class CoocDataset(Dataset):
    def __init__(self, cooc_matrix):
        word_idx, context_idx = torch.nonzero(cooc_matrix, as_tuple=True)
        counts = cooc_matrix[word_idx, context_idx]
        self.data = list(zip(word_idx.tolist(), context_idx.tolist(), counts.tolist()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wi, ci, count = self.data[idx]
        return torch.tensor(wi), torch.tensor(ci), torch.tensor(count, dtype=torch.float)

# ---------- Model ----------
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

# ---------- Weighting Function ----------
def weighting_function(x, x_max=100, alpha=0.75):
    wx = (x / x_max) ** alpha
    return torch.where(x < x_max, wx, torch.ones_like(x))

# ---------- Training Function ----------
def train(cooc_matrix, embedding_dim=50, epochs=50, batch_size=256, learning_rate=0.01,
          x_max=100, alpha=0.75, num_workers=2, save_dir="training_logs", use_gpu=True):
    """ Train a Log-Bilinear model on a co-occurrence matrix.
    Args:
        cooc_matrix (torch.Tensor): Co-occurrence matrix of shape (vocab_size, vocab_size).
        embedding_dim (int): Dimension of the word embeddings.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Learning rate for the optimizer.
        x_max (int): Constant for weighting function.
        alpha (float): Exponent for weighting function.
        num_workers (int): Number of workers for DataLoader.
        save_dir (str): Directory to save model weights and logs.
        use_gpu (bool): Whether to use GPU if available.
    Function:
        Trains a Log-Bilinear model on the provided co-occurrence matrix.
        Saves the model weights and training loss history to the specified directory.
    Returns:
        None
    """
    
    print("Preparing training directory and device...")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")

    vocab_size = cooc_matrix.shape[0]
    print(f"Vocabulary size: {vocab_size}, Embedding dim: {embedding_dim}")
    model = LogBilinearModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Creating dataset and dataloader...")
    dataset = CoocDataset(cooc_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    loss_history = []

    print("Starting training loop...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} started.")
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for word_idx, context_idx, count in pbar:
            word_idx = word_idx.to(device)
            context_idx = context_idx.to(device)
            count = count.to(device)

            optimizer.zero_grad()
            prediction = model(word_idx, context_idx)
            log_count = torch.log(count + 1e-10)
            weight = weighting_function(count.to(device), x_max=x_max, alpha=alpha)

            loss = weight * (prediction - log_count) ** 2
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value * len(word_idx)
            pbar.set_postfix(loss=f"{loss_value:.4f}")

        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch + 1} finished. Average loss: {avg_loss:.6f}")

        print(f"Saving weights for epoch {epoch + 1}...")
        torch.save(model.word_embeddings.weight.cpu().detach(),
                   os.path.join(save_dir, f"weights_epoch_{epoch + 1}.pt"))

    np.save(os.path.join(save_dir, "loss_history.npy"), np.array(loss_history))

    print("Training complete. Saving final embeddings...")
    final_embeddings = model.word_embeddings.weight.cpu().detach()
    torch.save(final_embeddings, os.path.join(save_dir, "final_word_vectors.pt"))
    print("Final embeddings saved.")


''' Not used.
# ---------- Visualization ----------
def visualize_embeddings(embeddings, labels=None, title="Word Vector Visualization (PCA)", save_path=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], edgecolors='k', c='lightblue')

    if labels:
        for i, label in enumerate(labels):
            plt.text(reduced[i, 0], reduced[i, 1], str(label), fontsize=9, ha='right', va='bottom')

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if save_path:
        plt.savefig(save_path)
    plt.show()
'''


if __name__ == "__main__":
    import numpy as np

    print("Loading co-occurrence matrix...")
    npy_file_path = 'testing_scrap_misc/scrap_01/cooccurrence_probability_matrix.npy'
    cooc_matrix = np.load(npy_file_path, allow_pickle=True)
    print("Co-occurrence matrix loaded.")

    # Convert cooc_matrix to a torch tensor
    cooc_matrix = torch.tensor(cooc_matrix, dtype=torch.float32)

    print("Starting training process...")
    final_embeddings = train(cooc_matrix, embedding_dim=10, epochs=10, batch_size=256, learning_rate=0.01,
          x_max=100, alpha=0.75, num_workers=0, save_dir="training_logs", use_gpu=True)
    print("Training process finished.")