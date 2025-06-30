import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def custom_fnn(X: np.ndarray, labels: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    # Train/Val split
    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    # Model
    class FeedforwardNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.net(x)

    model = FeedforwardNN(X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_f1 = 0
    patience = 3
    wait = 0

    train_losses, val_accuracies, val_f1s = [], [], []

    plt.ion()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].set_title("Training Loss")
    ax[1].set_title("Validation Accuracy")
    ax[2].set_title("Validation F1 Score")

    for epoch in range(30):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False)
        for X_batch, y_batch in train_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_preds, val_targets = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False)
        with torch.no_grad():
            for X_val, y_val in val_bar:
                X_val = X_val.to(device)
                y_val = y_val.to(device).unsqueeze(1)
                probs = torch.sigmoid(model(X_val))
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())

        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)

        train_losses.append(total_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        # Live update plots
        ax[0].clear(); ax[0].plot(train_losses); ax[0].set_title("Training Loss")
        ax[1].clear(); ax[1].plot(val_accuracies); ax[1].set_title("Validation Accuracy")
        ax[2].clear(); ax[2].plot(val_f1s); ax[2].set_title("Validation F1 Score")
        plt.pause(0.01)

        print(f"Epoch {epoch + 1}: Loss={total_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), "best_custom_fnn_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    plt.ioff()
    plt.show()
    print(f"Best validation F1: {best_val_f1:.4f}")
