"""
Day 4: LSTM and Transformer-based Sentiment Classification with PyTorch

Builds on the preprocessed data from day 1 and TF-IDF baseline from day 3.
Implements an LSTM model and a simple Transformer encoder for comparison.
Uses a synthetic dataset simulating Amazon product review sentiment.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import json
from collections import Counter
import math

# ─── reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── synthetic data (mirrors day-1 preprocessing output) ────────────────────
def make_review_dataset(n=2000):
    positive = [
        "absolutely love this product great quality",
        "works perfectly exceeded my expectations highly recommend",
        "amazing value for money will buy again",
        "fantastic build quality very satisfied with purchase",
        "excellent product fast shipping great seller",
        "best purchase I made this year love it",
        "superb quality arrived quickly very happy",
        "outstanding product does exactly what it says",
        "really impressed with the quality and packaging",
        "five stars without hesitation perfect in every way",
    ]
    negative = [
        "terrible quality broke after one day waste of money",
        "very disappointed does not work as described",
        "poor build quality stopped working within a week",
        "completely useless return requested immediately",
        "worst product ever bought total scam avoid",
        "cheap materials broke immediately do not buy",
        "absolute garbage nothing like the description",
        "fell apart on first use very dissatisfied",
        "does not match photos quality is shocking",
        "horrible experience product arrived damaged",
    ]
    neutral = [
        "product is okay nothing special average quality",
        "decent for the price but nothing exceptional",
        "arrived on time packaging was acceptable",
        "works as expected not great not terrible",
        "mediocre product does the job I suppose",
    ]
    reviews, labels = [], []
    for _ in range(n):
        r = np.random.random()
        if r < 0.45:
            reviews.append(np.random.choice(positive))
            labels.append("positive")
        elif r < 0.80:
            reviews.append(np.random.choice(negative))
            labels.append("negative")
        else:
            reviews.append(np.random.choice(neutral))
            labels.append("neutral")
    return pd.DataFrame({"review": reviews, "sentiment": labels})


df = make_review_dataset(2000)
print(f"Dataset shape: {df.shape}")
print(df["sentiment"].value_counts())

# ─── tokenization ────────────────────────────────────────────────────────────
def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


all_tokens = [tok for review in df["review"] for tok in basic_tokenize(review)]
freq = Counter(all_tokens)
vocab = ["<PAD>", "<UNK>"] + [w for w, c in freq.most_common() if c >= 2]
word2idx = {w: i for i, w in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
PAD_IDX = 0
MAX_LEN = 20
print(f"Vocabulary size: {VOCAB_SIZE}")


def encode(text, max_len=MAX_LEN):
    tokens = basic_tokenize(text)
    ids = [word2idx.get(t, 1) for t in tokens][:max_len]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])
NUM_CLASSES = len(le.classes_)
print(f"Classes: {le.classes_}")

X = np.array([encode(r) for r in df["review"]])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = ReviewDataset(X_train, y_train)
test_ds = ReviewDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)


# ─── LSTM Model ──────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        emb = self.dropout(self.embedding(x))          # (B, L, E)
        out, (hidden, _) = self.lstm(emb)
        # concat last forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden))


# ─── Transformer Model ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # build padding mask: True where token is PAD
        pad_mask = (x == PAD_IDX)
        emb = self.pos_enc(self.embedding(x))
        out = self.transformer(emb, src_key_padding_mask=pad_mask)
        # mean pooling over non-pad positions
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.dropout(pooled))


# ─── training helpers ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds.extend(logits.argmax(1).cpu().numpy())
            truths.extend(y_batch.numpy())
    return np.array(preds), np.array(truths)


def run_training(model, epochs=10, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    best_acc = 0
    history = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_preds, val_truths = evaluate(model, test_loader)
        val_acc = accuracy_score(val_truths, val_preds)
        scheduler.step(1 - val_acc)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d} | loss {tr_loss:.4f} | train acc {tr_acc:.3f} | val acc {val_acc:.3f}")
    return best_acc, history


# ─── run experiments ──────────────────────────────────────────────────────────
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 10

print("\n--- LSTM Classifier (Bidirectional, 2 layers) ---")
lstm_model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
lstm_best, lstm_history = run_training(lstm_model, epochs=EPOCHS)
lstm_preds, lstm_truths = evaluate(lstm_model.to(DEVICE), test_loader)
print(f"\nLSTM Best Val Accuracy: {lstm_best:.4f}")
print(classification_report(lstm_truths, lstm_preds, target_names=le.classes_))

print("\n--- Transformer Classifier (2 layers, 4 heads) ---")
transformer_model = TransformerClassifier(VOCAB_SIZE, EMBED_DIM, nhead=4, num_layers=2, num_classes=NUM_CLASSES)
trans_best, trans_history = run_training(transformer_model, epochs=EPOCHS)
trans_preds, trans_truths = evaluate(transformer_model.to(DEVICE), test_loader)
print(f"\nTransformer Best Val Accuracy: {trans_best:.4f}")
print(classification_report(trans_truths, trans_preds, target_names=le.classes_))

# ─── results summary ─────────────────────────────────────────────────────────
results = {
    "lstm": {
        "best_val_accuracy": round(lstm_best, 4),
        "architecture": "BiLSTM 2-layer, embed=64, hidden=128",
        "params": sum(p.numel() for p in lstm_model.parameters()),
    },
    "transformer": {
        "best_val_accuracy": round(trans_best, 4),
        "architecture": "TransformerEncoder 2-layer 4-head, embed=64",
        "params": sum(p.numel() for p in transformer_model.parameters()),
    },
}

print("\n=== Model Comparison ===")
for name, info in results.items():
    print(f"{name.upper()}: acc={info['best_val_accuracy']}, params={info['params']:,}")

# save results for day 5 final comparison
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to model_results.json for day 5 final report.")
