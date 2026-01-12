"""
Unified training script for the three-stage extraction pipeline.

Stage 1: Union Detection (contrastive)
Stage 2: Affiliation Classification (contrastive centroids)
Stage 3: Designation Extraction (pointer network)

Usage:
    python train.py              # Train all stages
    python train.py --stage 1    # Train only union detector
    python train.py --stage 2    # Train only affiliation classifier
    python train.py --stage 3    # Train only designation extractor
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from labor_union_parser.char_cnn import (
    CharacterCNN,
    get_special_token_id,
    tokenize_to_chars,
)
from labor_union_parser.extractor import (
    DesignationExtractor,
    create_desig_label,
    extract_desig_from_pred,
)

sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
DATA_DIR = Path(__file__).parent / "data"

MAX_TOKENS = 70


def load_trained_char_cnn():
    """Load the trained CharCNN from model weights."""
    char_cnn = CharacterCNN(embed_dim=64, char_embed_dim=16)
    weights_path = WEIGHTS_DIR / "char_cnn.pt"

    if weights_path.exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        char_cnn_state = {}
        for k, v in state["model_state_dict"].items():
            if k.startswith("char_cnn."):
                char_cnn_state[k[len("char_cnn.") :]] = v
        char_cnn.load_state_dict(char_cnn_state)
        print("  Loaded trained CharCNN weights")
    else:
        print("  WARNING: No trained weights found, using random initialization")

    return char_cnn


# =============================================================================
# Shared Encoder Architecture
# =============================================================================


class CrossAttentionEncoder(nn.Module):
    """Encoder with cross-attention pooling instead of mean pooling.

    Uses a learned query to attend over token embeddings, allowing the model
    to learn which tokens are most relevant for classification.
    """

    def __init__(
        self,
        char_cnn: CharacterCNN,
        embed_dim: int = 64,
        num_embed_dim: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        self.char_cnn = char_cnn
        self.char_embed_dim = char_cnn.embed_dim
        self.num_embed_dim = num_embed_dim
        self.input_dim = self.char_embed_dim + num_embed_dim

        # is_number embedding (0 = not number, 1 = number)
        self.num_embed = nn.Embedding(2, num_embed_dim)

        # Learned query for "what class is this?"
        self.query = nn.Parameter(torch.randn(1, 1, self.input_dim) * 0.02)

        # Cross-attention: query attends to token sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, char_ids, token_type, is_number, return_attention=False):
        batch_size = char_ids.shape[0]

        # Get token embeddings from CharCNN
        token_emb = self.char_cnn(char_ids)

        # Add is_number embedding
        num_emb = self.num_embed(is_number)
        token_emb = torch.cat([token_emb, num_emb], dim=-1)

        # Create padding mask (True = ignore)
        key_padding_mask = token_type == 4

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Cross-attention: query attends to tokens
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=token_emb,
            value=token_emb,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        # Remove sequence dimension and project
        pooled = attn_out.squeeze(1)
        proj = self.projector(pooled)
        normalized = F.normalize(proj, p=2, dim=-1)

        if return_attention:
            return normalized, attn_weights.squeeze(1)
        return normalized


# Aliases for backwards compatibility
UnionEncoder = CrossAttentionEncoder
AffiliationEncoder = CrossAttentionEncoder


# =============================================================================
# Stage 1: Union Detection
# =============================================================================


class UnionDataset(Dataset):
    """Dataset for union vs non-union classification."""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        char_ids, _, is_number, token_type = tokenize_to_chars(
            self.texts[idx], max_tokens=MAX_TOKENS
        )
        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "token_type": torch.tensor(token_type, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "label": self.labels[idx],
        }


def one_class_contrastive_loss(embeddings, labels, temperature=0.1):
    """One-class contrastive loss: only union examples form positive pairs.

    Fixed to avoid in-place operations for autograd compatibility.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask for positive pairs (both are unions)
    labels_row = labels.unsqueeze(0)
    labels_col = labels.unsqueeze(1)
    pos_mask = ((labels_row == 1) & (labels_col == 1)).float()

    # Remove diagonal using (1 - eye) instead of fill_diagonal_
    eye = torch.eye(batch_size, device=device)
    pos_mask = pos_mask * (1 - eye)

    # Numerical stability
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max

    # Compute exp(sim) excluding self
    exp_sim = torch.exp(sim) * (1 - eye)
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Log probability
    log_prob = sim - torch.log(denom + 1e-8)

    # Average over positive pairs
    num_pos = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)

    # Only include union samples with positive pairs
    union_mask = (labels == 1) & (num_pos > 0)
    if union_mask.sum() > 0:
        loss = -mean_log_prob_pos[union_mask].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def train_union_detector(train_indices=None):
    """Train Stage 1: Union vs Non-Union detector."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training Union Detector")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")
    union_df = df[~df["aff_abbr"].isin(["UNK"])]

    # Filter to training indices if provided
    if train_indices is not None:
        union_df = union_df[union_df.index.isin(train_indices)]

    # Load non-union examples
    nonunion_path = DATA_DIR / "nonunion_examples.csv"
    nonunion_df = pd.read_csv(nonunion_path)
    nonunion_texts = nonunion_df["text"].tolist()

    # Load additional union examples if available
    additional_path = DATA_DIR / "unions_model_missed.csv"
    additional_texts = []
    if additional_path.exists():
        additional_texts = pd.read_csv(additional_path)["text"].tolist()

    # Sample union examples
    union_sample = union_df.sample(n=min(10000, len(union_df)), random_state=42)
    union_texts = union_sample["text"].tolist() + additional_texts

    print(f"  Union examples: {len(union_texts)}")
    print(f"  Non-union examples: {len(nonunion_texts)}")

    # Create dataset
    all_texts = union_texts + nonunion_texts
    all_labels = [1] * len(union_texts) + [0] * len(nonunion_texts)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_dataset = UnionDataset(train_texts, train_labels)
    test_dataset = UnionDataset(test_texts, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Build model (cross-attention encoder)
    char_cnn = load_trained_char_cnn().to(DEVICE)
    model = CrossAttentionEncoder(
        char_cnn, embed_dim=64, num_embed_dim=8, num_heads=4
    ).to(DEVICE)

    # Train - freeze char_cnn, train attention and projector
    for param in model.char_cnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.query, "lr": 1e-3},
            {"params": model.cross_attn.parameters(), "lr": 1e-3},
            {"params": model.projector.parameters(), "lr": 1e-3},
            {"params": model.num_embed.parameters(), "lr": 1e-4},
        ]
    )

    print("\n  Training...")
    model.train()
    for epoch in tqdm(range(30), desc="  Epochs"):
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            embeddings = model(char_ids, token_type, is_number)
            loss = one_class_contrastive_loss(embeddings, labels)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

    # Compute union centroid
    model.eval()
    union_embs = []
    with torch.no_grad():
        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            labels = batch["label"]
            embeddings = model(char_ids, token_type, is_number)
            for i, label in enumerate(labels.tolist()):
                if label == 1:
                    union_embs.append(embeddings[i].cpu())

    union_centroid = F.normalize(torch.stack(union_embs).mean(dim=0), p=2, dim=0).to(
        DEVICE
    )

    # Find optimal threshold
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            labels = batch["label"]
            embeddings = model(char_ids, token_type, is_number)
            sims = torch.matmul(embeddings, union_centroid.unsqueeze(0).T).squeeze(-1)
            y_true.extend(labels.tolist())
            y_scores.extend(sims.cpu().tolist())

    y_true, y_scores = np.array(y_true), np.array(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    accuracy = accuracy_score(y_true, (y_scores > optimal_threshold).astype(int))
    roc_auc = roc_auc_score(y_true, y_scores)

    print("\n  Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ROC-AUC: {roc_auc:.4f}")
    print(f"    Optimal threshold: {optimal_threshold:.4f}")

    # Save
    save_path = WEIGHTS_DIR / "union_detector.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "union_centroid": union_centroid.cpu(),
            "optimal_threshold": optimal_threshold,
        },
        save_path,
    )
    print(f"\n  Saved to {save_path}")

    return model, union_centroid, optimal_threshold


# =============================================================================
# Stage 2: Affiliation Classification
# =============================================================================


class AffiliationDataset(Dataset):
    """Dataset for affiliation classification."""

    def __init__(self, texts, affiliations):
        self.texts = texts
        self.affiliations = affiliations
        self.all_affs = list(set(affiliations))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        char_ids, _, is_number, token_type = tokenize_to_chars(
            self.texts[idx], max_tokens=MAX_TOKENS
        )
        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "token_type": torch.tensor(token_type, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "aff_idx": self.all_affs.index(self.affiliations[idx]),
        }


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss.

    Fixed to avoid in-place operations for autograd compatibility.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask for positive pairs (same label)
    labels_col = labels.unsqueeze(1)
    labels_row = labels.unsqueeze(0)
    pos_mask = (labels_col == labels_row).float()

    # Remove diagonal using (1 - eye) instead of fill_diagonal_
    eye = torch.eye(batch_size, device=device)
    pos_mask = pos_mask * (1 - eye)

    # Numerical stability
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max

    # Compute exp(sim) excluding self
    exp_sim = torch.exp(sim) * (1 - eye)
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Log probability
    log_prob = sim - torch.log(denom + 1e-8)

    # Average over positive pairs
    num_pos = pos_mask.sum(dim=1)
    loss_per_sample = -(pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)

    # Only include samples with at least one positive pair
    valid = num_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_per_sample[valid].mean()


def train_affiliation_classifier(train_indices=None):
    """Train Stage 2: Affiliation classifier."""
    print("\n" + "=" * 60)
    print("STAGE 2: Training Affiliation Classifier")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")
    # Exclude UNAFF, UNK, and MULTI (no centroid for these)
    known_df = df[~df["aff_abbr"].isin(["UNAFF", "UNK", "MULTI"])]

    # Filter to training indices if provided
    if train_indices is not None:
        known_df = known_df[known_df.index.isin(train_indices)]

    # Filter rare affiliations
    aff_counts = known_df["aff_abbr"].value_counts()
    valid_affs = aff_counts[aff_counts >= 5].index
    known_df = known_df[known_df["aff_abbr"].isin(valid_affs)]

    print(f"  Training examples: {len(known_df)}")
    print(f"  Affiliations: {known_df['aff_abbr'].nunique()}")

    # Split
    train_df, test_df = train_test_split(
        known_df, test_size=0.2, random_state=42, stratify=known_df["aff_abbr"]
    )

    train_dataset = AffiliationDataset(
        train_df["text"].tolist(), train_df["aff_abbr"].tolist()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=True
    )

    # Build model (cross-attention encoder)
    char_cnn = load_trained_char_cnn().to(DEVICE)
    model = CrossAttentionEncoder(
        char_cnn, embed_dim=64, num_embed_dim=8, num_heads=4
    ).to(DEVICE)

    # Train - freeze char_cnn, train attention and projector
    for param in model.char_cnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.query, "lr": 1e-3},
            {"params": model.cross_attn.parameters(), "lr": 1e-3},
            {"params": model.projector.parameters(), "lr": 1e-3},
            {"params": model.num_embed.parameters(), "lr": 1e-4},
        ]
    )

    print("\n  Training...")
    model.train()
    for epoch in tqdm(range(50), desc="  Epochs"):
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            aff_idx = batch["aff_idx"].to(DEVICE)

            optimizer.zero_grad()
            embeddings = model(char_ids, token_type, is_number)
            loss = supervised_contrastive_loss(embeddings, aff_idx)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

    # Compute centroids
    model.eval()
    aff_list = train_dataset.all_affs
    aff_embeddings = {i: [] for i in range(len(aff_list))}

    with torch.no_grad():
        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            aff_idx = batch["aff_idx"]
            embeddings = model(char_ids, token_type, is_number)
            for i, aff in enumerate(aff_idx.tolist()):
                aff_embeddings[aff].append(embeddings[i].cpu())

    centroids = []
    for i in range(len(aff_list)):
        if aff_embeddings[i]:
            centroid = F.normalize(
                torch.stack(aff_embeddings[i]).mean(dim=0), p=2, dim=0
            )
            centroids.append(centroid)
        else:
            centroids.append(torch.zeros(64))
    centroids = torch.stack(centroids).to(DEVICE)

    # Evaluate
    correct = 0
    total = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  Evaluating"):
        char_ids, _, is_number, token_type = tokenize_to_chars(
            row["text"], max_tokens=MAX_TOKENS
        )
        char_ids_t = torch.tensor([char_ids], dtype=torch.long, device=DEVICE)
        token_type_t = torch.tensor([token_type], dtype=torch.long, device=DEVICE)
        is_number_t = torch.tensor([is_number], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            emb = model(char_ids_t, token_type_t, is_number_t)
        similarities = torch.matmul(emb, centroids.T)
        pred_idx = similarities.argmax(dim=1).item()
        pred_aff = aff_list[pred_idx]

        if pred_aff == row["aff_abbr"]:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"\n  Classification accuracy: {accuracy:.4f}")

    # Save
    model_path = WEIGHTS_DIR / "contrastive_aff_model.pt"
    centroids_path = WEIGHTS_DIR / "contrastive_aff_centroids.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "aff_list": aff_list,
        },
        model_path,
    )
    torch.save(centroids.cpu(), centroids_path)

    print(f"\n  Saved model to {model_path}")
    print(f"  Saved centroids to {centroids_path}")

    return model, aff_list, centroids


# =============================================================================
# Stage 3: Designation Extraction
# =============================================================================


class DesignationDataset(Dataset):
    """Dataset for designation extraction."""

    def __init__(self, texts, desig_nums, affiliations, aff_list):
        self.texts = texts
        self.desig_nums = desig_nums
        self.affiliations = affiliations
        self.aff_list = aff_list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        desig_num = self.desig_nums[idx]
        aff = self.affiliations[idx]

        char_ids, tokens, is_number, token_type = tokenize_to_chars(
            text, max_tokens=MAX_TOKENS
        )

        # Get special IDs for non-word tokens
        special_ids = []
        for i, tt in enumerate(token_type):
            if tt != 0:  # Not a word
                special_ids.append(
                    get_special_token_id(tokens[i] if i < len(tokens) else "")
                )
            else:
                special_ids.append(0)

        # Create designation label
        desig_label = create_desig_label(
            text, str(desig_num) if pd.notna(desig_num) else ""
        )

        # Get affiliation index
        aff_idx = self.aff_list.index(aff) if aff in self.aff_list else 0

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "token_type": torch.tensor(token_type, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "special_ids": torch.tensor(special_ids, dtype=torch.long),
            "token_mask": torch.tensor(
                [1 if tt != 4 else 0 for tt in token_type], dtype=torch.long
            ),
            "aff_idx": aff_idx,
            "desig_label": desig_label,
            "text": text,
            "desig_num": str(desig_num) if pd.notna(desig_num) else "",
        }


def train_designation_extractor(aff_list, train_indices=None):
    """Train Stage 3: Designation extractor."""
    print("\n" + "=" * 60)
    print("STAGE 3: Training Designation Extractor")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")

    # Filter to known affiliations (not UNAFF/UNK) that are in our aff_list
    known_df = df[df["aff_abbr"].isin(aff_list)]

    # Filter to training indices if provided
    if train_indices is not None:
        known_df = known_df[known_df.index.isin(train_indices)]

    # Filter to examples with designation numbers
    has_desig = known_df["desig_num"].notna()
    print(f"  Examples with designation: {has_desig.sum()}")
    print(f"  Examples without designation: {(~has_desig).sum()}")

    # Use all examples (both with and without designation)
    train_df, test_df = train_test_split(
        known_df, test_size=0.2, random_state=42, stratify=known_df["aff_abbr"]
    )

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    train_dataset = DesignationDataset(
        train_df["text"].tolist(),
        train_df["desig_num"].tolist(),
        train_df["aff_abbr"].tolist(),
        aff_list,
    )

    test_dataset = DesignationDataset(
        test_df["text"].tolist(),
        test_df["desig_num"].tolist(),
        test_df["aff_abbr"].tolist(),
        aff_list,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Build model
    model = DesignationExtractor(
        num_affs=len(aff_list),
        token_embed_dim=64,
        hidden_dim=512,
        aff_embed_dim=64,
        num_attn_heads=4,
        num_feature_dim=16,
        char_embed_dim=16,
        num_attn_layers=3,
    ).to(DEVICE)

    # Load CharCNN weights
    weights_path = WEIGHTS_DIR / "char_cnn.pt"
    if weights_path.exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        char_cnn_state = {}
        for k, v in state["model_state_dict"].items():
            if k.startswith("char_cnn."):
                char_cnn_state[k[len("char_cnn.") :]] = v
        model.char_cnn.load_state_dict(char_cnn_state)
        print("  Loaded CharCNN weights")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("\n  Training...")
    model.train()
    for epoch in tqdm(range(10), desc="  Epochs"):
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_mask = batch["token_mask"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            special_ids = batch["special_ids"].to(DEVICE)
            aff_idx = batch["aff_idx"].to(DEVICE)
            desig_labels = batch["desig_label"].to(DEVICE)

            optimizer.zero_grad()
            results = model(
                char_ids,
                token_mask,
                is_number,
                token_type,
                special_ids,
                aff_idx,
                desig_labels,
            )

            loss = results["desig_loss"]
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            tqdm.write(f"    Epoch {epoch+1}: loss = {avg_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Evaluating"):
            char_ids = batch["char_ids"].to(DEVICE)
            token_mask = batch["token_mask"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            special_ids = batch["special_ids"].to(DEVICE)
            aff_idx = batch["aff_idx"].to(DEVICE)

            results = model(
                char_ids, token_mask, is_number, token_type, special_ids, aff_idx
            )
            preds = results["desig_pred"].cpu().tolist()

            for i, pred in enumerate(preds):
                text = batch["text"][i]
                true_desig = batch["desig_num"][i]
                pred_desig = extract_desig_from_pred(text, pred)

                # Normalize for comparison
                true_norm = str(true_desig).lstrip("0") if true_desig else ""
                if "." in true_norm:
                    true_norm = true_norm.split(".")[0]
                pred_norm = pred_desig.lstrip("0") if pred_desig else ""

                if true_norm == pred_norm:
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"\n  Designation accuracy: {accuracy:.4f}")

    # Save
    save_path = WEIGHTS_DIR / "designation_extractor.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "aff_list": aff_list,
        },
        save_path,
    )
    print(f"\n  Saved to {save_path}")

    return model


# =============================================================================
# Main
# =============================================================================


def create_held_out_test_set():
    """Create a held-out test set before any training and save to CSV."""
    print("\n" + "=" * 60)
    print("CREATING HELD-OUT TEST SET")
    print("=" * 60)

    # Load all data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")
    known_df = df[~df["aff_abbr"].isin(["UNK"])]

    # Filter out classes with < 2 members (can't stratify)
    aff_counts = known_df["aff_abbr"].value_counts()
    valid_affs = aff_counts[aff_counts >= 2].index
    stratifiable_df = known_df[known_df["aff_abbr"].isin(valid_affs)]
    unstratifiable_df = known_df[~known_df["aff_abbr"].isin(valid_affs)]

    # Stratified split on stratifiable data
    train_strat, test_strat = train_test_split(
        stratifiable_df,
        test_size=0.05,
        random_state=99,
        stratify=stratifiable_df["aff_abbr"],
    )

    # Add unstratifiable to train (too rare for test)
    train_df = pd.concat([train_strat, unstratifiable_df])
    test_df = test_strat

    print(f"  Total known examples: {len(known_df)}")
    print(f"  Training pool: {len(train_df)}")
    print(f"  Held-out test: {len(test_df)}")

    # Update split column in the dataframe and save
    df["split"] = "train"  # Default
    df.loc[test_df.index, "split"] = "test"
    df.loc[df["aff_abbr"] == "UNK", "split"] = "exclude"
    df.to_csv(DATA_DIR / "labeled_data.csv", index=False)
    print(f"  Saved split to {DATA_DIR / 'labeled_data.csv'}")

    # Return indices for use in training
    train_indices = set(train_df.index.tolist())

    return train_indices, test_df


def evaluate_pipeline(test_df):
    """Evaluate the full pipeline on held-out test data."""
    print("\n" + "=" * 60)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 60)

    # Import here to get freshly trained weights
    from labor_union_parser.extractor import Extractor

    extractor = Extractor()
    print(f"  Stage 3 learned model: {extractor._use_learned_desig}")
    print(f"  Affiliations: {len(extractor.aff_list)}")
    print(f"  Test examples: {len(test_df)}")

    # Run extraction
    results = list(
        extractor.extract_all(
            test_df["text"].tolist(), batch_size=256, show_progress=True
        )
    )

    # Evaluate
    union_detected = 0
    aff_correct = 0
    aff_total = 0
    desig_correct = 0
    desig_total = 0

    aff_errors = []
    desig_errors = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        result = results[i]
        true_aff = row["aff_abbr"]
        true_desig = (
            str(row["desig_num"]).split(".")[0] if pd.notna(row["desig_num"]) else ""
        )
        true_desig = true_desig.lstrip("0") or ""

        if result["is_union"]:
            union_detected += 1

            # Affiliation accuracy (exclude UNAFF and MULTI from ground truth)
            if true_aff not in ("UNAFF", "MULTI"):
                aff_total += 1
                pred_aff = result["affiliation"]
                if pred_aff == true_aff:
                    aff_correct += 1
                else:
                    aff_errors.append(
                        (row["text"][:60], true_aff, pred_aff, result["aff_score"])
                    )

            # Designation accuracy
            if true_desig:
                desig_total += 1
                pred_desig = (
                    result["designation"].lstrip("0") if result["designation"] else ""
                )
                if pred_desig == true_desig:
                    desig_correct += 1
                else:
                    desig_errors.append((row["text"][:60], true_desig, pred_desig))

    print("\n  RESULTS:")
    print(
        f"    Union detection: {union_detected}/{len(test_df)} ({100*union_detected/len(test_df):.1f}%)"
    )
    print(
        f"    Affiliation accuracy: {aff_correct}/{aff_total} ({100*aff_correct/aff_total:.1f}%)"
    )
    print(
        f"    Designation accuracy: {desig_correct}/{desig_total} ({100*desig_correct/desig_total:.1f}%)"
    )

    print("\n  Sample affiliation errors:")
    for text, true, pred, sim in aff_errors[:5]:
        print(f"    {text}...")
        print(f"      True: {true}, Pred: {pred}, Sim: {sim:.3f}")

    print("\n  Sample designation errors:")
    for text, true, pred in desig_errors[:5]:
        print(f"    {text}...")
        print(f"      True: {true}, Pred: {pred}")

    return {
        "union_detection": union_detected / len(test_df),
        "affiliation_accuracy": aff_correct / aff_total,
        "designation_accuracy": desig_correct / desig_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive extraction pipeline"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        help="Train only specific stage (1=union, 2=affiliation, 3=designation)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CONTRASTIVE EXTRACTION PIPELINE TRAINING")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Weights directory: {WEIGHTS_DIR}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == 3:
        # Train only designation extractor - load aff_list from saved model
        aff_model_path = WEIGHTS_DIR / "contrastive_aff_model.pt"
        if not aff_model_path.exists():
            print("ERROR: No affiliation model found. Run stage 2 first.")
            sys.exit(1)
        state = torch.load(aff_model_path, map_location="cpu", weights_only=False)
        aff_list = state["aff_list"]
        print(
            f"\nLoaded aff_list with {len(aff_list)} affiliations from existing model"
        )
        train_designation_extractor(aff_list, train_indices=None)
        return

    if args.stage == 2:
        # Train only affiliation classifier
        train_affiliation_classifier(train_indices=None)
        return

    if args.stage == 1:
        # Train only union detector
        train_union_detector(train_indices=None)
        return

    # Full pipeline training
    # Create held-out test set BEFORE any training
    train_indices, test_df = create_held_out_test_set()

    # Stage 1
    train_union_detector(train_indices)

    # Stage 2
    _, aff_list, _ = train_affiliation_classifier(train_indices)

    # Stage 3
    train_designation_extractor(aff_list, train_indices)

    # Final evaluation on held-out test set
    metrics = evaluate_pipeline(test_df)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nAll weights saved to {WEIGHTS_DIR}")
    print("\nFinal metrics on held-out test set:")
    print(f"  Union detection:      {100*metrics['union_detection']:.1f}%")
    print(f"  Affiliation accuracy: {100*metrics['affiliation_accuracy']:.1f}%")
    print(f"  Designation accuracy: {100*metrics['designation_accuracy']:.1f}%")
    print("\nTo use the trained models:")
    print("  from labor_union_parser import Extractor")
    print("  extractor = Extractor()")
    print("  result = extractor.extract('SEIU Local 1199')")


if __name__ == "__main__":
    main()
