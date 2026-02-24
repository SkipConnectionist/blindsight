from argparse import ArgumentParser
from pathlib import Path

import torch


def parse_args():
    parser = ArgumentParser(description="Train a linear probe (logistic regressor).")
    parser.add_argument(
        "--train-feats",
        type=Path,
        required=True,
        help="Path to a torch file containing train features/labels.",
    )
    parser.add_argument(
        "--val-feats",
        type=Path,
        required=True,
        help="Path to a torch file containing validation features/labels.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for SGD updates.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log training loss and validation accuracy every N steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    return parser.parse_args()


def load_features(path: Path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")

    missing_keys = {"features", "labels"} - set(payload.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in {path}: {sorted(missing_keys)}")

    features = payload["features"]
    labels = payload["labels"]

    if features.ndim != 2:
        raise ValueError(f"Expected features shape [N, D] in {path}, got {tuple(features.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape [N] in {path}, got {tuple(labels.shape)}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature/label count mismatch in {path}: {features.shape[0]} vs {labels.shape[0]}"
        )

    return features.float(), labels.long()


@torch.no_grad()
def compute_binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits > 0).long()
    return (preds == labels).float().mean().item()


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_x, train_y = load_features(args.train_feats)
    val_x, val_y = load_features(args.val_feats)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    num_train, feat_dim = train_x.shape
    model = torch.nn.Linear(feat_dim, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Train samples: {num_train}, feature dim: {feat_dim}")
    print(f"Val samples:   {val_x.shape[0]}")
    print(f"Device:        {device}")

    for step in range(1, args.steps + 1):
        batch_indices = torch.randint(0, num_train, (args.batch_size,), device=device)
        batch_x = train_x[batch_indices]
        batch_y = train_y[batch_indices].float()

        logits = model(batch_x).squeeze(1)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x).squeeze(1)
                val_acc = compute_binary_accuracy(val_logits, val_y)
            model.train()
            print(
                f"Step {step:5d}/{args.steps} | "
                f"train_loss={loss.item():.6f} | "
                f"val_acc={val_acc:.4f}"
            )


if __name__ == "__main__":
    main()
