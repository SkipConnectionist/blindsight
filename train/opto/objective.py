import torch.nn.functional as F


def compute_tail_cross_entropy(output, batch):
    return F.cross_entropy(
        output.logits[:, -1],
        batch['targets'].squeeze(-1)
    )


def compute_accuracy(logits, targets):
    correct = logits[:, -1].argmax(dim=-1) == targets.squeeze(-1)
    return correct.float().mean()
