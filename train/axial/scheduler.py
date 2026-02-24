import torch
import math


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    min_lr_ratio: float = 0.1
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        kind: Type of scheduler ("cosine", "linear", "constant")
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR as ratio of initial LR (for cosine)

    Returns:
        Configured scheduler
    """
    if kind.lower() == "cosine":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif kind.lower() == "linear":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif kind.lower() == "constant":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unsupported scheduler type: {kind}")
