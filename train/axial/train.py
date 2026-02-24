
import torch
import torch.nn as nn

from logging import getLogger
from typing import Protocol, TypedDict, Sequence, Any

from axial.utils.format import human_readable_number
from axial.config.core import LossFunction
from axial.data.typing import SizedIterable

logger = getLogger('train')

type Optimizer = torch.optim.Optimizer
type LearningRateScheduler = torch.optim.lr_scheduler.LRScheduler


class TrainingState(TypedDict):
    step_index: int
    epoch_index: int
    batch: dict[str, torch.Tensor]
    output: Any
    loss: float
    avg_loss: float | None
    current_lr: float
    model: nn.Module
    optimizer: Optimizer
    scheduler: LearningRateScheduler | None


class StepCallback(Protocol):

    def __call__(self, state: TrainingState) -> None:
        ...


class TrainerRestorationState(TypedDict):
    step_index: int
    epoch_index: int


def make_data_iter(
    num_epochs: int,
    data_source: SizedIterable,
    restoration_state: TrainerRestorationState | None = None
):
    if restoration_state is None:
        restoration_state = {'step_index': 0, 'epoch_index': 0}

    epoch_offset = restoration_state['epoch_index']
    step_idx = restoration_state['step_index']

    for epoch_idx in range(epoch_offset, num_epochs):
        logger.info(f'Starting epoch {epoch_idx + 1}')
        for batch in data_source:
            yield epoch_idx, step_idx, batch
            step_idx += 1


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def get_learning_rate(optimizer: Optimizer) -> float:
    return optimizer.param_groups[0]['lr']


def train(
    model: nn.Module,
    data_source: SizedIterable,
    loss_fn: LossFunction,
    optimizer: Optimizer,
    num_epochs: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    gradient_accumulation_steps: int = 1,
    max_gradient_norm: float | None = None,
    log_step_count: int = 100,
    log_loss_alpha: float = 0.9,
    post_step_callbacks: Sequence[StepCallback] = (),
    restoration_state: TrainerRestorationState | None = None
):
    model.train()

    trainable_params = get_trainable_params(model)
    num_trainable_params = sum(p.numel() for p in trainable_params)

    logger.info(f'Training model: {model.__class__.__name__}')
    logger.info(f'Model has {human_readable_number(num_trainable_params)} trainable parameters')
    logger.info(f'Using {gradient_accumulation_steps} gradient accumulation steps')
    logger.info(f'Using max gradient norm: {max_gradient_norm}' if max_gradient_norm else 'No gradient clipping')
    logger.info(f'Using optimizer: {optimizer.__class__.__name__}')
    logger.info(f'The model will be trained for {num_epochs} epochs')

    avg_loss: float | None = None
    data_iter = make_data_iter(
        num_epochs=num_epochs,
        data_source=data_source,
        restoration_state=restoration_state
    )

    for epoch_idx, step_idx, batch in data_iter:
        # Forward pass
        output = model(**batch['inputs'])

        # Compute loss
        loss = loss_fn(output=output, batch=batch)
        avg_loss = (
            log_loss_alpha * avg_loss + (1 - log_loss_alpha) * loss.item()
            if avg_loss is not None else
            loss.item()
        )

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every gradient_accumulation_steps
        if (step_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_gradient_norm
                )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Scheduler step (assuming per-step scheduling)
            if lr_scheduler is not None:
                lr_scheduler.step()

            if step_idx % log_step_count == 0:
                logger.info(
                    f'Step {(step_idx + 1):6d} | '
                    f'Loss: {avg_loss:g} | '
                    f'LR: {get_learning_rate(optimizer):g}'
                )

        # Invoke post-step callbacks
        if post_step_callbacks:
            state = TrainingState(
                step_index=step_idx,
                epoch_index=epoch_idx,
                batch=batch,
                output=output,
                loss=loss.item(),
                avg_loss=avg_loss,
                current_lr=get_learning_rate(optimizer),
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler
            )
            for callback in post_step_callbacks:
                callback(state)
