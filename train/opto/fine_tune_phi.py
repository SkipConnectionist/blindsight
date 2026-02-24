from typing import Iterable
from tqdm import tqdm
from argparse import ArgumentParser
from functools import wraps, partial

from axial.config import configs
from axial.logging import setup_logging
from axial.checkpoint import Checkpointer
from axial.utils.format import human_readable_number
from axial.utils.platform import get_platform_default_device
from axial.data.util import MapToDevice
from axial.data.typing import SizedIterable
from axial.train import (train,
                         get_trainable_params,
                         TrainingState,
                         TrainerRestorationState,
                         StepCallback)

from opto import config as _
from opto.objective import compute_accuracy

import torch
import logging

logger = logging.getLogger('Train')


def as_periodic_callback(
    callback: StepCallback,
    # Invoked after every `period` steps.
    period: int,
    # If True, the callback is also invoked on the first step.
    include_first_step: bool = False
) -> StepCallback:

    @wraps(callback)
    def periodic_callback(state: TrainingState) -> None:
        step_idx = state['step_index']
        if ((step_idx == 0) and include_first_step) or ((step_idx + 1) % period) == 0:
            callback(state)

    return periodic_callback


def write_checkpoint_callback(state: TrainingState, checkpointer: Checkpointer) -> None:
    checkpointer.save(
        state_dict=state['model'].state_dict(),
        optimizer_state=state['optimizer'].state_dict(),
        scheduler_state=state['scheduler'].state_dict() if state['scheduler'] else None,
        step_index=state['step_index'],
        epoch_index=state['epoch_index'],
        metadata={
            'loss': state['loss'],
            'avg_loss': state.get('avg_loss', None),
            'current_lr': state['current_lr']
        }
    )


def compute_mean_training_accuracy(state: TrainingState) -> float:
    return (
        compute_accuracy(
            logits=state['output'].logits,
            targets=state['batch']['targets']
        )
        .item()
    )


def log_accuracy_callback(state: TrainingState) -> None:
    mean_accuracy = compute_mean_training_accuracy(state)
    logger.info(
        f'Step {(state["step_index"] + 1):6d} | '
        f'Mean Accuracy: {mean_accuracy:g}'
    )


def log_data_details(**sources: SizedIterable):
    for name, source in sources.items():
        logger.info(f'{name.title()} data source has size: {human_readable_number(len(source))} [batches]')


@torch.no_grad
def validate_model(
    model,
    data_source: Iterable,
    step: int
):
    logger.info('Validating model...')

    logits = []
    targets = []
    for batch in tqdm(data_source):
        output = model(**batch['inputs'])
        logits.append(output.logits.cpu())
        targets.append(batch['targets'].cpu())

    accuracy = compute_accuracy(
        logits=torch.cat(logits, dim=0),
        targets=torch.cat(targets, dim=0)
    )

    logger.info(
        f'Step {(step):6d} | '
        f'Validation Mean Accuracy: {accuracy:g}'
    )


def validate_model_callback(
    state: TrainingState,
    data_source: Iterable,
) -> None:
    validate_model(
        model=state['model'],
        data_source=data_source,
        step=state['step_index'] + 1,
    )


def parse_args():
    parser = ArgumentParser(description='Fine tune the vision encoder of Phi-3.5 on the intersection task.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration to use for training'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=get_platform_default_device(),
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run training on.'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory to save logs (default: none)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints (default: none)'
    )
    parser.add_argument(
        '--restore',
        type=str,
        default=None,
        help='Path to a checkpoint to restore from (default: none)'
    )
    parser.add_argument(
        '--pre-validate',
        action='store_true',
        default=False,
        help='Run validation before starting training (default: false)'
    )
    return parser.parse_args()


def run():
    args = parse_args()

    setup_logging(
        log_directory=args.log_dir
    )

    # Get the configuration source
    try:
        cfg_source = configs[args.config]
    except KeyError:
        print(
            f'Unknown configuration: {args.config}. Available configurations: {list(configs.keys())}'
        )
        exit(-1)

    # Create the config
    cfg = cfg_source()

    # Move model to the specified device
    device = args.device
    logger.info(f'Using device: {device}')
    model = cfg['model']().to(device)

    # Callbacks invoked periodically after each training step
    post_step_callbacks: list[StepCallback] = [
        # Periodically log training accuracy metrics
        as_periodic_callback(
            log_accuracy_callback,
            period=10,
            include_first_step=True
        )
    ]

    # Checkpointing
    if args.checkpoint_dir:
        checkpoint_prefix = f'{model.__class__.__name__}_{args.config}'
        checkpoint_dir = Checkpointer.make_checkpoint_dir(
            root=args.checkpoint_dir,
            prefix=checkpoint_prefix
        )
        checkpointer = Checkpointer(
            checkpoint_dir=checkpoint_dir,
            prefix=checkpoint_prefix,
            max_checkpoints=20
        )
        post_step_callbacks.append(
            as_periodic_callback(
                partial(write_checkpoint_callback, checkpointer=checkpointer),
                period=1000,
                include_first_step=False
            )
        )

    # Data Sources
    training_data = MapToDevice(
        cfg['training_data'](),
        device=device
    )
    validation_data = MapToDevice(
        cfg['validation_data'](),
        device=device,
    )
    log_data_details(
        training=training_data,
        validation=validation_data
    )

    # Validation
    post_step_callbacks.append(
        as_periodic_callback(
            partial(
                validate_model_callback,
                data_source=validation_data
            ),
            period=500
        )
    )

    optimizer = cfg['optimizer'](get_trainable_params(model))
    scheduler = cfg['scheduler'](optimizer) if 'scheduler' in cfg else None

    # Restore from checkpoint if specified
    restoration_state: TrainerRestorationState | None = None
    if args.restore:
        logger.info(f'Restoring model from checkpoint: {args.restore}')
        checkpoint = Checkpointer.restore(
            path=args.restore,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        restoration_state = {
            'step_index': checkpoint['step_index'],
            'epoch_index': checkpoint['epoch_index']
        }

    # Model Initialization
    model_initializer = cfg.get('model_initializer')
    if model_initializer:
        logger.info('Invoking model initializer')
        model_initializer(model)

    # Validate before training if specified
    if args.pre_validate:
        validate_model(
            model=model,
            data_source=validation_data,
            step=restoration_state['step_index'] if restoration_state else 0
        )

    # Begin training
    train(
        model=model,
        data_source=training_data,
        loss_fn=cfg['loss_fn'](),
        optimizer=optimizer,
        num_epochs=cfg.get('num_epochs', 1_000_000),
        lr_scheduler=scheduler,
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 1),
        max_gradient_norm=cfg.get('max_gradient_norm'),
        log_step_count=1,
        post_step_callbacks=post_step_callbacks,
        restoration_state=restoration_state
    )


if __name__ == '__main__':
    run()
