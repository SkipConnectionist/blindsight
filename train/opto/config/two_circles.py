import torch

from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from axial.config.core import Config, register, get_host_data_path
from axial.data.util import RandomizingDataLoader, DataPipeline
from axial.data.source import NumpyDataSource
from axial.scheduler import create_scheduler

from opto.model import Phi_3_5_Vision_Instruct
from opto.data.transforms import ImagePrompter, TargetTokenizer, decode_image, select_keys
from opto.objective import compute_tail_cross_entropy


def two_circles_data_source(
    path: Path,
    batch_size,
    processor,
    prompt: str
):
    return RandomizingDataLoader(
        DataLoader(
            dataset=DataPipeline(
                NumpyDataSource[dict](path),
                decode_image,
                ImagePrompter(
                    processor=processor(),
                    prompt=prompt,
                ),
                TargetTokenizer(
                    processor=processor()
                ),
                partial(
                    select_keys,
                    keys=['inputs', 'targets']
                )

            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    )


def fine_tune_vision_projection(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for params in model.model.vision_embed_tokens.img_projection.parameters():
        params.requires_grad = True


@register
def two_circles_finetune_phi_v1():
    ctx = Phi_3_5_Vision_Instruct
    prompt = "Are the two circles touching each other? Answer with Yes/No."

    return Config(
        model=ctx.model,
        model_initializer=fine_tune_vision_projection,
        loss_fn=lambda: compute_tail_cross_entropy,
        training_data=partial(
            two_circles_data_source,
            processor=ctx.processor,
            prompt=prompt,
            path=get_host_data_path('multimodal/two_circles_train.npy'),
            batch_size=8
        ),
        validation_data=partial(
            two_circles_data_source,
            processor=ctx.processor,
            prompt=prompt,
            path=get_host_data_path('multimodal/two_circles_val.npy'),
            batch_size=10
        ),
        optimizer=partial(
            torch.optim.AdamW,
            lr=1e-4,
            weight_decay=0
        ),
        # scheduler=partial(
        #     create_scheduler,
        #     kind='constant',
        #     num_warmup_steps=500,
        #     num_training_steps=1_000_000
        # ),
        max_gradient_norm=1.0,
        gradient_accumulation_steps=1
    )
