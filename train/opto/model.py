from typing import Callable
from transformers import AutoModelForCausalLM, AutoProcessor
from functools import cache, partial

import torch


class ModelContext:

    def __init__(
        self,
        model_loader: Callable[[], torch.nn.Module],
        processor_loader: Callable[[], torch.nn.Module]
    ):
        self.model_loader = model_loader
        self.processor_loader = processor_loader

    @cache
    def model(self):
        return self.model_loader()

    @cache
    def processor(self):
        return self.processor_loader()


Phi_3_5_Vision_Instruct = ModelContext(
    model_loader=partial(
        AutoModelForCausalLM.from_pretrained,
        'microsoft/Phi-3.5-vision-instruct',
        device_map='cuda',
        trust_remote_code=True,
        torch_dtype='auto',
        _attn_implementation='flash_attention_2'
    ),
    processor_loader=partial(
        AutoProcessor.from_pretrained,
        'microsoft/Phi-3.5-vision-instruct',
        trust_remote_code=True,
        num_crops=1
    )
)
