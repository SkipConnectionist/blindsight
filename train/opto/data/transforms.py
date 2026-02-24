from PIL import Image
import io


def decode_image(
    datum: dict,
    input_key='image',
    output_key='image'
) -> dict:
    image_bytes = datum[input_key]
    datum[output_key] = Image.open(io.BytesIO(image_bytes))
    return datum


class ImagePrompter:

    def __init__(
        self,
        processor,
        prompt: str,
        source_key: str = 'image',
        dest_key: str = 'inputs'
    ):
        self.processor = processor
        self.source_key = source_key
        self.dest_key = dest_key
        self.prompt = processor.tokenizer.apply_chat_template(
            [
                {
                    'role': 'user',
                    'content': f'<|image_1|>\n{prompt}'
                }
            ],
            tokenize=False,
            add_generation_prompt=True
        )

    def __call__(self, datum: dict) -> dict:
        processed = self.processor(
            self.prompt,
            datum[self.source_key],
            return_tensors="pt"
        )
        # Remove singleton batch dimensions
        datum[self.dest_key] = {k: v.squeeze(0) for k, v in processed.items()}
        return datum


class TargetTokenizer:

    def __init__(
        self,
        processor,
        source_key: str = 'answer',
        dest_key: str = 'targets'
    ):
        self.processor = processor
        self.source_key = source_key
        self.dest_key = dest_key

    def __call__(self, datum: dict) -> dict:
        datum[self.dest_key] = (
            self.processor.tokenizer(
                datum[self.source_key],
                add_special_tokens=False,
                return_tensors='pt'
            )
            .input_ids
            .squeeze(0)
        )
        return datum


def select_keys(datum: dict, keys: list[str]) -> dict:
    return {k: datum[k] for k in keys}
