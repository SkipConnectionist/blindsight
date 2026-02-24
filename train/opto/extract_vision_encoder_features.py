from argparse import ArgumentParser
from pathlib import Path
from opto.data.intersection import IntersectionDataset
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    args = parser.parse_args()

    dataset = IntersectionDataset(args.image_dir)
    print(f"Dataset size: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct',
        device_map='cuda',
        trust_remote_code=True,
        torch_dtype='auto',
        _attn_implementation='flash_attention_2'
    )
    processor = AutoProcessor.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct',
        trust_remote_code=True,
        num_crops=1
    )

    with torch.no_grad():
        features = []
        labels = []

        for datum in tqdm(dataset):
            img = datum['image']
            labels.append(datum['intersects'])
            output = model.model.vision_embed_tokens.img_processor(
                processor.image_processor(img)['pixel_values'][:, 0].to('cuda'),
                output_hidden_states=True
            )
            features.append(output.pooler_output.squeeze().cpu())

    torch.save(
        {
            'features': torch.stack(features),
            'labels': torch.as_tensor(labels).long()
        },
        args.output_features
    )
    print(f"Saved features and labels to {args.output_features}")


if __name__ == '__main__':
    main()
