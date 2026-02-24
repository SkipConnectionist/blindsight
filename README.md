# Overview

## Blindness test generators

- These scripts generate the various test inputs mentioned in the report
- Most of these have minimal requirements: numpy and matplotlib should be sufficient to run these

## Generating intersection datasets

- Run the `generate_intersection_dataset.py` with an output directory and the number of images:

Example:

```sh
python generate_intersection_dataset.py --output_dir ~/data/AntIntersect/test/ --count 1000
```

## Computing accuracy metrics

- Use the `compute_accuracy.py` script in conjuction with synthetic dataset to compute the LLMs accuracy.
- By default, uses Claude Haiku 4.5 (expects an apikey to be present in `apikey.txt`)
- Invoke with --use-local to fetch and use Phi 3.5 Vision Instruct instead
- By default, reads the prompt from `prompt.txt`
- Caches the results in `<cwd>/response_cache` (scoped by dataset name, model name, prompt hash).

Example:

```sh
python compute_accuracy.py --input-dir ~/data/AntIntersect/test_without_grid/ --workers 4
```

## Viewing the results

- Use the web-based viewer in `response_viewer` to visually explore the LLM results
- Pass in the `response_cache` output sub-directory (created by `compute_accuracy.py`) and the corresponding images directory.

Example:

```sh
python app.py --response-cache ../response_cache/test/claude-haiku-4-5-20251001/8a09b8cfc4dd4de1/ --image-dir ~/data/AntIntersect/test/
```

## Phi 3.5 Vision Instruct: Linear Probe

- `cd train`

### Step 1: Extract vision features

```sh
python -m opto.extract_vision_encoder_features --image-dir ~/data/AntIntersect/test --output-features ~/data/AntIntersect/test_phi_vision_features.pt
```

### Step 2: Train the linear classifier

```sh
python -m opto.train_linear_probe --train-feats ~/data/AntIntersect/train_phi_vision_features.pt --val-feats ~/data/AntIntersect/test_phi_vision_features.pt --steps 100000 --batch-size 10000 --lr 1e-4 --weight-decay 1e-2
```

## Phi 3.5 Vision Instruct: Fine Tune

```sh
cd train

python -m opto.fine_tune_phi --config intersection_fine_tune_phi_rev1 --checkpoint-dir ~/data/checkpoints/
```
