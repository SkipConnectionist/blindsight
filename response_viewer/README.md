# Response Viewer

Simple web server for browsing model responses against image ground truth encoded in filenames.

## What it expects

- Image files named like `intersect_Y_distractor_N_*.png`
- Response cache text files named like `<image_filename>.txt`
    - This matches `compute_accuracy.py` cache entries, e.g. `intersect_Y_...png.txt`

## Run

```bash
python3 response_viewer/app.py \
  --response-cache response_cache/.../<prompt_sha> \
  --image-dir /path/to/images \
  --host 127.0.0.1 \
  --port 8000
```

You can also pass the parent `response_cache` directory. The server will recursively discover `.txt` response files.

## UI behavior

- Green cards: correct prediction
- Red cards: incorrect prediction
- Amber cards: no parseable yes/no in response
- Top filter: `All`, `Correct`, `Incorrect`
