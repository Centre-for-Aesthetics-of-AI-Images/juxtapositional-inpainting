# Juxtapositional Inpainting

This repository implements the code to replicate the experiments performed in the paper *A Different View: Investigating Image Collections with Juxtapositional Inpainting* (in progress).

## Usage

### Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) to manage the Python environment. Install `uv` and project dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync 
```

### Data Downloading

You can scrape and download metadata and images using the local CLI script:

```bash
uv run jip_local.py download-metadata postcards --output-dir output/metadata --query Aarhus
uv run jip_local.py download-images postcards --metadata-dir output/metadata --output-dir output/images
```

Replace `postcards` with `aerial` for aerial datasets. See `python jip_local.py --help` for all options.

### Model Finetuning

Currently, model finetuning is supported via [Modal](https://modal.com/docs/guide). Use the following command to run finetuning:

```bash
uv run modal run jip_modal_finetune.py
```

You can pass arguments to customize the finetuning process. See the script for available parameters.

### Inpainting

Inpainting is also performed via Modal:

```bash
uv run modal run jip_modal_inpaint.py
```

Adjust the arguments as needed for your use case.

### Notes

- At present, only Modal-based finetuning and inpainting are supported. A local version is under development.
- For more information on using Modal, see the [Modal documentation](https://modal.com/docs/guide).
