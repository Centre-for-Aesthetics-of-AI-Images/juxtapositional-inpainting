def download_models(config):
    import torch
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=config.model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )

    DiffusionPipeline.from_pretrained(config.model_dir, torch_dtype=torch.bfloat16)


def train(config):
    import subprocess

    from accelerate.utils import write_basic_config  # type: ignore

    # load data locally
    img_path = config.data_dir

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # define the training prompt
    prompt = f"{config.instance_prompt}".strip()

    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:  # type: ignore
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("launching dreambooth training script")

    # Override batch size and gradient accumulation for memory reasons
    model_dir = config.model_dir
    output_dir = config.model_dir / config.lora_name
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "./train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={model_dir}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={output_dir}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",  # Keep original resolution for now
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
            "--gradient_checkpointing",  # Trade compute for memory
        ]
        + (
            [
                "--report_to=wandb",
                # validation output tracking is useful, but currently broken for Flux LoRA training
                # f"--validation_prompt={prompt} in space",  # simple test prompt
                # f"--validation_epochs={config.max_train_steps // 5}",
            ]
            if config.use_wandb
            else []
        ),
    )


def finetune(config):
    """Finetune an image generation model using LoRA and Dreambooth.
    Args:
        config (FinetuneConfig): Configuration for the finetuning process.
    """
    print(f"Downloading base model to {config.model_dir}...")
    download_models(config)
    print(f"Starting the finetuning process with data from {config.data_dir}...")
    train(config)
    print(
        f"Finetuning completed successfully. The model is saved in {config.model_dir}/{config.lora_name}."
    )
