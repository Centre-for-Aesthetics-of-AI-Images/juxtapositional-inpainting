import src.modal_remote.setup as modal_setup
from src.config import FinetuneConfig

MODEL_DIR = "/root/model"  # Directory where the model will be downloaded and trained
app = modal_setup.app


@app.function(
    volumes={MODEL_DIR: modal_setup.volume},
    image=modal_setup.finetuning_image,
    secrets=modal_setup.get_secrets(),
    timeout=600,  # 10 minutes
)
def download_models(config):
    import torch
    from diffusers import (
        DiffusionPipeline,  # type: ignore (relic from the version of diffusers used in the modal image)
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )

    DiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    # Persist the downloaded model files
    modal_setup.volume.commit()


@app.function(
    image=modal_setup.finetuning_image,
    gpu=modal_setup.GPU_SPECS,
    volumes={MODEL_DIR: modal_setup.volume},  # stores fine-tuned model
    timeout=18000,  # 300 minutes
    secrets=modal_setup.get_secrets(),
)
def train(config: FinetuneConfig):
    import subprocess

    from accelerate.utils import write_basic_config  # type: ignore

    # load data locally
    img_path = modal_setup.INPUT_DIR

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
    model_dir = MODEL_DIR
    output_dir = MODEL_DIR + "/" + config.lora_name
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_flux.py",
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

    modal_setup.volume.commit()


def finetune(config: FinetuneConfig):
    """
    Finetune an image generation model using LoRA and Dreambooth.
    This function orchestrates the downloading of models and the training process.
    """

    print("Starting model download...")
    download_models.remote(config)
    print("Starting finetuning process...")
    train.remote(config)
