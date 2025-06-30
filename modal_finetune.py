import modal
from pathlib import Path
import src.modal_remote.setup as modal_setup
import src.modal_remote.finetuning as modal_finetuning
from src.config import FinetuneConfig

app = modal_setup.app

@app.local_entrypoint()
def jip_finetune(
    instance_prompt: str = "A postcard of Aarhus the Danish city",
    input_data_dir: str = "./data/input",
    use_wandb: bool = False,
    model_name: str = "black-forest-labs/FLUX.1-schnell",
    lora_name: str = "postcard_lora",
    resolution: int = 512,
    train_batch_size: int = 2,
    rank: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 4e-4,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    max_train_steps: int = 1000,
    checkpointing_steps: int = 1000,
    seed: int = 117,
):
    """
    Finetune an image generation model using LoRA and Dreambooth.

    Args:
        instance_prompt: Prompt for the instance images.
        input_data_dir: The directory containing the input data.
        use_wandb: Whether to use Weights & Biases for logging.
        model_name: The huggingface-name of the model to finetune.
        lora_name: The name to save the LoRA model as.
        resolution: The resolution of generated images.
        train_batch_size: Batch size for training.
        rank: LoRA rank.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        learning_rate: Learning rate.
        lr_scheduler: Learning rate scheduler.
        lr_warmup_steps: Number of warmup steps for learning rate.
        max_train_steps: Maximum number of training steps.
        checkpointing_steps: Number of steps between checkpoints.
        seed: Random seed.
    """
    print("Entered function")
    config = FinetuneConfig(
        data_dir=input_data_dir,
        use_wandb=use_wandb,
        model_name=model_name,
        lora_name=lora_name,
        resolution=resolution,
        train_batch_size=train_batch_size,
        rank=rank,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        max_train_steps=max_train_steps,
        checkpointing_steps=checkpointing_steps,
        seed=seed,
        instance_prompt=instance_prompt
    )
    modal_finetuning.finetune(config)

