from dataclasses import dataclass

@dataclass
class FinetuneConfig:
    data_dir: str = "data"
    model_dir: str = "model"
    use_wandb: bool = False
    
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-dev"
    lora_name: str = "postcard_lora"
    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 2
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1000
    checkpointing_steps: int = 1000
    seed: int = 117
    instance_prompt: str = "A postcard of Aarhus the Danish city"


