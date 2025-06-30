import os
import modal

app = modal.App(name="finetune_juxtapositional_inpainting")
INPUT_DIR = "./data/input"

def get_finetuning_image():
    # specify the commit to fetch for diffusers for reproducibility
    GIT_SHA = "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  
    return modal.Image.debian_slim(python_version="3.10"
    ).pip_install(
        "accelerate==0.31.0",
        "datasets~=2.13.0",
        "fastapi[standard]==0.115.4",
        "ftfy~=6.1.0",
        "gradio~=5.5.0",
        "huggingface-hub==0.26.2",
        "hf_transfer==0.1.8",
        "numpy<2",
        "pathlib~=1.0.0",
        "peft==0.11.1",
        "pydantic==2.9.2",
        "sentencepiece>=0.1.91,!=0.1.92",
        "smart_open~=6.4.0",
        "starlette==0.41.2",
        "transformers~=4.41.2",
        "torch~=2.2.0",
        "torchvision~=0.16",
        "triton~=2.2.0",
        "wandb==0.17.6",
    ).apt_install(
        "git"
    ).run_commands(
        # Perform a shallow fetch of just the target `diffusers` commit, checking out
        # the commit in the container's home directory, /root. Then install `diffusers`
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    ).env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    ).add_local_python_source("src").add_local_dir(INPUT_DIR, remote_path="/root/data/input")

finetuning_image = get_finetuning_image()

def get_inpainting_image():
    modal.Image.debian_slim(python_version="3.10"
    ).pip_install(
        "accelerate",  
        "diffusers",  
        "transformers", 
        "torch",       
        "torchvision", 
        "pillow",      
        "peft",        
        "huggingface-hub",
        "hf_transfer",
        "numpy<2",
        "pydantic",
        "sentencepiece>=0.1.91,!=0.1.92",
        "triton",      # Keep triton, might be needed
        # fastapi/starlette kept in case a web endpoint is added later
        "fastapi[standard]",
        "starlette",
        "ftfy",
    ).env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )

inpainting_image = get_inpainting_image()

def get_volume():
    return modal.Volume.from_name(
        "finetune_juxtapositional_inpainting_volume", create_if_missing=True
    )

volume = get_volume()

def get_secrets():
    return [modal.Secret.from_name(
        "huggingface-secret", required_keys=["HF_TOKEN"]
    )] + (
       [modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])]
        if os.getenv("USE_WANDB")
        else []
    )

GPU_SPECS = "A100-80GB"
