from pathlib import Path
import modal
from src.config import InpaintingConfig
import src.modal_remote.setup as modal_setup
from PIL import Image, ImageOps
import numpy as np
import os

MODEL_DIR = "/root/model"
app = modal_setup.app

# Preprocessing strategy constants
PREPROCESS_REPLACE_NOISE = "replace_noise"
PREPROCESS_ADD_NOISE = "add_noise"
PREPROCESS_NONE = "none"

@app.cls(
    image=modal_setup.inpainting_image,
    gpu=modal_setup.GPU_SPECS,
    volumes={MODEL_DIR: modal_setup.volume},
    secrets=modal_setup.get_secrets(),
    timeout=3600,
    min_containers=1,
)
class InpaintingModel:

    @modal.enter()
    def load_model(self):
        from diffusers import AutoPipelineForInpainting
        import torch

        modal_setup.volume.reload()
        pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        )
        # Try to load LoRA weights if present
        lora_path = os.path.join(MODEL_DIR, 'postcards_lora')
        try:
            pipe.load_lora_weights(lora_path)
        except Exception:
            pass
        pipe.to("cuda")
        self.pipe = pipe
        print("✅ Model (AutoPipelineForInpainting) and LoRA weights loaded successfully.")

    def _preprocess_image(self, image: Image.Image, mask_image: Image.Image, strategy: str, noise_scale: float, grayscale: bool):
        original_image_for_postprocessing = None
        if strategy == PREPROCESS_NONE:
            return image, None
        image_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask_image.convert("L"))
        mask_bool = mask_np > 128
        if strategy == PREPROCESS_REPLACE_NOISE:
            noise = (np.random.randint(0, 256, image_np.shape, dtype=np.uint8) - 128) * noise_scale + 128
            image_np[mask_bool] = noise[mask_bool]
            preprocessed_image = Image.fromarray(image_np)
        elif strategy == PREPROCESS_ADD_NOISE:
            noise = np.random.randint(-127, 128, image_np.shape, dtype=np.int16) * noise_scale
            mask_3d = np.stack([mask_bool]*3, axis=-1) if image_np.ndim == 3 else mask_bool
            # Add noise only to masked area, clip to valid range
            masked_pixels = image_np[mask_bool]
            noisy_pixels = np.clip(masked_pixels.astype(np.int16) + noise[mask_bool], 0, 255).astype(np.uint8)
            image_np[mask_bool] = noisy_pixels
            preprocessed_image = Image.fromarray(image_np)
        else:
            raise ValueError(f"Unknown preprocessing strategy: {strategy}")
        if grayscale:
            original_image_for_postprocessing = image.copy()
            preprocessed_image = ImageOps.grayscale(image).convert("RGB")
        return preprocessed_image, original_image_for_postprocessing

    def _postprocess_grayscale_inpaint(self, inpainted_image: Image.Image, original_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """Restores original colors to non-inpainted areas after grayscale inpainting."""
        print("🔧 Postprocessing: Restoring original colors for non-masked areas...")
        inpainted_np = np.array(inpainted_image.convert("RGB"))
        original_np = np.array(original_image.convert("RGB"))
        mask_np = np.array(mask_image.convert("L"))
        mask_bool = mask_np <= 128 # Invert mask: Select areas NOT inpainted

        # Restore original pixels where the mask was black (or <= 128)
        inpainted_np[mask_bool] = original_np[mask_bool]
        return Image.fromarray(inpainted_np)

    @modal.method()
    def inference(self, image: Image.Image, mask_image: Image.Image, config: InpaintingConfig) -> Image.Image:
        """
        Performs inpainting using the configuration object.
        Returns the output path of the saved image.
        """
        if not self.pipe:
            raise RuntimeError("Model pipe not loaded. Was @modal.enter run?")

        # Preprocess
        preprocessed_image, original_image_for_post = self._preprocess_image(
            image, mask_image, config.preprocessing_strategy, config.noise_scale, config.grayscale
        )

        # Inference
        result = self.pipe(
            prompt=config.prompt,
            image=preprocessed_image,
            mask_image=mask_image,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            strength=config.strength,
        ).images[0]

        # Postprocess if needed
        if config.grayscale and original_image_for_post is not None:
            output_size = result.size
            original_resized = original_image_for_post.resize(output_size, Image.Resampling.LANCZOS)
            mask_resized = mask_image.resize(output_size, Image.Resampling.NEAREST)
            result = self._postprocess_grayscale_inpaint(result, original_resized, mask_resized)

        print(f"✅ Inpainting completed")
        return result

def inpaint(image_path: str, mask_path: str, config: InpaintingConfig):
    """
    Orchestrate inpainting using the provided configuration.
    """
    print("Starting inpainting process...")
    model = InpaintingModel()
    image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB") 
    result = model.inference.remote(image=image, mask_image=mask_image, config=config)
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    result.save(config.output_path)
    print(f"✅ Inpainting completed and saved to {config.output_path}")
