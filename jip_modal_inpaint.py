from pathlib import Path
import modal
import src.modal_remote.setup as modal_setup
import src.modal_remote.inpainting as modal_inpainting
from src.config import InpaintingConfig

app = modal_setup.app

@app.local_entrypoint()
def jip_inpaint(
    prompt: str = "A building from a postcard of Aarhus",
    input_image_path: str = "data/input/images/paludan_møllers_vej.jpg",
    mask_image_path: str = "data/input/masks/paludan_møllers_vej_mask.jpg",
    output_path: str = "data/output/inpainting/result.png",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    strength: float = 0.8,
    grayscale: bool = False,
    preprocessing_strategy: str = "none",
    noise_scale: float = 0.5,
):
    """
    Run inpainting on an image using the specified configuration.
    """
    config = InpaintingConfig(
        prompt=prompt,
        output_path=output_path,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        grayscale=grayscale,
        preprocessing_strategy=preprocessing_strategy,
        noise_scale=noise_scale,
    )
    modal_inpainting.inpaint(image_path=input_image_path, mask_path=mask_image_path, config=config)
