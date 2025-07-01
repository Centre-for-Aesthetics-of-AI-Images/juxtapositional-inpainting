import argparse
from pathlib import Path

from src.datasets import download_data


def main():
    """Main function to run the CLI locally."""
    parser = argparse.ArgumentParser(
        description="Download metadata, images, finetune models and perform inpainting locally for the Juxtapositional Inpainting project.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- Metadata command ---
    parser_meta = subparsers.add_parser(
        "download-metadata", help="Download metadata files."
    )
    parser_meta.add_argument(
        "dataset",
        choices=["postcards", "aerial"],
        help="The dataset to download metadata for.",
    )
    parser_meta.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/metadata"),
        help="The directory to save the metadata file in.",
    )
    parser_meta.add_argument(
        "--query",
        type=str,
        default="Aarhus",
        help="The query to use for searching for postcards. Will return all postcards with an exact, case-insensitive match in the title text.",
    )

    # --- Images command ---
    parser_images = subparsers.add_parser(
        "download-images", help="Download images from metadata files."
    )
    parser_images.add_argument(
        "dataset",
        choices=["postcards", "aerial"],
        help="The dataset to download images for.",
    )
    parser_images.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("output/metadata"),
        help="Directory where the metadata JSON files are stored.",
    )
    parser_images.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/images"),
        help="The directory to save the downloaded images in.",
    )

    # --- Finetuning command ---
    parser_finetune = subparsers.add_parser(
        "finetune", help="Finetune an image generation model using LoRA and Dreambooth"
    )

    parser_finetune.add_argument(
        "--instance-prompt",
        type=str,
        default="A postcard of Aarhus the Danish city",
        help="Prompt for the instance images",
    )
    parser_finetune.add_argument(
        "--model-dir",
        type=Path,
        default=Path.cwd() / "model",
        help="The directory where the pretrained model is stored",
    )
    parser_finetune.add_argument(
        "--input-data-dir",
        type=Path,
        default=Path.cwd() / "data" / "input",
        help="The directory containing the input data",
    )
    parser_finetune.add_argument(
        "--use-wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging",
    )
    parser_finetune.add_argument(
        "--model-name",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="The huggingface-name of the model to finetune",
    )
    parser_finetune.add_argument(
        "--lora-name",
        type=str,
        default="postcard_lora",
        help="The name to save the LoRA model as",
    )
    parser_finetune.add_argument(
        "--resolution", type=int, default=512, help="The resolution of generated images"
    )
    parser_finetune.add_argument(
        "--train-batch-size", type=int, default=2, help="Batch size for training"
    )
    parser_finetune.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser_finetune.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser_finetune.add_argument(
        "--learning-rate", type=float, default=4e-4, help="Learning rate"
    )
    parser_finetune.add_argument(
        "--lr-scheduler", type=str, default="constant", help="Learning rate scheduler"
    )
    parser_finetune.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate",
    )
    parser_finetune.add_argument(
        "--max-train-steps",
        type=int,
        default=1000,
        help="Maximum number of training steps",
    )
    parser_finetune.add_argument(
        "--checkpointing-steps",
        type=int,
        default=1000,
        help="Number of steps between checkpoints",
    )
    parser_finetune.add_argument("--seed", type=int, default=117, help="Random seed")

    args = parser.parse_args()

    # Ensure base output directories exist
    if hasattr(args, "output_dir"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "metadata_dir"):
        args.metadata_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "download-metadata":
        if args.dataset == "postcards":
            output_file = args.output_dir / download_data.POSTCARD_METADATA_FILENAME
            download_data.fetch_postcard_metadata(output_file, args.query)
        elif args.dataset == "aerial":
            output_file = args.output_dir / download_data.AERIAL_METADATA_FILENAME
            download_data.fetch_aerial_metadata(output_file)

    elif args.command == "download-images":
        if args.dataset == "postcards":
            metadata_file = args.metadata_dir / download_data.POSTCARD_METADATA_FILENAME
            images_output_dir = args.output_dir / "postcards"
            download_data.download_images(metadata_file, images_output_dir)
        elif args.dataset == "aerial":
            metadata_file = args.metadata_dir / download_data.AERIAL_METADATA_FILENAME
            images_output_dir = args.output_dir / "aerial"
            download_data.download_images(metadata_file, images_output_dir)

    elif args.command == "finetune":
        raise NotImplementedError()

    elif args.commang == "inpaint":
        raise NotImplementedError()


if __name__ == "__main__":
    main()
