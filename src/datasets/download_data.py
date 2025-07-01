import argparse
import json
import time
from pathlib import Path

import requests
from lxml import etree

# --- Configuration ---

# Define the namespace map for easily accessing MODS elements
# (from the provided XML output: xmlns:md="http://www.loc.gov/mods/v3")
nsmap = {"md": "http://www.loc.gov/mods/v3"}

# Default filenames for metadata
POSTCARD_METADATA_FILENAME = "aarhus_postcards_metadata.json"
AERIAL_METADATA_FILENAME = "aerial_photos_metadata.json"

# Delay between image downloads to be polite to the server
DOWNLOAD_DELAY_S = 2

DEFAULT_POSTCARD_QUERY = "Aarhus"
DEFAULT_BBOX_COORDS = (
    10.22772788652219,
    56.163722869756825,
    10.176143642747775,
    56.14747002122093,
)


def get_postcard_url(
    query_string: str = DEFAULT_POSTCARD_QUERY, items_per_page: int = 5000
):
    return f"http://www5.kb.dk/cop/syndication/images/billed/2010/okt/billeder/subject3795/en?query={query_string}&itemsPerPage={items_per_page}&format=mods"


def get_aerial_photos_url(
    bbox_coords: tuple[float, float, float, float] = DEFAULT_BBOX_COORDS,
    items_per_page: int = 500,
):
    x_lon, x_lat, y_lon, y_lat = bbox_coords
    return f"https://api.kb.dk/data/rest/api/dsfl?bbo={x_lon},{x_lat},{y_lon},{y_lat}&itemsPerPage={items_per_page}"


# --- Metadata Fetching Functions ---


def fetch_postcard_metadata(output_filepath: Path, query_string: str):
    """
    Fetches postcard metadata from the Royal Danish Library API, filters for
    records with "Aarhus" in the title, and saves them to a JSON file.
    """
    postcard_url = get_postcard_url(query_string=query_string)
    print(f"Downloading postcard metadata from: {postcard_url}")
    try:
        response = requests.get(postcard_url)
        response.raise_for_status()
        print("Download successful. Parsing XML...")

        xml_root = etree.fromstring(response.content)  # type: ignore
        namespaces = {"mods": "http://www.loc.gov/mods/v3"}
        records = xml_root.findall(".//mods:mods", namespaces)
        print(f"Found {len(records)} records in the XML.")

        filtered_data = []
        for record in records:
            title_element = record.find(".//mods:title", namespaces)
            if (
                title_element is not None
                and query_string.lower() in title_element.text.lower()
            ):
                record_data = {"title": title_element.text}
                for identifier in record.findall(
                    './md:identifier[@type="uri"]', nsmap
                ):  # Look only at immediate children of md:mods
                    if "displayLabel" in identifier.attrib:
                        if identifier.attrib["displayLabel"] == "image":
                            record_data["image_url"] = identifier.text

                filtered_data.append(record_data)

        print(
            f"Filtered down to {len(filtered_data)} records containing {query_string}."
        )

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        print(f"Filtered metadata saved to {output_filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def fetch_aerial_metadata(output_filepath: Path):
    """
    Fetches aerial photo metadata from the Royal Danish Library API
    and saves it to a JSON file.
    """
    aerial_url = get_aerial_photos_url()
    print(f"Attempting to fetch data from: {aerial_url}")
    try:
        response = requests.get(aerial_url)
        response.raise_for_status()
        print("Successfully fetched data. Parsing JSON...")

        data = response.json()
        record_data = [
            {
                "title": record["properties"]["name"],
                "image_url": record["properties"]["src"],
            }
            for record in data
        ]
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=4)

        print(f"Successfully parsed and saved data to: {output_filepath}")
        print(f"Total items saved: {len(data)}")

    except requests.exceptions.RequestException as e:
        print(f"Error during requests to {aerial_url}: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response.")
        print(
            "Response Text:",
            response.text[:500] + "..." if response else "No response object",
        )  # type: ignore
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Image Downloading Functions ---


def download_images(metadata_filepath: Path, output_dir: Path):
    """
    Downloads images based on a metadata JSON file.
    """
    if not metadata_filepath.exists():
        print(f"Error: Metadata file not found at {metadata_filepath}")
        print("Please run the 'metadata' command first.")
        return

    with open(metadata_filepath, encoding="utf-8") as f:
        metadata = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Images will be saved in: {output_dir}")
    print(f"Found {len(metadata)} items in metadata file.")

    download_count = 0
    error_count = 0
    skipped_count = 0

    for i, item in enumerate(metadata):
        print(f"\nProcessing item {i + 1}/{len(metadata)}...")

        # Adapt to different metadata structures
        image_url = item.get("image_url")
        title = item.get("title", f"image_{i + 1}")

        if not image_url:
            print("  - Skipping item, no image URL found.")
            skipped_count += 1
            continue

        # Sanitize filename
        safe_filename = "".join(
            [c for c in title if c.isalpha() or c.isdigit() or c in (" ", "-")]
        ).rstrip()
        image_filename = f"{i + 1}_{safe_filename}.jpg"
        output_path = output_dir / image_filename

        if output_path.exists():
            print(f"  - Skipping, file already exists: {image_filename}")
            skipped_count += 1
            continue

        try:
            print(f"  - Downloading from: {image_url}")
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"  - Successfully saved to {output_path.name}")
            download_count += 1

        except requests.exceptions.RequestException as e:
            print(f"  - Error downloading {image_url}: {e}")
            error_count += 1
        except OSError as e:
            print(f"  - Error writing file {output_path.name}: {e}")
            error_count += 1

        print(f"  - Waiting for {DOWNLOAD_DELAY_S} seconds...")
        time.sleep(DOWNLOAD_DELAY_S)

    print("\n--- Download Summary ---")
    print(f"Successfully downloaded: {download_count}")
    print(f"Skipped (already exists or no URL): {skipped_count}")
    print(f"Errors: {error_count}")
    print("------------------------")


# --- Main CLI Logic ---


def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(
        description="Download image data from the Royal Danish Library (kb.dk).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- Metadata command ---
    parser_meta = subparsers.add_parser("metadata", help="Download metadata files.")
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
        "images", help="Download images from metadata files."
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

    args = parser.parse_args()

    # Ensure base output directories exist
    if hasattr(args, "output_dir"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "metadata_dir"):
        args.metadata_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "metadata":
        if args.dataset == "postcards":
            output_file = args.output_dir / POSTCARD_METADATA_FILENAME
            fetch_postcard_metadata(output_file, args.query)
        elif args.dataset == "aerial":
            output_file = args.output_dir / AERIAL_METADATA_FILENAME
            fetch_aerial_metadata(output_file)

    elif args.command == "images":
        if args.dataset == "postcards":
            metadata_file = args.metadata_dir / POSTCARD_METADATA_FILENAME
            images_output_dir = args.output_dir / "postcards"
            download_images(metadata_file, images_output_dir)
        elif args.dataset == "aerial":
            metadata_file = args.metadata_dir / AERIAL_METADATA_FILENAME
            images_output_dir = args.output_dir / "aerial"
            download_images(metadata_file, images_output_dir)


if __name__ == "__main__":
    main()
