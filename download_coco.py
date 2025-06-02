import csv
import requests
import os
from PIL import Image
from io import BytesIO

# Path to your TSV file
tsv_path = "models/experimental/stable_diffusion_xl_base/coco2014/captions.tsv"
output_folder = "coco_images"
os.makedirs(output_folder, exist_ok=True)

# Number of images to download
max_images = 100

with open(tsv_path, newline="", encoding="utf-8") as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter="\t")

    for idx, row in enumerate(reader):
        if idx >= max_images:
            break
        url = row["coco_url"]
        original_filename = row["file_name"]
        new_filename = os.path.splitext(original_filename)[0] + ".png"
        output_path = os.path.join(output_folder, new_filename)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Convert image to PNG using PIL
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(output_path, format="PNG")

            print(f"[{idx+1}] Downloaded and converted: {new_filename}")
        except Exception as e:
            print(f"[{idx+1}] Failed to download {original_filename}: {e}")
