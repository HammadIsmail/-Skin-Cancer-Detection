import os
import requests
from zipfile import ZipFile
from io import BytesIO
import yaml

def download_and_extract_data():
    """Downloads the dataset from Hugging Face and extracts it."""
    url = "https://huggingface.co/datasets/makhresearch/skin-lesion-segmentation-classification/resolve/main/skin-lesion-segmentation-classification.zip"
    
    print("Downloading the dataset ZIP file...")
    response = requests.get(url)
    response.raise_for_status()
    print("✅ Download complete.")

    # We will extract it directly into the data/raw folder to follow standard practices
    extract_path = "data/raw"
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"\nExtracting files to {extract_path}...")
    with ZipFile(BytesIO(response.content)) as zf:
        zf.extractall(extract_path)
    print("✅ Extraction complete.")
    
    return extract_path

def print_dataset_stats(base_path):
    """Prints the number of images and labels in each dataset split."""
    dataset_dir = os.path.join(base_path, "skin-lesion-segmentation-classification")
    for split in ["train", "valid", "test"]:
        print(f"\n--- {split.upper()} ---")
        try:
            print("Images:", len(os.listdir(os.path.join(dataset_dir, split, "images"))))
            print("Labels:", len(os.listdir(os.path.join(dataset_dir, split, "labels"))))
        except FileNotFoundError:
            print(f"Could not find split: {split}")

def create_data_yaml(base_path):
    """Generates the data.yaml configuration file required by YOLOv8."""
    # Absolute path to the extracted dataset so YOLO can always find it
    abs_path = os.path.abspath(os.path.join(base_path, "skin-lesion-segmentation-classification"))
    
    yaml_content = {
        'path': abs_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'lesion'
        }
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print("\n✅ Created data.yaml configuration file.")

if __name__ == "__main__":
    print("Starting data setup process...")
    extract_dir = download_and_extract_data()
    print_dataset_stats(extract_dir)
    create_data_yaml(extract_dir)
    print("\nData setup is fully complete. You can now train the model.")
