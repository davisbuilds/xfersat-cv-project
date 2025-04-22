import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

def download_eurosat(destination_folder="data/raw"):
    """
    Download the EuroSAT dataset (RGB version)
    """
    os.makedirs(destination_folder, exist_ok=True)
    
    # URL for the RGB version
    url = "https://madm.dfki.de/files/sentinel/EuroSAT_RGB.zip"
    destination_path = os.path.join(destination_folder, "EuroSAT_RGB.zip")
    
    if os.path.exists(destination_path):
        print(f"File already exists at {destination_path}")
        return destination_path
    
    print(f"Downloading EuroSAT RGB dataset to {destination_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination_path, 'wb') as file, tqdm(
            desc="Download progress",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    return destination_path

def extract_eurosat(zip_path, extract_folder="data/raw"):
    """
    Extract the EuroSAT dataset
    """
    print(f"Extracting dataset to {extract_folder}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    print("Extraction complete!")

if __name__ == "__main__":
    zip_path = download_eurosat()
    extract_eurosat(zip_path)