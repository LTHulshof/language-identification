"""
Script for downloading the WiLi-2018 dataset into the ./data/wili-2018 directory.
"""
import os
import urllib.request
import zipfile

DATA_DIR = os.path.join("data", "wili-2018")
os.makedirs(DATA_DIR, exist_ok=True)

ZIP_URL = "https://zenodo.org/record/841984/files/wili-2018.zip?download=1"
ZIP_PATH = os.path.join(DATA_DIR, "wili-2018.zip")

# Download the ZIP
if not os.path.exists(ZIP_PATH):
    print("Downloading WiLI-2018 dataset")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"Saved ZIP to {ZIP_PATH}")
else:
    print("ZIP already exists")

# Unzip
print("Extracting ZIP")
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

