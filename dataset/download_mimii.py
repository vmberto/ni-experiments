import os
import requests
from tqdm import tqdm
import zipfile

# Define the output directory
output_dir = "mimii_dataset"
os.makedirs(output_dir, exist_ok=True)

# Define the base URL and the list of files to download
base_url = "https://zenodo.org/record/3384388/files"
files = [
    "0_dB_fan.zip",
    "0_dB_pump.zip",
    "0_dB_slider.zip",
    "0_dB_valve.zip",
]


def download_and_extract(file_name):
    url = f"{base_url}/{file_name}"
    local_path = os.path.join(output_dir, file_name)

    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {file_name}"):
                f.write(chunk)
        print(f"Downloaded {file_name}")

        # Extract the file
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted {file_name}")
    else:
        print(f"Failed to download {file_name}: Status code {response.status_code}")


# Download and extract each file
for file_name in files:
    download_and_extract(file_name)