#!/usr/bin/env python3
"""
Script to download JianyuanWang/co3d_anno dataset from Hugging Face.
This dataset contains annotation files for the Co3D dataset.
"""

import os
import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub is not installed. Installing it now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

def download_co3d_anno(download_dir="./Co3D/Co3D_annotation"):
    """
    Download the JianyuanWang/co3d_anno dataset from Hugging Face.
    
    Args:
        download_dir (str): Directory to download the dataset files to.
    """
    
    # Create download directory if it doesn't exist
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading co3d_anno dataset to: {download_path.absolute()}")
    print("This may take a while depending on your internet connection...")
    
    try:
        # Download the entire repository
        snapshot_download(
            repo_id="JianyuanWang/co3d_anno",
            repo_type="dataset",
            local_dir=str(download_path),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
        )
        
        print(f"\n✅ Successfully downloaded co3d_anno dataset to {download_path.absolute()}")
        
        # List downloaded files
        downloaded_files = list(download_path.glob("*.jgz"))
        print(f"\nDownloaded {len(downloaded_files)} annotation files:")
        for file in sorted(downloaded_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nAlternatively, you can download manually using git:")
        print("git clone https://huggingface.co/datasets/JianyuanWang/co3d_anno")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download JianyuanWang/co3d_anno dataset")
    parser.add_argument(
        "--download_dir", 
        type=str, 
        default="./Co3D/Co3D_annotation",
        help="Directory to download the dataset files to (default: ./Co3D/Co3D_annotation)"
    )
    
    args = parser.parse_args()
    
    success = download_co3d_anno(args.download_dir)
    sys.exit(0 if success else 1)
