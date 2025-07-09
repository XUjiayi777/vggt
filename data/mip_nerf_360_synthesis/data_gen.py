import numpy as np
import argparse
import os
from PIL import Image
import glob

FOGGY = {
    "attenuation": 2.4, 
    "backscatter": 2.4, 
    "backscatter_color": [0.7, 0.7, 0.7]  
}

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--data_dir", type=str, default="/data/jxucm/mip_nerf_360/garden/images_8",
        help="Path to the dataset directory containing images.",
    )

    return parser

def read_first_image(data_dir):
    """Read the first image from the data directory in RGB format."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Find all image files in the directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {data_dir}")
        return None
    
    # Sort to ensure consistent ordering
    image_files.sort()
    first_image_path = image_files[0]
    
    print(f"Reading first image: {first_image_path}")
    
    # Use PIL to read image in RGB format
    try:
        image_pil = Image.open(first_image_path)
        # Convert to RGB if it's not already (handles RGBA, grayscale, etc.)
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        image_rgb = np.array(image_pil)
        print(f"Image shape (RGB): {image_rgb.shape}")
        print(f"Image dtype: {image_rgb.dtype}")
        print(f"Image min/max values: {image_rgb.min()}/{image_rgb.max()}")
        return image_rgb, first_image_path
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def foggy_image(image_rgb, fog_params, depth=None):
    """
    Apply a fog effect to the RGB image using atmospheric scattering model.
    
    Formula: I = J * e^(-β*d) + A * (1 - e^(-β*d))
    Where:
    - I: observed foggy image
    - J: original clear image (input)
    - β: attenuation coefficient
    - d: depth (distance from camera)
    - A: atmospheric light (backscatter color)
    """
    # Convert image to float [0, 1] for processing
    image_processed = image_rgb.astype(np.float32) / 255.0
    
    beta = fog_params["attenuation"]
    backscatter_color = np.array(fog_params["backscatter_color"])
    
    # Transmission term: e^(-β*d)
    transmission = np.exp(-beta * depth)
    
    # Expand transmission to RGB channels
    if len(transmission.shape) == 2:
        transmission = transmission[:, :, np.newaxis]
    
    # Apply fog effect: I = J * transmission + A * (1 - transmission)
    foggy_image = (image_processed * transmission + 
                   backscatter_color * (1 - transmission))
    
    foggy_image = np.clip(foggy_image, 0, 1)
    foggy_image_uint8 = (foggy_image * 255).astype(np.uint8)
    
    return foggy_image_uint8 

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Data directory does not exist: {args.data_dir}")
        return
    
    # Read the first image
    result = read_first_image(args.data_dir)
    if result:
        image_rgb, image_path = result
        print(f"Successfully read RGB image from: {image_path}")
        print(f"RGB image ready for processing with shape: {image_rgb.shape}")
        
        # Apply fog effect
        foggy_img = foggy_image(image_rgb, FOGGY)
        print("Fog effect applied successfully!")
        
        # Save the foggy image
        output_dir = "/home/jxucm/vggt/data/mip_nerf_360_synthesis/foggy"
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"foggy_{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        foggy_pil = Image.fromarray(foggy_img)
        foggy_pil.save(output_path)
        print(f"Foggy image saved to: {output_path}")
    
if __name__ == "__main__":
    main()
