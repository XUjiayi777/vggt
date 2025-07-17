import numpy as np
import argparse
import os
from PIL import Image
import glob
import shutil
import pdb

WATER = {
    # Base parameters - these will be randomized within ranges
    "attenuation_range": [[0.2, 0.6], [0.1, 0.3], [0.05, 0.15]],  # [R, G, B] ranges
    "backscatter_range": [[0.2, 0.4], [0.1, 0.2], [0.05, 0.1]], # [R, G, B] ranges
    "backscatter_color_range": [[0.05, 0.1], [0.15, 0.25], [0.35, 0.55]], # [R, G, B] ranges
    "max_depth_range": [10.0, 40.0],           # Range for maximum depth
    "depth_scale_range": [0.15, 0.25],         # Range for depth scaling
}

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Generate underwater effects for Virtual KITTI dataset"
    )
    parser.add_argument( 
        "--data_dir", type=str, default="/data/jxucm/vkitti",
        help="Path to the Virtual KITTI dataset directory containing scenes",
    )
    parser.add_argument( 
        "--scene", type=str, default=None,
        help="Specific scene ID to process (e.g., Scene01, Scene02). If not specified, processes all scenes.",
    )
    parser.add_argument(
        "--source_clone", type=str, default="clone",
        help="Name of the source clone directory to copy from (default: clone)",
    )
    parser.add_argument(
        "--target_clone", type=str, default="underwater", 
        help="Name of the new clone directory to create (default: underwater)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with detailed output and breakpoints",
    )
    return parser

def generate_random_water_params(base_params, seed=None):
    """
    Generate randomized water parameters for each frame.
    
    Args:
        base_params: Dictionary with parameter ranges
        seed: Optional seed for reproducibility
    
    Returns:
        Dictionary with randomized water parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    attenuation = [
        np.random.uniform(base_params["attenuation_range"][0][0], base_params["attenuation_range"][0][1]),  # Red
        np.random.uniform(base_params["attenuation_range"][1][0], base_params["attenuation_range"][1][1]),  # Green  
        np.random.uniform(base_params["attenuation_range"][2][0], base_params["attenuation_range"][2][1])   # Blue
    ]
    

    backscatter = [
        np.random.uniform(base_params["backscatter_range"][0][0], base_params["backscatter_range"][0][1]),  # Red
        np.random.uniform(base_params["backscatter_range"][1][0], base_params["backscatter_range"][1][1]),  # Green
        np.random.uniform(base_params["backscatter_range"][2][0], base_params["backscatter_range"][2][1])   # Blue
    ]
    
    backscatter_color = [
        np.random.uniform(base_params["backscatter_color_range"][0][0], base_params["backscatter_color_range"][0][1]),  # Red
        np.random.uniform(base_params["backscatter_color_range"][1][0], base_params["backscatter_color_range"][1][1]),  # Green
        np.random.uniform(base_params["backscatter_color_range"][2][0], base_params["backscatter_color_range"][2][1])   # Blue
    ]
    
    max_depth = np.random.uniform(base_params["max_depth_range"][0], base_params["max_depth_range"][1])
    depth_scale = np.random.uniform(base_params["depth_scale_range"][0], base_params["depth_scale_range"][1])
    
    return {
        "attenuation": attenuation,
        "backscatter": backscatter,
        "backscatter_color": backscatter_color,
        "max_depth": max_depth,
        "depth_scale": depth_scale,
    }


def add_effect_to_image(image_rgb, effect_params, depth, debug=False):
    """
    Apply underwater effect to the RGB image using atmospheric scattering model.
    
    Formula: I = J * e^(-β1*d) + A * (1 - e^(-β2*d))
    Where:
    - I: observed underwater image
    - J: original clear image (input)
    - β1: attenuation coefficient per channel
    - β2: backscatter coefficient per channel
    - d: depth (distance from camera)
    - A: atmospheric light (backscatter color per channel)
    """
    # Convert image to float [0, 1] for processing
    image_processed = image_rgb.astype(np.float32) / 255.0
    
    # Scale depth to reasonable underwater range
    depth_scaled = depth * effect_params["depth_scale"]
    depth_scaled = np.clip(depth_scaled, 0, effect_params["max_depth"])
    
    if debug:
        print("image max and min", image_processed.max(), image_processed.min())
        print("original depth max and min", depth.max(), depth.min())
        print("scaled depth max and min", depth_scaled.max(), depth_scaled.min())
    
    atten_beta = np.array(effect_params["attenuation"])  # Shape: (3,)
    backscatter_beta = np.array(effect_params["backscatter"])  # Shape: (3,)
    backscatter_color = np.array(effect_params["backscatter_color"])  # Shape: (3,)
    
    # Expand depth to match image dimensions: (H, W) -> (H, W, 3)
    if len(depth_scaled.shape) == 2:
        depth_expanded = depth_scaled[:, :, np.newaxis]  # Shape: (H, W, 1)
    else:
        depth_expanded = depth_scaled
    
    # Calculate transmission for each RGB channel
    # atten_beta shape (3,) broadcasts with depth_expanded shape (H, W, 1) -> (H, W, 3)
    atten_transmission = np.exp(-atten_beta * depth_expanded)
    backscatter_transmission = np.exp(-backscatter_beta * depth_expanded)
    
    if debug:
        print("transmission range - atten:", atten_transmission.min(), "to", atten_transmission.max())
        print("transmission range - backscatter:", backscatter_transmission.min(), "to", backscatter_transmission.max())
    
    # Apply underwater effect per channel
    uw_image = (image_processed * atten_transmission + 
                backscatter_color * (1 - backscatter_transmission))
    
    uw_image = np.clip(uw_image, 0, 1)
    uw_image_uint8 = (uw_image * 255).astype(np.uint8)
    
    if debug:
        print("final image max and min", uw_image.max(), uw_image.min())
        pdb.set_trace()
    
    return uw_image_uint8 

def load_depth_image(depth_path):
    """Load depth image from Virtual KITTI depth file."""
    try:
        # Virtual KITTI depth images are typically 16-bit PNG files
        depth_pil = Image.open(depth_path)
        depth_array = np.array(depth_pil, dtype=np.float32)
        
        # Virtual KITTI depth is stored in centimeters, convert to meters for processing
        depth_array = depth_array / 100.0
        
        return depth_array
    except Exception as e:
        print(f"Error loading depth image {depth_path}: {e}")
        return None

def process_scene_clone_to_underwater(scene_path, target_clone="clone", new_clone="underwater", debug=False):
    """
    Create an underwater version of a Virtual KITTI scene by copying the clone
    and applying underwater effects to RGB images using corresponding depth maps.
    
    Args:
        scene_path: Path to the scene directory (e.g., /data/jxucm/vkitti/Scene01)
        target_clone: Name of the source clone directory (default: "clone")
        new_clone: Name of the new underwater clone directory (default: "underwater")
    """
    clone_path = os.path.join(scene_path, target_clone)
    underwater_path = os.path.join(scene_path, new_clone)
    
    if not os.path.exists(clone_path):
        print(f"Source clone directory does not exist: {clone_path}")
        return False
    
    if os.path.exists(underwater_path):
        print(f"Target underwater directory already exists: {underwater_path}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            return False
        shutil.rmtree(underwater_path)
    
    print(f"Copying {clone_path} to {underwater_path}...")
    shutil.copytree(clone_path, underwater_path)
    
    # Find RGB and depth directories
    rgb_dir = os.path.join(underwater_path, "frames", "rgb")
    depth_dir = os.path.join(underwater_path, "frames", "depth")
    
    if not os.path.exists(rgb_dir):
        print(f"RGB directory not found: {rgb_dir}")
        return False
    
    if not os.path.exists(depth_dir):
        print(f"Depth directory not found: {depth_dir}")
        return False
    
    # Find camera subdirectories
    camera_dirs = []
    for item in os.listdir(rgb_dir):
        camera_path = os.path.join(rgb_dir, item)
        if os.path.isdir(camera_path) and item.startswith('Camera'):
            camera_dirs.append(item)
    
    if not camera_dirs:
        print(f"No camera directories found in {rgb_dir}")
        return False
    
    camera_dirs.sort()
    print(f"Found camera directories: {camera_dirs}")
    
    total_processed = 0
    total_files = 0
    
    # Process each camera directory
    for camera_dir in camera_dirs:
        print(f"\n--- Processing {camera_dir} ---")
        
        camera_rgb_dir = os.path.join(rgb_dir, camera_dir)
        camera_depth_dir = os.path.join(depth_dir, camera_dir)
        
        if not os.path.exists(camera_depth_dir):
            print(f"Warning: Corresponding depth directory not found: {camera_depth_dir}")
            continue
        
        # Get list of RGB images for this camera
        rgb_files = glob.glob(os.path.join(camera_rgb_dir, "*.jpg")) + glob.glob(os.path.join(camera_rgb_dir, "*.png"))
        rgb_files.sort()
        
        if not rgb_files:
            print(f"No RGB images found in {camera_rgb_dir}")
            continue
        
        print(f"Found {len(rgb_files)} RGB images for {camera_dir}")
        total_files += len(rgb_files)
        
        processed_count = 0
        for rgb_file in rgb_files:
            try:
                # Get corresponding depth file
                rgb_basename = os.path.splitext(os.path.basename(rgb_file))[0]
                
                # Try different depth file naming conventions
                depth_file = None
                for depth_ext in ['.png', '.jpg']:
                    # Virtual KITTI typically uses rgb_xxxxx.jpg and depth_xxxxx.png
                    if rgb_basename.startswith('rgb_'):
                        depth_name = rgb_basename.replace('rgb_', 'depth_') + depth_ext
                    else:
                        depth_name = rgb_basename + depth_ext
                    
                    potential_depth = os.path.join(camera_depth_dir, depth_name)
                    if os.path.exists(potential_depth):
                        depth_file = potential_depth
                        break
                
                if depth_file is None:
                    print(f"Warning: No corresponding depth file found for {rgb_basename} in {camera_dir}")
                    continue
                
                # Load RGB image
                rgb_image = Image.open(rgb_file)
                if rgb_image.mode != 'RGB':
                    rgb_image = rgb_image.convert('RGB')
                rgb_array = np.array(rgb_image)
                
                # Load depth image
                depth_array = load_depth_image(depth_file)
                if depth_array is None:
                    continue
                
                # Ensure RGB and depth have compatible dimensions
                if rgb_array.shape[:2] != depth_array.shape[:2]:
                    print(f"Warning: RGB {rgb_array.shape[:2]} and depth {depth_array.shape[:2]} dimensions don't match for {rgb_basename}")
                    depth_pil = Image.fromarray(depth_array).resize((rgb_array.shape[1], rgb_array.shape[0]), Image.NEAREST)
                    depth_array = np.array(depth_pil)
                
                # Generate random water parameters for this frame
                # Use frame index as seed for reproducible but varied effects
                frame_seed = hash(rgb_basename) % 10000 
                random_water_params = generate_random_water_params(WATER, seed=frame_seed)
                
                if debug and processed_count == 0:  # Show params for first image only
                    print(f"Random water params for {rgb_basename}:")
                    print(f"  Attenuation: {random_water_params['attenuation']}")
                    print(f"  Backscatter: {random_water_params['backscatter']}")
                    print(f"  Water color: {random_water_params['backscatter_color']}")
                    print(f"  Max depth: {random_water_params['max_depth']}")
                    print(f"  Depth scale: {random_water_params['depth_scale']}")
                
                # Apply underwater effect with random parameters
                underwater_image = add_effect_to_image(rgb_array, random_water_params, depth_array, debug)
                
                # Save the processed image
                underwater_pil = Image.fromarray(underwater_image)
                underwater_pil.save(rgb_file) 
                
                processed_count += 1
                total_processed += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(rgb_files)} images for {camera_dir}...")
                
            except Exception as e:
                print(f"Error processing {rgb_file}: {e}")
                continue
        
        print(f"Completed {camera_dir}: {processed_count}/{len(rgb_files)} images processed")
    
    print(f"\n=== Summary ===")
    print(f"Total images processed: {total_processed}/{total_files}")
    print(f"Underwater clone created at: {underwater_path}")
    return True 

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Data directory does not exist: {args.data_dir}")
        return
    
    if args.scene is None:
        # Process all scenes
        scene_dirs = [d for d in os.listdir(args.data_dir) 
                     if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('Scene')]
        scene_dirs.sort()
        
        if not scene_dirs:
            print(f"No scene directories found in {args.data_dir}")
            return
        
        print(f"Found scenes: {scene_dirs}")
        for scene in scene_dirs:
            scene_path = os.path.join(args.data_dir, scene)
            print(f"\n--- Processing {scene} ---")
            process_scene_clone_to_underwater(scene_path, args.source_clone, args.target_clone, args.debug)
    else:
        # Process specific scene
        scene_path = os.path.join(args.data_dir, args.scene)
        if not os.path.exists(scene_path):
            print(f"Scene directory does not exist: {scene_path}")
            return
        
        print(f"Processing scene: {args.scene}")
        process_scene_clone_to_underwater(scene_path, args.source_clone, args.target_clone, args.debug)
    
if __name__ == "__main__":
    main()
