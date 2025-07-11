import cv2
import glob
import os
import argparse
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--dataset_dir", type=str, help="Path to directory containing video files to process.")
    parser.add_argument("--video_path", type=str, help="Path to a single video file.")
    return parser

def video_2_images(video_path, output_folder):    
    if os.path.isdir(video_path):
        img_paths = sorted(glob.glob(f"{video_path}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {video_path}")
        frame_interval = round(video_fps)
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Total frames: {total_frames}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_lists = []
        for i in range(len(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_lists.append(frame_path)
        cap.release()
    return img_lists

def process_single_video(video_path, output_dir, model, device, dtype):
    """Process a single video file and save results."""
    print(f"\nProcessing video: {video_path}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    images_folder = os.path.join(output_dir, video_name, "images")
    depths_folder = os.path.join(output_dir, video_name, "depths")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(depths_folder, exist_ok=True)
    
    try:
        img_lists = video_2_images(video_path, images_folder)
        images = load_and_preprocess_images(img_lists).to(device)
        print("images shape:", images.shape)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                
        print(predictions.keys())
        print(predictions['depth'].shape)
        print(predictions['depth_conf'].shape)
        
        # Save depth maps for each frame
        depths = predictions['depth'].cpu().numpy()  
        depths = depths.squeeze()  
        
        for i, depth_map in enumerate(depths):
            depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype('uint8')
            depth_path = os.path.join(depths_folder, f"depth_{i}.jpg")
            cv2.imwrite(depth_path, depth_normalized)
        print(f"Saved {len(depths)} depth maps to {depths_folder}")
         
        # save raw depth values as numpy array    
        depth_raw_path = os.path.join(output_dir, video_name, f"_depth.npy")
        depth_info = {
            "depths": depths,
            "depth_conf": predictions['depth_conf'].cpu().numpy(),
        }
        np.save(depth_raw_path, depth_info)

        # Save camera parameters
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        print("extrinsic shape:", extrinsic.shape)
        print("intrinsic shape:", intrinsic.shape)
        camera_info = { 
                "intrinsics": intrinsic.cpu().numpy().astype(np.float16),
                "extrinsic": extrinsic.cpu().numpy().astype(np.float16),
        }
        np.save(os.path.join(output_dir, video_name, "_camera.npy"), camera_info)
        print(f"Successfully processed: {video_name}")
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False
    
    return True

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.dataset_dir and not args.video_path:
        print("Error: Must provide either --dataset_dir or --video_path")
        return
    
    if args.dataset_dir and args.video_path:
        print("Error: Cannot use both --dataset_dir and --video_path. Choose one.")
        return
    
    # Initialize model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    model = VGGT()
    model.load_state_dict(torch.load("weights/model.pt"))
    model.to(device)
    print("Model loaded successfully")
    
    # Get list of videos to process
    video_files = []
    if args.dataset_dir:
        # Process all videos in directory
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.dataset_dir, ext)))
            video_files.extend(glob.glob(os.path.join(args.dataset_dir, ext.upper())))
        
        if not video_files:
            print(f"No video files found in {args.dataset_dir}")
            return
        
        video_files.sort()
        print(f"Found {len(video_files)} video files to process")
    else:
        # Process single video
        video_files = [args.video_path]
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_files):
        print(f"\n=== Processing video {i+1}/{len(video_files)} ===")
        success = process_single_video(video_path, args.output_dir, model, device, dtype)
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed: {failed} videos")
    return

if __name__ == "__main__":
    main()
    
# Single video processing:
# python vggt_inference.py --output_dir ./sample --video_path /data/jxucm/underwater_easy/src_videos/P1010835_0-20_src.mp4

# Directory processing:
# python vggt_inference.py --output_dir /data/jxucm/underwater_easy_processed --dataset_dir /data/jxucm/underwater_easy/src_videos