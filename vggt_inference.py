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
    parser.add_argument( "--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument( "--video_path", type=str, required=True, help="Path to the video file.")
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

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    images_folder = os.path.join(args.output_dir, video_name, "images")
    depths_folder = os.path.join(args.output_dir, video_name, "depths")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(depths_folder, exist_ok=True)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT()
    model.load_state_dict(torch.load("weights/model.pt"))
    model.to(device)
    
    img_lists = video_2_images(args.video_path, images_folder)
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
        
        # # Optionally save raw depth values as numpy array
        # depth_raw_path = os.path.join(depths_folder, f"depth_{i:04d}.npy")
        # np.save(depth_raw_path, depth_map)
    print(f"Saved {len(depths)} depth maps to {depths_folder}")
    
    # Save camera parameters
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    print("extrinsic shape:", extrinsic.shape)
    print("intrinsic shape:", intrinsic.shape)
    camera_info = { 
            "intrinsics": intrinsic.cpu().numpy().astype(np.float16),
            "extrinsic": extrinsic.cpu().numpy().astype(np.float16),
    }
    np.save(os.path.join(args.output_dir, video_name, "_camera.npy"), camera_info)
    return

if __name__ == "__main__":
    main()

# python vggt_inference.py --output_dir ./sample --video_path /data/jxucm/underwater_video_easy/P1010835_0-20_src.mp4