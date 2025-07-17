import os
import torch
import numpy as np
import gzip
import json
import logging
import warnings
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_depth_maps
import cv2
from utils.eval_pose import calculate_auc_np, align_to_first_camera, se3_to_relative_pose_error
from utils.eval_depth import thresh_inliers,m_rel_ae,sq_rel_ae,align_pred_to_gt,correlation,si_mse,rmse,rmse_log
from utils.eval_pose import rotation_angle, translation_angle,closed_form_inverse_se3
from utils.general import set_random_seeds,load_model
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ba import run_vggt_with_ba
import argparse
import pdb

# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

def setup_args():
    """Set up command-line arguments for the Virtual KITTI evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on Virtual KITTI dataset')
    parser.add_argument( "--data_dir", type=str, default="/data/jxucm/vkitti",
        help="Path to the Virtual KITTI dataset directory containing scenes")
    parser.add_argument( "--sceneid", type=str, default=None,
        help="Specific scene ID to process (e.g., Scene01, Scene02). If not specified, processes all scenes.")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames to evaluate per scene")
    parser.add_argument("--stride", type=int, default=2, help="Stride for selecting frames")
    parser.add_argument("--scene", type=str, default=None, help="Scene type to evaluate (e.g., underwater, fog, etc.)")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--cuda_id',type=str, help='CUDA device ID')
    parser.add_argument('--model_path', type=str, default='../weights/model.pt', help='Path to the pre-trained model')
    return parser.parse_args()

def estimate_camera_pose_vkitti(predictions,gt_extrinsic, images, num_frames, device):
    """
    Process a single sequence and compute pose errors.

    Args:
        predictions: VGGT predictions
        gt_extrinsic: Ground truth extrinsics 
        images: Batched tensor of preprocessed images with shape (N, 3, H, W)
        num_frames: Number of frames to sample
        device: Device to run on
        

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """

    with torch.cuda.amp.autocast(dtype=torch.float64):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        pred_extrinsic = extrinsic[0] # Predicted extrinsic w2c
        gt_extrinsic = torch.from_numpy(gt_extrinsic).to(device)
        
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)
        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1) #w2c
        gt_se3 = gt_extrinsic
        
        # Set the coordinate of the first camera as the coordinate of the world
        # NOTE: DO NOT REMOVE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
        pred_se3 = align_to_first_camera(pred_se3)
        gt_se3 = align_to_first_camera(gt_se3)
        
        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"R_ACC@5: {Racc_5:.4f}")
        print(f"T_ACC@5: {Tacc_5:.4f}")

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()
   
def main():
    """Main function to evaluate VGGT on UVEB dataset."""
    args = setup_args()
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    print("Loading model...")
    model = load_model(device, model_path=args.model_path)

    # Set random seeds
    set_random_seeds(args.seed)

    if args.sceneid is None:
        scene_dirs = [d for d in os.listdir(args.data_dir) 
                     if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('Scene')]
        scene_dirs.sort()
        
        if not scene_dirs:
            print(f"No scene directories found in {args.data_dir}")
            return
    else:
        scene_path = os.path.join(args.data_dir, args.sceneid)
        if not os.path.exists(scene_path):
            print(f"Scene directory does not exist: {scene_path}")
            return
        
    if args.scene is not None:
        scene_eval_path = os.path.join(scene_path, args.scene)
        depth_path = os.path.join(scene_eval_path, "frames/depth")
        image_path = os.path.join(scene_eval_path, "frames/rgb")
        
        camera_parameters = np.loadtxt(
            os.path.join(scene_eval_path,  "extrinsic.txt"), 
            delimiter=" ", 
            skiprows=1
        )
        camera_intrinsic = np.loadtxt(
            os.path.join(scene_eval_path, "intrinsic.txt"), 
            delimiter=" ", 
            skiprows=1
        )
    
    all_auc_30 = []
    all_auc_15 = []
    all_auc_5 = []
    all_auc_3 = []
    all_Pearson_corr = []
    all_si_mse = []
    all_inlier_ratio = []

    for camera_id in [0, 1]:
        print(f"\n--- Processing Camera {camera_id} ---")
        each_camera_parameters = camera_parameters[camera_parameters[:, 1] == camera_id]
        each_camera_intrinsic = camera_intrinsic[camera_intrinsic[:, 1] == camera_id]
        depth_path_1 = os.path.join(depth_path, f"Camera_{camera_id}")
        image_path_1 = os.path.join(image_path, f"Camera_{camera_id}")
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_list = []
        
        if os.path.exists(image_path_1):
            for filename in os.listdir(image_path_1):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_list.append(os.path.join(image_path_1,filename))
            image_list.sort()  # Sort for consistent ordering
            print(f"Found {len(image_list)} images in {image_path_1}")
                    
            
        # Generate image indices 
        stride = args.stride
        max_images = min(args.num_frames, len(image_list) // stride)
        image_indices = list(range(0, max_images * stride, stride))
        selected_image_list = [image_list[idx] for idx in image_indices]
        
        print(f"Selected {len(selected_image_list)} images with stride {stride}")
        print(f"Image indices: {image_indices}")
        
        images = load_and_preprocess_images(selected_image_list).to(device)
        # print(images.shape)
        
        gt_depths = None
        if os.path.exists(depth_path_1):
            depth_arrays = []
            for image_idx in image_indices:
                depth_file_path = os.path.join(depth_path_1, f"depth_{image_idx:05d}.png")
                if os.path.exists(depth_file_path):
                    depth_img = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH)
                    depth_arrays.append(depth_img)
            
            if depth_arrays:
                depth_array = np.stack(depth_arrays, axis=0)
                gt_depths = preprocess_depth_maps(depth_array).to(device).squeeze()
                print(f"Loaded {len(depth_arrays)} depth maps, shape: {gt_depths.shape}")
            else:
                print("No depth files found")
        
        gt_extrinsic = []
        for image_idx in image_indices:
            extri_opencv = each_camera_parameters[image_idx][2:].reshape(4, 4)
            gt_extrinsic.append(extri_opencv)  #
        gt_extrinsic = np.stack(gt_extrinsic, axis=0)
                
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        
        # Camera pose estimation (AUC)
        rError, tError = estimate_camera_pose_vkitti(predictions, gt_extrinsic, images, args.num_frames, device)
        
        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)
        
        print(f"Camera {camera_id} - AUC@30: {Auc_30:.4f}, AUC@15: {Auc_15:.4f}, AUC@5: {Auc_5:.4f}, AUC@3: {Auc_3:.4f}")
        
        all_auc_30.append(Auc_30)
        all_auc_15.append(Auc_15)
        all_auc_5.append(Auc_5)
        all_auc_3.append(Auc_3)
        
        # Depth estimation 
        valid_mask = torch.logical_and(
        gt_depths.cpu() > 1e-3,     # filter out black background
        predictions['depth_conf'].cpu() > 5,
        ).squeeze().numpy()
        
        gt_depths = gt_depths.cpu().numpy()
        pre_depths = predictions['depth'].squeeze().cpu().numpy()
        
        Pearson_corr = correlation(gt_depths, pre_depths, mask=valid_mask)
        si_mse_value= si_mse(gt_depths, pre_depths, mask=valid_mask)
        inlier_ratio=thresh_inliers(gt_depths, pre_depths, 1.25, mask=valid_mask)
        print(f"Camera {camera_id} - Pearson Correlation: {Pearson_corr:.4f}, SI-MSE: {si_mse_value:.4f}, Inlier Ratio: {inlier_ratio:.4f}")
        
        all_Pearson_corr.append(Pearson_corr)
        all_si_mse.append(si_mse_value)
        all_inlier_ratio.append(inlier_ratio)

    mean_auc_30 = np.mean(all_auc_30)
    mean_auc_15 = np.mean(all_auc_15)
    mean_auc_5 = np.mean(all_auc_5)
    mean_auc_3 = np.mean(all_auc_3)
    mean_Pearson_corr = np.mean(all_Pearson_corr)
    mean_si_mse = np.mean(all_si_mse)
    mean_inlier_ratio = np.mean(all_inlier_ratio)
    
    print(f"\n--- Mean AUC across all cameras ---")
    print(f"Mean AUC@30: {mean_auc_30:.4f}, Mean AUC@15: {mean_auc_15:.4f}, Mean AUC@5: {mean_auc_5:.4f}, Mean AUC@3: {mean_auc_3:.4f}")
    print(f"Mean Pearson Correlation: {mean_Pearson_corr:.4f}, Mean SI-MSE: {mean_si_mse:.4f}, Mean Inlier Ratio: {mean_inlier_ratio:.4f}")   
    
    # inlier_ratio=thresh_inliers(blur_depth, gt_depth, 1.03)

if __name__ == "__main__":
    main()
    
'''
Example:
 python test_vkitti.py \
    --data_dir  /data/jxucm/vkitti \
    --sceneid Scene01 \
    --scene clone \
    --num_frames 20 \
    --stride 3 \
    --seed 77 \
    --cuda_id 0 \
    --model_path ../weights/model.pt 
'''
