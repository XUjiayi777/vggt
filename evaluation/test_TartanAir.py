import os
import torch
import numpy as np
import gzip
import json
import logging
import warnings
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_depth_maps
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.eval_pose import se3_to_relative_pose_error,align_to_first_camera,calculate_auc_np
from utils.eval_depth import thresh_inliers,m_rel_ae,pointwise_rel_ae
from utils.general import set_random_seeds,load_model
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
    """Set up command-line arguments for the TartanAir evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on TartanAir dataset')
    parser.add_argument('--TAir_dir', type=str, required=True, help='Path to TartanAir dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')
    parser.add_argument('--camera',type=str, required=True, choices=['left', 'right'], help='Camera chosen to use for testing')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (only test on specific category)')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--log_file_path', type=str, default='../logs/log_tartanair.txt', help='Path to log file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--cuda_id',type=str, help='CUDA device ID')
    return parser.parse_args()
def ned_to_c2w(ned):
    ned = np.array(ned, dtype=np.float32)
    #NOTE: we need to convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = ned[:3]
    qz, qx, qy, qw = ned[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    c2w[:3, 3] = np.array([x, y, z])
    return c2w

def estimate_camera_pose(predictions, images, num_frames, use_ba, device, gt_extrinsic):
    """
    Process a single sequence and compute pose errors.

    Args:
        predictions: VGGT predictions
        images: Batched tensor of preprocessed images with shape (N, 3, H, W)
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        gt_extrinsic: Ground truth extrinsics

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
        gt_se3 = torch.linalg.inv(gt_extrinsic) # w2c
        
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
    """Main function to evaluate VGGT on TartanAir dataset."""
    # Parse command-line arguments
    args = setup_args()

    # Setup device and data type
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Image processing
    print("Processing images...")
    image_path = os.path.join(args.TAir_dir, f"image_{args.camera}")
    image_names=[]
    for image in os.listdir(image_path):
        image_names.append(os.path.join(image_path, image)) 
    image_names=sorted(image_names)
    if len(image_names) >= args.num_frames:
        image_names = image_names[:args.num_frames]    
    images = load_and_preprocess_images(image_names).to(device)

    # Pose processing
    print("Processing poses...")
    pose_path = os.path.join(args.TAir_dir, f"pose_{args.camera}.txt")
    pose=np.loadtxt(pose_path)
    pose=pose[:args.num_frames]
    extrinsic_list=[ned_to_c2w(pose_line) for pose_line in pose]
    extrinsic = np.array(extrinsic_list)
    
    # Depth processing
    print("Processing depth maps...")
    depth_path = os.path.join(args.TAir_dir, f"depth_{args.camera}")
    depths_name = sorted(os.listdir(depth_path))[:args.num_frames]
    print(depths_name)
    depths_list = [np.load(os.path.join(depth_path, f)) for f in depths_name]
    gt_depth=preprocess_depth_maps(depths_list).squeeze(1).numpy()

    # Load model
    print("Loading model...")
    model = load_model(device, model_path=args.model_path)

    # Set random seeds
    set_random_seeds(args.seed)
    
    print("Making predictions...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Camera pose estimation (AUC)
    rError, tError = estimate_camera_pose(predictions, images, args.num_frames ,args.use_ba, device, extrinsic)
    
    Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
    Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
    Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
    Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)
    
    # os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)
    # with open(args.log_file_path, 'a') as log_file:
    #     log_file.write("="*80)
    #     log_file.write("\nSummary of AUC results:\n")
    #     log_file.write("-" * 50 + "\n")
    #     log_file.write(f"AUC: {Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3)\n")
    
    print("="*80)
    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    print(f"AUC: {Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3)")
    
    # Depth estimation:
    pre_depth, pre_depth_conf = predictions['depth'].squeeze(0).squeeze(-1).cpu().numpy(), predictions['depth_conf'].squeeze(0).cpu().numpy()
    depth_mask=pre_depth_conf>10
    
    print("pre_depth",pre_depth.shape)
    print("gt_depth",gt_depth.shape)
    print("pre_depth_conf",pre_depth_conf.shape)
    pdb.set_trace()
    inlier_ratio=thresh_inliers(gt_depth,pre_depth, 1.03, mask=pre_depth_conf,output_scaling_factor=100)
    print("inlier_ratio",inlier_ratio)
    mean_rel=m_rel_ae(gt_depth,pre_depth,mask=pre_depth_conf,output_scaling_factor=100)
    print("m_rel_ae",mean_rel)
    
    
if __name__ == "__main__":
    main()
'''
 python test_TartanAir.py \
    --model_path ../weights/model_tracker_fixed_e20.pt \
    --TAir_dir ../data/TartanAir_ocean/P006/ \
    --num_frames 20 \
    --seed 77 \
    --cuda_id 4 \
    --camera left 
'''