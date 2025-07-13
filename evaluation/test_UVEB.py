import os
import torch
import numpy as np
import gzip
import json
import logging
import warnings
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_depth_maps

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
    """Set up command-line arguments for the UIEB evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on UIEB dataset')
    parser.add_argument('--blur_dir', type=str, required=True, help='Path to UIEB blur dataset')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to UIEB gt dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')
    parser.add_argument('--video_name', type=str, default='cv_127_PUIE', help='Name of the video for evaluation')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--cuda_id',type=str, help='CUDA device ID')
    # parser.add_argument('--debug', action='store_true', help='Enable debug mode (only test on specific category)')
    # parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    # parser.add_argument('--log_file_path', type=str, default='../logs/log_tartanair.txt', help='Path to log file')
    return parser.parse_args()

def pose_relative_error(src_se3, ref_se3):
    """
    For UIEB dataset, compute pose error between source and reference images.
    Since we're comparing source vs reference images, we compute pairwise errors.
    """
        
    # Compute rotation error between source and reference poses
    rel_rangle_deg = rotation_angle(src_se3[:, :3, :3], ref_se3[:, :3, :3])
    
    # Compute translation error between source and reference poses  
    rel_tangle_deg = translation_angle(src_se3[:, :3, 3], ref_se3[:, :3, 3])
    
    return rel_rangle_deg, rel_tangle_deg

def camera_pose_diff(src_extrinsic, ref_extrinsic, device):
        # Ensure tensors are in float32 to avoid overflow issues
        src_extrinsic = src_extrinsic.float()
        ref_extrinsic = ref_extrinsic.float()
        
        add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32).expand(src_extrinsic.size(0), 1, 4)
        pred_src_se3 = torch.cat((src_extrinsic, add_row), dim=1) 
        pred_ref_se3 = torch.cat((ref_extrinsic, add_row), dim=1) 
        
        # Set the coordinate of the first camera as the coordinate of the world
        # NOTE: DO NOT REMOVE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
        src_se3 = align_to_first_camera(pred_src_se3)
        ref_se3 = align_to_first_camera(pred_ref_se3)
        
        
        rel_rangle_deg, rel_tangle_deg = pose_relative_error(src_se3, ref_se3)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"R_ACC@5: {Racc_5:.4f}")
        print(f"T_ACC@5: {Tacc_5:.4f}")

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()
   
def main():
    """Main function to evaluate VGGT on UVEB dataset."""
    args = setup_args()
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"

    blur_datapath = f"{args.blur_dir}/{args.video_name}"
    gt_datapath = f"{args.gt_dir}/{args.video_name}"
    blur_camera = np.load(f"{blur_datapath}/_camera.npy", allow_pickle=True).item()
    gt_camera = np.load(f"{gt_datapath}/_camera.npy", allow_pickle=True).item()

    blur_extrinsic = torch.tensor(blur_camera["extrinsic"], device=device, dtype=torch.float32).squeeze()
    gt_extrinsic = torch.tensor(gt_camera["extrinsic"], device=device, dtype=torch.float32).squeeze()
    print("blur extrinsic shape:", blur_extrinsic.shape)
    print("gt extrinsic shape:", gt_extrinsic.shape)

    rError, tError = camera_pose_diff(blur_extrinsic, gt_extrinsic, device)
    Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
    Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
    Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
    Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)
    print(f"AUC@30: {Auc_30:.4f}, AUC@15: {Auc_15:.4f}, AUC@5: {Auc_5:.4f}, AUC@3: {Auc_3:.4f}")

    blur_depth_info = np.load(f"{blur_datapath}/_depth.npy", allow_pickle=True).item()
    gt_depth_info = np.load(f"{gt_datapath}/_depth.npy", allow_pickle=True).item()
    blur_depth=blur_depth_info["depths"]
    gt_depth=gt_depth_info["depths"]
    blur_depth_conf=blur_depth_info["depth_conf"].squeeze()
    gt_depth_conf=gt_depth_info["depth_conf"].squeeze()
    
    inlier_ratio=thresh_inliers(blur_depth, gt_depth, 1.03)
    abs_rel=m_rel_ae(blur_depth, gt_depth)
    sq_rel=sq_rel_ae(blur_depth, gt_depth)  
    rmse_value= rmse(blur_depth, gt_depth)
    rmse_log_value= rmse_log(blur_depth, gt_depth)
    print(f"Inlier Ratio: {inlier_ratio:.4f}, Abs Rel: {abs_rel:.4f}, Sq Rel: {sq_rel:.4f}, RMSE: {rmse_value:.4f}, RMSE Log: {rmse_log_value:.4f}")

if __name__ == "__main__":
    main()
    
'''
Example:
 python test_UVEB.py \
    --blur_dir  /home/jxucm/vggt/sample/blur\
    --gt_dir /home/jxucm/vggt/sample/gt \
    --model_path ../weights/model.pt \
    --video_name cv_127 \
    --seed 77 \
    --cuda_id 0 
'''
