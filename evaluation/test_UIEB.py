import os
import torch
import numpy as np
import gzip
import json
import logging
import warnings
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_depth_maps

from utils.eval_pose import calculate_auc_np, align_to_first_camera, se3_to_relative_pose_error
from utils.eval_depth import thresh_inliers,m_rel_ae,pointwise_rel_ae,align_pred_to_gt,correlation,si_mse
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
    parser.add_argument('--source_dir', type=str, required=True, help='Path to UIEB source dataset')
    parser.add_argument('--ref_dir', type=str, required=True, help='Path to UIEB reference dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--cuda_id',type=str, help='CUDA device ID')
    # parser.add_argument('--debug', action='store_true', help='Enable debug mode (only test on specific category)')
    # parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    # parser.add_argument('--log_file_path', type=str, default='../logs/log_tartanair.txt', help='Path to log file')
    return parser.parse_args()

# def pose_relative_error(src_se3, ref_se3):
#     """
#     For UIEB dataset, compute pose error between source and reference images.
#     Since we're comparing source vs reference images, we compute pairwise errors.
#     """
        
#     # Compute rotation error between source and reference poses
#     rel_rangle_deg = rotation_angle(src_se3[:, :3, :3], ref_se3[:, :3, :3])
    
#     # Compute translation error between source and reference poses  
#     rel_tangle_deg = translation_angle(src_se3[:, :3, 3], ref_se3[:, :3, 3])
    
#     return rel_rangle_deg, rel_tangle_deg
    
    

# def camera_pose_diff(source_prediction, ref_prediction, device, image):
#     with torch.cuda.amp.autocast(dtype=torch.float64):
#         src_extrinsic, src_intrinsic = pose_encoding_to_extri_intri(source_prediction["pose_enc"], image.shape[-2:])
#         ref_extrinsic, ref_intrinsic = pose_encoding_to_extri_intri(ref_prediction["pose_enc"], image.shape[-2:])
#         pred_src_extrinsic = src_extrinsic[0] 
#         pred_ref_extrinsic = ref_extrinsic[0]
        
#         add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_src_extrinsic.size(0), 1, 4)
#         pred_src_se3 = torch.cat((pred_src_extrinsic, add_row), dim=1) 
#         pred_ref_se3 = torch.cat((pred_ref_extrinsic, add_row), dim=1) 
        
#         # Set the coordinate of the first camera as the coordinate of the world
#         # NOTE: DO NOT REMOVE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
#         src_se3 = align_to_first_camera(pred_src_se3)
#         ref_se3 = align_to_first_camera(pred_ref_se3)
        
        
#         rel_rangle_deg, rel_tangle_deg = pose_relative_error(src_se3, ref_se3)

#         Racc_5 = (rel_rangle_deg < 5).float().mean().item()
#         Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

#         print(f"R_ACC@5: {Racc_5:.4f}")
#         print(f"T_ACC@5: {Tacc_5:.4f}")

#         return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()
   
def main():
    """Main function to evaluate VGGT on UIEB dataset."""
    # Parse command-line arguments
    args = setup_args()

    # Setup device and data type
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    print("Loading model...")
    model = load_model(device, model_path=args.model_path)

    # Set random seeds
    set_random_seeds(args.seed)
    
    result_summary={}
    source_path = args.source_dir
    reference_path = args.ref_dir

    # Image processing
    print("Processing images...")
    source_images_names = sorted([
        os.path.join(source_path, fname)
        for fname in os.listdir(source_path)
        if fname.lower().endswith((".jpg", ".png"))
    ])
    source_images = load_and_preprocess_images(source_images_names).to(device)
    target_images_names = sorted([
        os.path.join(reference_path, fname)
        for fname in os.listdir(reference_path)
        if fname.lower().endswith((".jpg", ".png"))
    ])
    target_images = load_and_preprocess_images(target_images_names).to(device)
    
    print(source_images.shape, target_images.shape)
    print("Making predictions...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Add batch dimension to images for model input: (S, 3, H, W) -> (1, S, 3, H, W)
            for i in range(len(source_images)):
                source_image=source_images[i][None]
                target_image=target_images[i][None]
                
                source_prediction = model(source_image)
                target_prediction = model(target_image)
                
                source_depth = source_prediction['depth'].squeeze().cpu().numpy()
                target_depth = target_prediction['depth'].squeeze().cpu().numpy()
                
                # Depth estimation
                Pearson_corr = correlation(source_depth, target_depth)
                si_mse_value= si_mse(source_depth, target_depth)
                inlier_ratio=thresh_inliers(source_depth, target_depth, 1.03, mask=None, output_scaling_factor=100)
                mean_rel=m_rel_ae(source_depth, target_depth, mask=None, output_scaling_factor=100)

                result_summary[i] = {
                    'Pearson_corr': Pearson_corr,
                    'si_mse': si_mse_value,
                    'inlier_ratio': inlier_ratio,
                    'mean_rel': mean_rel
                }
                # # Camera pose estimation (AUC)
                # rError, tError = camera_pose_diff(source_prediction, target_prediction, device, source_image)

                # Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
                # Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
                # Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
                # Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

                # print(f"Index{i}, AUC@30: {Auc_30:.4f}, AUC@15: {Auc_15:.4f}, AUC@5: {Auc_5:.4f}, AUC@3: {Auc_3:.4f}")
                
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        WHITE = "\033[97m" 
        CYAN = "\033[36m"
        if result_summary:
            # mean_AUC_30 = np.mean([result_summary[index]["Auc_30"] for index in result_summary])
            # mean_AUC_15 = np.mean([result_summary[index]["Auc_15"] for index in result_summary])
            # mean_AUC_5 = np.mean([result_summary[index]["Auc_5"] for index in result_summary])
            # mean_AUC_3 = np.mean([result_summary[index]["Auc_3"] for index in result_summary])
            mean_inlier_ratio = np.mean([result_summary[index]["inlier_ratio"] for index in result_summary])
            mean_rel_result = np.mean([result_summary[index]["mean_rel"] for index in result_summary])
            mean_Pearson_corr = np.mean([result_summary[index]["Pearson_corr"] for index in result_summary])
            mean_si_mse = np.mean([result_summary[index]["si_mse"] for index in result_summary])
            
            print("-"*50)
            # print(f"{BOLD}{RED}Mean AUC: {RESET} {GREEN}{mean_AUC_30:.4f}{WHITE} (AUC@30){GREEN}, {mean_AUC_15:.4f}{WHITE} (AUC@15){GREEN}, {mean_AUC_5:.4f}{WHITE} (AUC@5){GREEN}, {mean_AUC_3:.4f}{WHITE} (AUC@3){RESET}")
            print(f"{BOLD}{RED}Mean inlier ratio: {RESET} {GREEN}{mean_inlier_ratio:.4f}, {BOLD}{RED}Mean rel: {RESET} {GREEN}{mean_rel_result:.4f}")
            print(f"{BOLD}{RED}Mean Pearson correlation: {RESET} {GREEN}{mean_Pearson_corr:.4f}, {BOLD}{RED}Mean si_mse: {RESET} {GREEN}{mean_si_mse:.4f}")
    
    
if __name__ == "__main__":
    main()
    
'''
Example:
 python test_UIEB.py \
    --source_dir /data/jxucm/UIEB_Dataset/source \
    --ref_dir /data/jxucm/UIEB_Dataset/reference \
    --model_path ../weights/model.pt \
    --seed 77 \
    --cuda_id 0 
'''
