import os
import torch
import numpy as np
import gzip
import json
import logging
import warnings
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_depth_maps

from utils.eval_pose import estimate_camera_pose,calculate_auc_np
from utils.eval_depth import thresh_inliers,m_rel_ae,pointwise_rel_ae,align_pred_to_gt
from utils.general import set_random_seeds,load_model
from ba import run_vggt_with_ba
import argparse
import random
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
    parser.add_argument('--mode', type=str, required=True, choices=['Easy','Hard'], help='Dataset mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (only test on specific category)')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    # parser.add_argument('--log_file_path', type=str, default='../logs/log_tartanair.txt', help='Path to log file')
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
   
def main():
    """Main function to evaluate VGGT on TartanAir dataset."""
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
    dataset_path = f"{args.TAir_dir}/{args.mode}"
    for seq in sorted(os.listdir(dataset_path)):
        seq_path=os.path.join(dataset_path, seq)
 
        # Image processing
        print("Processing images...")
        image_path = os.path.join(seq_path, f"image_{args.camera}")
        image_names = sorted([
            os.path.join(image_path, fname)
            for fname in os.listdir(image_path)
            if fname.lower().endswith((".jpg", ".png"))
        ])

        # Random sampling
        assert len(image_names) >= args.num_frames, "Not enough frames to sample"
        sample_indices = sorted(random.sample(range(len(image_names)), args.num_frames))
        print("Number of sampled frames:", args.num_frames)
        print("Sample indices:",sample_indices)
        sample_image_names = [image_names[i] for i in sample_indices]
        images = load_and_preprocess_images(sample_image_names).to(device)

        # Pose processing
        print("Processing poses...")
        pose_path = os.path.join(seq_path, f"pose_{args.camera}.txt")
        pose=np.loadtxt(pose_path)
        sample_extrinsic = [ned_to_c2w(pose[i]) for i in sample_indices]
        gt_extrinsic = np.array(sample_extrinsic)
    
        # Depth processing
        print("Processing depth maps...")
        depth_path = os.path.join(seq_path, f"depth_{args.camera}")
        depth_names = sorted([
            os.path.join(depth_path, depth)
            for depth in os.listdir(depth_path)
        ])
        sample_depth = [np.load(depth_names[i]) for i in sample_indices]
        gt_depths=preprocess_depth_maps(sample_depth).squeeze()
    
        print("Making predictions...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        # Camera pose estimation (AUC)
        rError, tError = estimate_camera_pose(predictions, images, args.num_frames ,args.use_ba, device, gt_extrinsic)
        
        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)
            
        # print("="*80)
        # # Print summary results
        # print(f"\nSummary of AUC results:({seq})")
        # print("-"*50)
        # print(f"AUC: {Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3)")
        
        # Depth estimation:
        pre_depths = predictions['depth'].squeeze().cpu().numpy()
        
        # predictions.keys(): ['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images']
        # print("pre_depths",pre_depths.shape, type(pre_depths))
        # print("gt_depth",gt_depths.shape, type(gt_depths))
        # print("pre_depth_conf",pre_depth_conf.shape, type(pre_depth_conf))
        
        valid_mask = torch.logical_and(
        gt_depths.cpu() > 1e-3,     # filter out black background
        predictions['depth_conf'].cpu() > 10
        ).squeeze().numpy()
        gt_depths= gt_depths.cpu().numpy()
        
        align_pred_depths=[]
        # print(valid_mask.shape)
        move_out= False
        for i in range(args.num_frames):  
            valid_mask_idx = valid_mask[i]  
            scale, shift, aligned_pred_depth, exclude = align_pred_to_gt(
                pre_depths[i], 
                gt_depths[i],
                valid_mask_idx
            )
            if exclude:
                move_out = True
                continue
            align_pred_depths.append(aligned_pred_depth)
        if move_out:
            continue
        align_pred_depths=np.array(align_pred_depths)
        
        inlier_ratio=thresh_inliers(gt_depths,align_pred_depths, 1.03, mask=valid_mask,output_scaling_factor=100)
        mean_rel=m_rel_ae(gt_depths,align_pred_depths,mask=valid_mask,output_scaling_factor=100)
        
        # with open(args.log_file_path, 'a') as log_file:
        #     log_file.write("="*80)
        #     log_file.write(f"\nSummary of depth estimation results:({seq})\n")
        #     log_file.write("-" * 50 + "\n")
        #     log_file.write(f"inlier_ratio:{inlier_ratio:.4f} \n")
        #     log_file.write(f"mean_rel{mean_rel:.4f}")
            
        # print("="*80)
        # print(f"\nSummary of depth estimation results: ({seq})")
        # print("-"*50)
        # print("inlier_ratio",inlier_ratio)
        # print("m_rel_ae",mean_rel)
        
        result_summary[seq] = {
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3,
            "inlier_ratio": inlier_ratio,
            "mean_rel": mean_rel,
        }
    
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    WHITE = "\033[97m" 
    CYAN = "\033[36m"
    
    for sequence in sorted(result_summary.keys()):
        print(f"{BOLD}{BLUE}{sequence:<5}: {RESET} {GREEN}{result_summary[sequence]['Auc_30']:.4f}{WHITE} (AUC@30){GREEN}, {result_summary[sequence]['Auc_15']:.4f}{WHITE} (AUC@15){GREEN}, {result_summary[sequence]['Auc_5']:.4f}{WHITE} (AUC@5){GREEN}, {result_summary[sequence]['Auc_3']:.4f}{WHITE} (AUC@3){RESET}")
        print(f'{BOLD}inlier_ratio:{RESET} {CYAN}{result_summary[sequence]["inlier_ratio"]:.4f}{RESET}, {BOLD}mean_rel:{RESET} {CYAN}{result_summary[sequence]["mean_rel"]:.4f}{RESET}')

        
    if result_summary:
        mean_AUC_30 = np.mean([result_summary[sequence]["Auc_30"] for sequence in result_summary])
        mean_AUC_15 = np.mean([result_summary[sequence]["Auc_15"] for sequence in result_summary])
        mean_AUC_5 = np.mean([result_summary[sequence]["Auc_5"] for sequence in result_summary])
        mean_AUC_3 = np.mean([result_summary[sequence]["Auc_3"] for sequence in result_summary])
        mean_inlier_ratio = np.mean([result_summary[sequence]["inlier_ratio"] for sequence in result_summary])
        mean_rel_result = np.mean([result_summary[sequence]["mean_rel"] for sequence in result_summary])
        
        print("-"*50)
        print(f"{BOLD}{RED}Mean AUC: {RESET} {GREEN}{mean_AUC_30:.4f}{WHITE} (AUC@30){GREEN}, {mean_AUC_15:.4f}{WHITE} (AUC@15){GREEN}, {mean_AUC_5:.4f}{WHITE} (AUC@5){GREEN}, {mean_AUC_3:.4f}{WHITE} (AUC@3){RESET}")
        print(f"{BOLD}{RED}Mean inlier ratio: {RESET} {GREEN}{mean_inlier_ratio:.4f}, {BOLD}{RED}Mean rel: {RESET} {GREEN}{mean_rel_result:.4f}")
    
if __name__ == "__main__":
    main()
    

'''
Example:
 python test_TartanAir.py \
    --model_path ../weights/model_tracker_fixed_e20.pt \
    --TAir_dir ../data/TartanAir/ocean/ \
    --mode Easy \
    --num_frames 20 \
    --seed 77 \
    --cuda_id 7 \
    --camera left 
'''