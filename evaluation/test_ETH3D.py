import os
import torch
import numpy as np
import gzip
import json
import logging
import warnings
import sys
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_depth_maps
import cv2
from chamfer_distance import ChamferDistance as chamfer_dist
from utils.eval_pose import calculate_auc_np, align_to_first_camera, se3_to_relative_pose_error
from utils.eval_depth import thresh_inliers,m_rel_ae,sq_rel_ae,align_pred_to_gt,correlation,si_mse,rmse,rmse_log
from utils.eval_pose import rotation_angle, translation_angle,closed_form_inverse_se3
from utils.general import set_random_seeds,load_model
from vggt.utils.geometry import unproject_depth_map_to_point_map, project_world_points_to_cam
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_util import predictions_to_glb
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch

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
    """Set up command-line arguments for the ETH3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on ETH3D dataset')
    parser.add_argument( "--data_dir", type=str, default="/data/jxucm/ETH3D",
        help="Path to the ETH dataset directory containing scenes")
    parser.add_argument( "--scene", type=str, default=None,
        help="Specific scene to process (e.g., meadow, office). If not specified, processes all scenes.")
    # parser.add_argument("--num_frames", type=int, default=15, help="Number of frames to evaluate per scene")
    # parser.add_argument("--stride", type=int, default=1, help="Stride for selecting frames")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--cuda_id',type=str, help='CUDA device ID')
    parser.add_argument('--model_path', type=str, default='../weights/model.pt', help='Path to the pre-trained model')
    parser.add_argument('--viz_output', type=str, default=None, help='Path to save visualization output')
    return parser.parse_args()

def quat2rot(quats):
    """Convert quaternions to rotation matrices.
    
    Args:
        quats: numpy array of shape (N, 4) where each quaternion is [qw, qx, qy, qz]
        
    Returns:
        rotation_matrices: numpy array of shape (N, 3, 3)
    """
    rotation_matrices = []
    for frame_idx in range(quats.shape[0]):
        qw, qx, qy, qz = quats[frame_idx]
        
        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Convert to rotation matrix
        rotation_matrix = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
        ])
        rotation_matrices.append(rotation_matrix)
    
    return np.stack(rotation_matrices, axis=0).astype(np.float32)

def parse_camera_param_file(extrinsic_file, intrinsic_file):
    """
    Parse COLMAP extrinsic and intrinsic files to return matched matrices.
    
    Args:
        extrinsic_file: Path to the extrinsic.txt file
        intrinsic_file: Path to the intrinsic.txt file
        
    Returns:
        extrinsic_matrices: (N, 3, 4) numpy array of extrinsic matrices
        intrinsic_matrices: (N, 3, 3) numpy array of intrinsic matrices
    """
    # First, parse the intrinsic file to create a camera_id -> intrinsic_matrix mapping
    intrinsic_dict = {}
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Parse: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
            parts = line.split()
            camera_id = int(parts[0])
            fx = float(parts[4])
            fy = float(parts[5])
            cx = float(parts[6])
            cy = float(parts[7])
            
            intrinsic_matrix = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])
            intrinsic_dict[camera_id] = intrinsic_matrix
    
    print(f"Found {len(intrinsic_dict)} cameras in intrinsics: {list(intrinsic_dict.keys())}")
    
    extrinsic_data = []
    camera_ids = []
    
    with open(extrinsic_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Parse: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            parts = line.split()
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            
            extrinsic_data.append([qw, qx, qy, qz, tx, ty, tz])
            camera_ids.append(camera_id)
    
    extrinsic_data = np.array(extrinsic_data)
    quaternions = extrinsic_data[:, :4]  # [qw, qx, qy, qz]
    translations = extrinsic_data[:, 4:7]  # [tx, ty, tz]
    pdb.set_trace()

    rotation_matrices = quat2rot(quaternions)
    
    num_frames = len(extrinsic_data)
    extrinsic_matrices = np.zeros((num_frames, 3, 4))
    extrinsic_matrices[:, :3, :3] = rotation_matrices
    extrinsic_matrices[:, :3, 3] = translations
    
    # Match intrinsic matrices to extrinsics based on camera IDs
    intrinsic_matrices = []
    for cam_id in camera_ids:
        if cam_id in intrinsic_dict:
            intrinsic_matrices.append(intrinsic_dict[cam_id])
        else:
            print(f"Warning: Camera ID {cam_id} not found in intrinsics file")
            # Use the first available camera as fallback
            fallback_id = list(intrinsic_dict.keys())[0]
            intrinsic_matrices.append(intrinsic_dict[fallback_id])
            print(f"Using camera {fallback_id} as fallback")
    
    intrinsic_matrices = np.array(intrinsic_matrices)
    
    print(f"Matched {len(camera_ids)} images with camera IDs: {set(camera_ids)}")
    
    return extrinsic_matrices, intrinsic_matrices

def estimate_camera_pose_ETH3D(pred_extrinsic, gt_extrinsic, num_frames, device):
    """
    Process a single sequence and compute pose errors.

    Args:
        pred_extrinsic: Predicted extrinsics from the model
        gt_extrinsic: Ground truth extrinsics 
        num_frames: Number of frames in the sequence
        device: Device to run on
        
    Returns:
        rError: Rotation errors
        tError: Translation errors
    """

    with torch.cuda.amp.autocast(dtype=torch.float64):
        pred_extrinsic = pred_extrinsic.to(device) # Predicted extrinsic w2c
        gt_extrinsic = gt_extrinsic.to(device)
        
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)
        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1) #w2c
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

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
    """Main function to evaluate VGGT on ETH3D dataset."""
    args = setup_args()
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    print("Loading model...")
    model = load_model(device, model_path=args.model_path)

    # Set random seeds
    set_random_seeds(args.seed)
    
    # Evaluation
    all_auc_30 = []
    all_auc_15 = []
    all_auc_5 = []
    all_auc_3 = []
    all_Pearson_corr = []
    all_si_mse = []
    all_inlier_ratio = []
    all_abs_rel = []
    all_sq_rel = []
    all_rmse = []
    all_rmse_log = []
    all_accuracy = []
    all_completeness = []
    all_chamfer_dis = []

    scene_dirs=[]
    if args.scene is None:
        scene_dirs = [d for d in os.listdir(args.data_dir) 
                     if os.path.isdir(os.path.join(args.data_dir, d))]
        scene_dirs.sort()
        if not scene_dirs:
            print(f"No scene directories found in {args.data_dir}")
            return
    else:
        scene_dirs = [args.scene]
        if not os.path.exists(os.path.join(args.data_dir, args.scene)):
            print(f"Scene directory does not exist: {scene_path}")
            return
    
    for scene in scene_dirs:
        if scene == "facade":
            continue
        print(f"\n--- Processing scene {scene} ---")
        scene_path = os.path.join(args.data_dir, scene)
        depth_path = os.path.join(scene_path, "depths")
        image_path = os.path.join(scene_path, "images")
        
        # GT image 
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_list = []
            
        if os.path.exists(image_path):
            for filename in os.listdir(image_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_list.append(os.path.join(image_path,filename))
            image_list.sort()  
            print(f"Found {len(image_list)} images in {image_path}")
            
        images = load_and_preprocess_images(image_list).to(device)
        print("images shape", images.shape)
        
        # GT camera
        extrinsic_file = os.path.join(scene_path, "cams/extrinsic.txt")
        intrinsic_file = os.path.join(scene_path, "cams/intrinsic.txt")
            
        gt_extrinsic, gt_intrinsic = parse_camera_param_file(extrinsic_file, intrinsic_file)
        print(f"Loaded {len(gt_extrinsic)} extrinsic matrices")
        print(f"Extrinsic matrices shape: {gt_extrinsic.shape}")
        print(f"Intrinsic matrices shape: {gt_intrinsic.shape}")   
        
        # GT depths
        gt_depths = None
        if os.path.exists(depth_path):
            depth_arrays = []
            for image_idx in range(len(gt_extrinsic)):
                depth_file_path = os.path.join(depth_path, f"{image_idx:04d}.png")
                if os.path.exists(depth_file_path):
                    depth_img = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH)
                    depth_arrays.append(depth_img)
            
            if depth_arrays:
                depth_array = np.stack(depth_arrays, axis=0)
                gt_depths = preprocess_depth_maps(depth_array).squeeze()
                print(f"Loaded {len(depth_arrays)} depth maps, shape: {gt_depths.shape}")
            else:
                print("No depth files found")
         
        # Normalize GT
        world_coords_points, camera_coords_points = (
            unproject_depth_map_to_point_map(gt_depths, gt_extrinsic, gt_intrinsic)
        )        
        new_extrinsics, new_cam_points, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
            torch.from_numpy(gt_extrinsic).unsqueeze(0), torch.from_numpy(camera_coords_points).unsqueeze(0), torch.from_numpy(world_coords_points).unsqueeze(0), gt_depths.unsqueeze(0),
            point_masks =  gt_depths.unsqueeze(0) > 1e-8)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                    
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic   
                        
            # Camera pose estimation (AUC)
            rError, tError = estimate_camera_pose_ETH3D(predictions["extrinsic"][0], new_extrinsics.squeeze(0), len(gt_extrinsic), device)
            
            Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
            Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
            Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
            Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)
            
            print(f"Scene {scene} - AUC@30: {Auc_30:.4f}, AUC@15: {Auc_15:.4f}, AUC@5: {Auc_5:.4f}, AUC@3: {Auc_3:.4f}")
            
            all_auc_30.append(Auc_30)
            all_auc_15.append(Auc_15)
            all_auc_5.append(Auc_5)
            all_auc_3.append(Auc_3)
            
            # Depth estimation 
            valid_mask = torch.logical_and(
            new_depths.cpu().squeeze(0) > 1e-3,     # filter out black background
            predictions['depth_conf'].cpu() > 1e-3,
            ).squeeze().numpy()
            
            gt_depths = new_depths.cpu().squeeze(0) .numpy()
            pre_depths = predictions['depth'].squeeze().cpu().numpy()
            
            Pearson_corr = correlation(gt_depths, pre_depths, mask=valid_mask)
            si_mse_value= si_mse(gt_depths, pre_depths, mask=valid_mask)
            
            inlier_ratio=thresh_inliers(gt_depths, pre_depths, 1.25, mask=valid_mask)
            abs_rel=m_rel_ae(gt_depths, pre_depths, mask=valid_mask)
            sq_rel=sq_rel_ae(gt_depths, pre_depths, mask=valid_mask)
            rmse_value= rmse(gt_depths, pre_depths, mask=valid_mask)
            rmse_log_value= rmse_log(gt_depths, pre_depths, mask=valid_mask)
            
            print(f"Scene {scene} - Pearson Correlation: {Pearson_corr:.4f}, SI-MSE: {si_mse_value:.4f}, Inlier Ratio: {inlier_ratio:.4f}")
            print(f"abs_rel: {abs_rel:.4f}, sq_rel: {sq_rel:.4f}, rmse: {rmse_value:.4f}, rmse_log: {rmse_log_value:.4f}")
            
            all_Pearson_corr.append(Pearson_corr)
            all_si_mse.append(si_mse_value)
            all_inlier_ratio.append(inlier_ratio)
            all_abs_rel.append(abs_rel)
            all_sq_rel.append(sq_rel)
            all_rmse.append(rmse_value)
            all_rmse_log.append(rmse_log_value)
            
            # Accuracy, completeness and Chamfer distance
            chd = chamfer_dist()
            pre_pt = predictions['world_points'].squeeze(0).view(len(gt_extrinsic),-1,3).to(dtype=torch.float32, device=device)
            gt_pt = new_world_points.squeeze(0).view(len(gt_extrinsic),-1,3).to(dtype=torch.float32, device=device)
            dist1, dist2, idx1, idx2 = chd(pre_pt,gt_pt)
            dist3, _,_,_ = chd(gt_pt, pre_pt)
            accuracy =torch.mean(dist1).cpu().numpy()
            completeness = torch.mean(dist2).cpu().numpy()
            chamfer_dis = (accuracy + completeness) / 2
            print(f"Accuracy: {accuracy.item():.4f}, Completeness: {completeness.item():.4f}, Chamfer Distance: {chamfer_dis.item():.4f}")
            
            all_accuracy.append(accuracy.item())
            all_completeness.append(completeness.item())
            all_chamfer_dis.append(chamfer_dis)
            
            # # Virtualization
            # if args.viz_output is not None:
            #     for key in predictions.keys():
            #         if isinstance(predictions[key], torch.Tensor):
            #             predictions[key] = predictions[key].cpu().numpy().squeeze(0)  
            #     predictions['pose_enc_list'] = None # remove pose_enc_list
            #     world_points, _ = unproject_depth_map_to_point_map(predictions["depth"], predictions["extrinsic"], predictions["intrinsic"])
            #     predictions["world_points_from_depth"] = world_points
                
            #     os.makedirs(args.viz_output, exist_ok=True)
            #     glbfile = os.path.join(
            #         args.viz_output,
            #         f"vkitti_{args.sceneid}_{args.scene}_{args.stride}.glb"
            #     )
                
            #     # Convert predictions to GLB
            #     glbscene = predictions_to_glb(
            #         predictions,
            #         conf_thres=10,
            #         filter_by_frames="all",
            #         mask_black_bg=False,
            #         mask_white_bg=False,
            #         show_cam=False,
            #         mask_sky=False,
            #         target_dir=args.viz_output,
            #         prediction_mode="Depth",
            #     )
            #     glbscene.export(file_obj=glbfile)

    mean_auc_30 = np.mean(all_auc_30)
    mean_auc_15 = np.mean(all_auc_15)
    mean_auc_5 = np.mean(all_auc_5)
    mean_auc_3 = np.mean(all_auc_3)
    mean_Pearson_corr = np.mean(all_Pearson_corr)
    mean_si_mse = np.mean(all_si_mse)
    mean_inlier_ratio = np.mean(all_inlier_ratio)
    mean_abs_rel = np.mean(all_abs_rel)
    mean_sq_rel = np.mean(all_sq_rel)
    mean_rmse = np.mean(all_rmse)
    mean_rmse_log = np.mean(all_rmse_log)
    mean_accuracy = np.mean(all_accuracy)
    mean_completeness = np.mean(all_completeness)
    mean_chamfer_dis = np.mean(all_chamfer_dis)
    
    print(f"\n--- Mean AUC across all scenes ---")
    print(f"Mean AUC@30: {mean_auc_30:.4f}, Mean AUC@15: {mean_auc_15:.4f}, Mean AUC@5: {mean_auc_5:.4f}, Mean AUC@3: {mean_auc_3:.4f}")
    print(f"Mean Pearson Correlation: {mean_Pearson_corr:.4f}, Mean SI-MSE: {mean_si_mse:.4f}, Mean Inlier Ratio: {mean_inlier_ratio:.4f}")  
    print(f"Mean abs_rel: {mean_abs_rel:.4f}, Mean sq_rel: {mean_sq_rel:.4f}, Mean RMSE: {mean_rmse:.4f}, Mean RMSE Log: {mean_rmse_log:.4f}") 
    print(f"Mean Accuracy: {mean_accuracy:.4f}, Mean Completeness: {mean_completeness:.4f}, Mean Chamfer Distance: {mean_chamfer_dis:.4f}")
    
if __name__ == "__main__":
    main()
    
'''
Example:
 python test_ETH3D.py \
    --data_dir  /data/jxucm/ETH3D \
    --seed 77 \
    --cuda_id 0 \
    --model_path ../weights/model.pt 
'''
