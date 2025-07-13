import torch
import random
import numpy as np
from vggt.models.vggt import VGGT
from .eval_pose import align_to_first_camera,closed_form_inverse_se3

def load_model(device, model_path):
    """
    Load the VGGT model.

    Args:
        device: Device to load the model on
        model_path: Path to the model checkpoint

    Returns:
        Loaded VGGT model
    """

    model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    print(f"USING {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    return model

def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_cameras_and_depths(camera_c2w, depths, mask, width=640, height=640, focal_length=320):
    """
    Directly normalize cameras, depths, and 3D points to canonical coordinate frame and scale.
    
    This function combines create_dataset_batch and normalize_cameras_depths_with_pts3d
    to directly process camera poses, depths, and masks without intermediate batching.
    
    Args:
        camera_c2w: Camera-to-world matrices (num_views, 4, 4)
        depths: Depth maps (num_views, H, W)
        mask: Validity mask for depth maps (num_views, H, W)
        width: Image width in pixels (default: 640)
        height: Image height in pixels (default: 640)
        focal_length: Focal length in pixels (default: 320)
    
    Returns:
        tuple: (normalized_c2w, normalized_pts3d, scene_scale, normalized_depth)
            - normalized_c2w: Normalized camera-to-world matrices (num_views, 4, 4)
            - normalized_pts3d: Normalized 3D points (num_views, H, W, 3)
            - scene_scale: Scene scale factor used for normalization
            - normalized_depth: Normalized depth maps (num_views, H, W)
    """
    num_views = depths.shape[0]
    H, W = depths.shape[-2:]
    
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=camera_c2w.device),
        torch.arange(W, dtype=torch.float32, device=camera_c2w.device),
        indexing='ij'
    )
    
    x_norm = (x - width / 2) / focal_length
    y_norm = (y - height / 2) / focal_length
    
    # Create 3D points in camera coordinates
    pts3d_camera = torch.stack([
        x_norm * depths,  # X
        y_norm * depths,  # Y 
        depths            # Z
    ], dim=-1)  # (num_views, H, W, 3)
    
    # Transform 3D points to world coordinates
    pts3d_homo = torch.cat([pts3d_camera, torch.ones_like(pts3d_camera[..., :1])], dim=-1)
    pts3d_reshaped = pts3d_homo.view(num_views, H*W, 4)
    pts3d_transformed = torch.bmm(camera_c2w, pts3d_reshaped.transpose(-1, -2))
    pts3d_world = pts3d_transformed.transpose(-1, -2).view(num_views, H, W, 4)[..., :3]
    
    # Express all cameras in the first camera's coordinate system
    # Convert c2w to w2c for align_to_first_camera function
    camera_w2c = closed_form_inverse_se3(camera_c2w)
    aligned_w2c = align_to_first_camera(camera_w2c)
    transformed_camera_c2w = closed_form_inverse_se3(aligned_w2c)
    
    # Transform 3D points to first camera coordinate system
    pts3d_world_homo = torch.cat([pts3d_world, torch.ones_like(pts3d_world[..., :1])], dim=-1)
    pts3d_world_reshaped = pts3d_world_homo.flatten(1, 2).transpose(1, 2)  # (num_views, 4, H*W)
    transformed_pts3d_homo = torch.bmm(aligned_w2c, pts3d_world_reshaped)  # (num_views, 4, H*W)
    transformed_pts3d = transformed_pts3d_homo.transpose(1, 2)[..., :3]  # (num_views, H*W, 3)
    transformed_pts3d = transformed_pts3d.view(num_views, H, W, 3)
    
    # Compute scene scale as average distance of valid points to origin
    valid_points = transformed_pts3d.flatten(0, 2)[mask.flatten(0, 2).bool()]
    scene_scale = torch.norm(valid_points, dim=-1).mean()
    
    # Normalize cameras, 3D points, and depths by scene scale
    normalized_c2w = transformed_camera_c2w.clone()
    normalized_c2w[:, :3, 3] = normalized_c2w[:, :3, 3] / scene_scale
    normalized_pts3d = transformed_pts3d / scene_scale
    normalized_depth = depths / scene_scale
    
    return normalized_c2w, normalized_pts3d, scene_scale, normalized_depth
