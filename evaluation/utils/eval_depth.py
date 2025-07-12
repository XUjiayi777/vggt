import numpy as np
import pandas as pd
from tqdm import tqdm

# Scale-invariant depth estimation
# Pearson Correlation Coefficient
def correlation(gt, pred, mask=None):
    """Computes the Pearson correlation coefficient for predicted and ground truth depth maps.
    
    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
    
    Returns:
        Scalar that indicates the Pearson correlation coefficient. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)
    
    gt_valid = gt[mask > 0]
    pred_valid = pred[mask > 0]
    
    if len(gt_valid) < 2:
        return np.nan
    
    mu_gt = np.mean(gt_valid)
    mu_pred = np.mean(pred_valid)
    
    cov = np.mean((gt_valid - mu_gt) * (pred_valid - mu_pred))
    sigma_gt = np.std(gt_valid)
    sigma_pred = np.std(pred_valid)
    
    if sigma_gt < 1e-8 or sigma_pred < 1e-8:
        return np.nan
    
    corr = cov / (sigma_gt * sigma_pred)
    corr = corr if np.isfinite(corr) else np.nan
    
    return corr

# SI-MSE (In log space)
def si_mse(gt, pred, mask=None):
    """Computes the scale-invariant mean squared error in log space.
    
    D(y, y*) = (1/n) * sum(d_i^2) - (1/n^2) * (sum(d_i))^2
    where d_i = log(y_i) - log(y*_i)
    
    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
    
    Returns:
        Scalar that indicates the scale-invariant MSE. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)
    
    gt_valid = gt[mask > 0]
    pred_valid = pred[mask > 0]
    
    if len(gt_valid) < 1:
        return np.nan

    if np.any(pred_valid <= 0):
        pred_valid = np.maximum(pred_valid, 1e-8)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_gt = np.log(gt_valid)
        log_pred = np.log(pred_valid)
    
    if not np.all(np.isfinite(log_gt)) or not np.all(np.isfinite(log_pred)):
        return np.nan
    
    d = log_pred - log_gt  # d_i = log(y_i) - log(y*_i)
        
    # Equation 3: D(y, y*) = (1/n) * sum(d_i^2) - (1/n^2) * (sum(d_i))^2
    term1 = np.mean(d * d)  # (1/n) * sum(d_i^2)
    term2 = (np.mean(d)) ** 2  # (1/n^2) * (sum(d_i))^2
    
    si_mse_value = term1 - term2
    
    return si_mse_value if np.isfinite(si_mse_value) else np.nan

# Scale-variant depth estimation
def align_pred_to_gt(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_pixels: int = 100,
) -> tuple[float, float, np.ndarray]:
    """
    Aligns a predicted depth map to a ground truth depth map using scale and shift.
    The alignment is: gt_aligned_to_pred ≈ scale * pred_depth + shift.

    Args:
        pred_depth (np.ndarray): The HxW predicted depth map.
        gt_depth (np.ndarray): The HxW ground truth depth map.
        min_gt_depth (float): Minimum valid depth value for GT.
        max_gt_depth (float): Maximum valid depth value for GT.
        min_pred_depth (float): Minimum valid depth value for predictions.
        min_valid_pixels (int): Minimum number of valid overlapping pixels
                                 required to perform the alignment.
        robust_median_scale (bool): If True, uses median(gt/pred) for scale and then
                                    median(gt - scale*pred) for shift. Otherwise, uses
                                    least squares for both scale and shift simultaneously.

    Returns:
        tuple[float, float, np.ndarray]:
            - scale (float): The calculated scale factor. (NaN if alignment failed)
            - shift (float): The calculated shift offset. (NaN if alignment failed)
            - aligned_pred_depth (np.ndarray): The HxW predicted depth map after
                                               applying scale and shift. (Original pred_depth
                                               if alignment failed)
    """
    exclude = False
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Predicted depth shape {pred_depth.shape} must match GT depth shape {gt_depth.shape}"
        )

    # Extract valid depth values
    gt_masked = gt_depth[valid_mask]
    pred_masked = pred_depth[valid_mask]

    if len(gt_masked) < min_valid_pixels:
        print(
            f"Warning: Not enough valid pixels ({len(gt_masked)} < {min_valid_pixels}) to align. "
            "Using all pixels."
        )
        exclude = True
        gt_masked = gt_depth.reshape(-1)
        pred_masked = pred_depth.reshape(-1)


    # Handle case where pred_masked has no variance (e.g., all zeros or a constant value)
    if np.std(pred_masked) < 1e-6: # Small epsilon to check for near-constant values
        print(
            "Warning: Predicted depth values in the valid mask have near-zero variance. "
            "Scale is ill-defined. Setting scale=1 and solving for shift only."
        )
        scale = 1.0
        shift = np.mean(gt_masked) - np.mean(pred_masked) # or np.median(gt_masked) - np.median(pred_masked)
    else:
        A = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
        try:
            x, residuals, rank, s_values = np.linalg.lstsq(A, gt_masked, rcond=None)
            scale, shift = x[0], x[1]
        except np.linalg.LinAlgError as e:
            print(f"Warning: Least squares alignment failed ({e}). Returning original prediction.")
            return np.nan, np.nan, pred_depth.copy()


    aligned_pred_depth = scale * pred_depth + shift
    return scale, shift, aligned_pred_depth, exclude

    # valid_mask = torch.logical_and(
    #     gt_depth.squeeze().cpu() > 1e-3,     # filter out black background
    #     predictions["depth_conf"].squeeze().cpu() > args.depth_conf_thres
    # )
    # valid_mask = valid_mask.numpy()[0]  # Take first item in batch
    

    # align_mask = valid_mask.copy()
    
    # scale, shift, aligned_pred_depth = align_pred_to_gt(
    #     depth_map.squeeze()[0].cpu().numpy(), 
    #     gt_depth.squeeze()[0].cpu().numpy(), 
    #     align_mask
    # )

def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(valid_mean, copy=False, nan=0, posinf=0, neginf=0)

    return valid_mean, is_valid


def thresh_inliers(gt, pred, thresh, mask=None, output_scaling_factor=1.0):
    """Computes the inlier (=error within a threshold) ratio for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        thresh: Threshold for the relative difference between the prediction and ground truth.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the inlier ratio. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_1 = np.nan_to_num(gt / pred, nan=thresh+1, posinf=thresh+1, neginf=thresh+1)  # pred=0 should be an outlier
        rel_2 = np.nan_to_num(pred / gt, nan=0, posinf=0, neginf=0)  # gt=0 is masked out anyways

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(np.float32)  # 1 for inliers, 0 for outliers

    inlier_ratio, valid = valid_mean(inliers, mask)

    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan

    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the mean-relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).


    Returns:
        Scalar that indicates the mean-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    ae = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ae = np.nan_to_num(ae / gt, nan=0, posinf=0, neginf=0)

    m_rel_ae, valid = valid_mean(rel_ae, mask)

    m_rel_ae = m_rel_ae * output_scaling_factor
    m_rel_ae = m_rel_ae if valid else np.nan

    return m_rel_ae


def sq_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the squared-relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the squared-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    se = e * e  # squared error
    with np.errstate(divide='ignore', invalid='ignore'):
        sq_rel_ae = np.nan_to_num(se / gt, nan=0, posinf=0, neginf=0)

    sq_rel_ae_mean, valid = valid_mean(sq_rel_ae, mask)

    sq_rel_ae_mean = sq_rel_ae_mean * output_scaling_factor
    sq_rel_ae_mean = sq_rel_ae_mean if valid else np.nan

    return sq_rel_ae_mean


def rmse(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the root-mean-square-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the root-mean-square-error. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    se = e * e  # squared error

    mse, valid = valid_mean(se, mask)
    rmse_value = np.sqrt(mse) if valid and mse >= 0 else np.nan

    rmse_value = rmse_value * output_scaling_factor
    rmse_value = rmse_value if np.isfinite(rmse_value) else np.nan

    return rmse_value


def rmse_log(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the root-mean-square-error in log space for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the root-mean-square-error in log space. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    # Ensure both gt and pred are positive for log computation
    gt_valid = gt[mask > 0]
    pred_valid = pred[mask > 0]
    
    if len(gt_valid) < 1:
        return np.nan

    # Clamp predicted values to avoid log(0) or log(negative)
    pred_valid = np.maximum(pred_valid, 1e-8)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_gt = np.log(gt_valid)
        log_pred = np.log(pred_valid)
    
    if not np.all(np.isfinite(log_gt)) or not np.all(np.isfinite(log_pred)):
        return np.nan
    
    e_log = log_pred - log_gt
    se_log = e_log * e_log  # squared error in log space
    
    mse_log = np.mean(se_log)
    rmse_log_value = np.sqrt(mse_log) if mse_log >= 0 else np.nan

    rmse_log_value = rmse_log_value * output_scaling_factor
    rmse_log_value = rmse_log_value if np.isfinite(rmse_log_value) else np.nan

    return rmse_log_value


def sparsification(gt, pred, uncertainty, mask=None, error_fct=m_rel_ae, show_pbar=False, pbar_desc=None):
    """Computes the sparsification curve for a predicted and ground truth depth map and a given ranking.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        uncertainty: Uncertainty measure for the predicted depth map. Numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        error_fct: Function that computes a metric between ground truth and prediction for the sparsification curve.
        show_pbar: Show progress bar.
        pbar_desc: Prefix for the progress bar.

    Returns:
        Pandas Series with (sparsification_ratio, error_ratio) values of the sparsification curve.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    y, x = np.unravel_index(np.argsort((uncertainty - uncertainty.min() + 1) * mask, axis=None), uncertainty.shape)
    # (masking out values that are anyways not considered for computing the error)
    ranking = np.flip(np.stack((x, y), axis=1), 0).tolist()

    num_valid = np.sum(mask.astype(bool))
    sparsification_steps = [int((num_valid / 100) * i) for i in range(100)]

    base_error = error_fct(gt=gt, pred=pred, mask=mask)
    sparsification_x, sparsification_y = [], []

    num_masked = 0
    pbar = tqdm(total=num_valid, desc=pbar_desc,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}',
                disable=not show_pbar, unit="removed pixels", ncols=80)
    for x, y in ranking:
        if num_masked >= num_valid:
            break

        if mask[y, x] == 0:
            raise RuntimeError('This should never happen. If it happens, please open a GitHub issue.')

        if num_masked in sparsification_steps:
            cur_error = error_fct(gt=gt, pred=pred, mask=mask)
            sparsification_frac = num_masked / num_valid
            error_frac = cur_error / base_error
            if np.isfinite(cur_error):
                sparsification_x.append(sparsification_frac)
                sparsification_y.append(error_frac)

        mask[y, x] = 0
        num_masked += 1
        pbar.update(1)

    pbar.close()
    x = np.linspace(0, 0.99, 100)

    if len(sparsification_x) > 1:
        sparsification = np.interp(x, sparsification_x, sparsification_y)
    else:
        sparsification = np.array([np.nan] * 100, dtype=np.float64)
    sparsification = pd.Series(sparsification, index=x)

    return sparsification