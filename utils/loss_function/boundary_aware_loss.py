"""Boundary-constrained Morphology Coordinate Loss (BMC Loss)."""

import torch
import torch.nn.functional as F


def _normalize_points(points, width, height, depth):
    scale = torch.tensor([width, height, depth], device=points.device, dtype=points.dtype)
    return points / scale


def _normalize_prior(prior, eps=1e-6):
    while prior.dim() > 4:
        prior = prior.squeeze(1)
    prior = prior.float()
    max_value = prior.amax(dim=[1, 2, 3], keepdim=True).clamp(min=eps)
    min_value = prior.amin(dim=[1, 2, 3], keepdim=True)
    return ((prior - min_value) / (max_value - min_value + eps)).clamp(0.0, 1.0)


def _response_distribution(heatmaps, eps=1e-6):
    positive = heatmaps.clamp_min(0.0)
    mass = positive.sum(dim=[2, 3, 4], keepdim=True).clamp(min=eps)
    return positive / mass


def differentiable_center_of_mass(heatmaps, eps=1e-6):
    """Decode 3D heatmaps into differentiable xyz coordinates."""

    batch, channels, depth, height, width = heatmaps.shape
    prob = _response_distribution(heatmaps, eps)

    grid_z = torch.arange(depth, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, depth, 1, 1)
    grid_y = torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 1, height, 1)
    grid_x = torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 1, 1, width)

    center_z = (prob * grid_z).sum(dim=[2, 3, 4], keepdim=True)
    center_y = (prob * grid_y).sum(dim=[2, 3, 4], keepdim=True)
    center_x = (prob * grid_x).sum(dim=[2, 3, 4], keepdim=True)
    return torch.cat([center_x, center_y, center_z], dim=-1).view(batch, channels, 3)


def distribution_regularization(heatmaps, eps=1e-6):
    """Lreg: peak concentration plus normalized second-order spatial spread."""

    batch, channels, depth, height, width = heatmaps.shape
    prob = _response_distribution(heatmaps, eps)
    coords = differentiable_center_of_mass(heatmaps, eps)

    grid_z = torch.arange(depth, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, depth, 1, 1)
    grid_y = torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 1, height, 1)
    grid_x = torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 1, 1, width)

    cx = coords[..., 0].view(batch, channels, 1, 1, 1)
    cy = coords[..., 1].view(batch, channels, 1, 1, 1)
    cz = coords[..., 2].view(batch, channels, 1, 1, 1)
    variance = (
        (prob * (grid_x - cx).pow(2)).sum(dim=[2, 3, 4])
        + (prob * (grid_y - cy).pow(2)).sum(dim=[2, 3, 4])
        + (prob * (grid_z - cz).pow(2)).sum(dim=[2, 3, 4])
    )
    spread = variance / float(width**2 + height**2 + depth**2)

    peak = heatmaps.amax(dim=[2, 3, 4])
    peak_loss = F.smooth_l1_loss(peak, torch.ones_like(peak), reduction="mean")
    return peak_loss + spread.mean()


def boundary_prior_consistency(heatmaps, boundary_prior, eps=1e-6):
    """Lpri: penalize normalized response energy in the complementary barrier field."""

    prior = _normalize_prior(boundary_prior, eps).unsqueeze(1)
    barrier = 1.0 - prior
    response = heatmaps.clamp_min(0.0)
    numerator = (response * barrier).sum(dim=[2, 3, 4])
    denominator = response.sum(dim=[2, 3, 4]).clamp(min=eps)
    return (numerator / denominator).mean()


def coordinate_refinement_loss(heatmaps, target_points, landmark_weight_scheduler=None, mode="train", eps=1e-6):
    """Lref: adaptive weighted coordinate supervision after center-of-mass decoding."""

    _, _, depth, height, width = heatmaps.shape
    pred_coords = differentiable_center_of_mass(heatmaps, eps)
    pred_norm = _normalize_points(pred_coords, width, height, depth)
    target_norm = _normalize_points(target_points.float(), width, height, depth)

    if landmark_weight_scheduler is not None:
        landmark_weight_scheduler.accumulate(pred_norm, target_norm, mode)
        weights = landmark_weight_scheduler.get_weights_tensor(heatmaps.device)
    else:
        weights = heatmaps.new_ones(1, heatmaps.shape[1])

    squared_error = (pred_norm - target_norm).pow(2).sum(dim=2)
    return (squared_error * weights).mean()


def bmc_loss(
    pred,
    target,
    target_points,
    seg_boundary,
    landmark_weight_scheduler=None,
    mode="train",
    alpha=1.0,
    beta=3.0,
    eps=1e-6,
):
    """Paper-aligned BMC Loss.

    Returns the historical six-value tuple expected by training and evaluation:
    heatmap_loss, boundary_loss, concentrate_loss, peak_loss, coord_loss, total_loss.
    The first and fourth slots are kept as zero tensors because BMC merges peak and
    spread into Lreg.
    """

    heatmap_loss = pred.new_tensor(0.0)
    regularization_loss = distribution_regularization(pred, eps)
    prior_loss = boundary_prior_consistency(pred, seg_boundary, eps)
    peak_loss = pred.new_tensor(0.0)
    coord_loss = coordinate_refinement_loss(pred, target_points, landmark_weight_scheduler, mode, eps)
    total_loss = alpha * regularization_loss + beta * prior_loss + coord_loss
    return heatmap_loss, prior_loss, regularization_loss, peak_loss, coord_loss, total_loss
