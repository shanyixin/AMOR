"""Adaptive landmark weighting for coordinate losses."""

import numpy as np
import torch


class AdaptiveLandmarkWeights:
    """Track landmark-wise errors and periodically update coordinate-loss weights."""

    def __init__(
        self,
        num_landmarks,
        start_epoch=100,
        update_interval=15,
        eps=1e-6,
        min_weight=0.8,
        max_weight=1.5,
    ):
        self.num_landmarks = num_landmarks
        self.start_epoch = start_epoch
        self.update_interval = update_interval
        self.eps = eps
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.current_weights = np.ones(num_landmarks, dtype=np.float32)
        self.accumulated_errors = np.zeros(num_landmarks, dtype=np.float32)
        self.num_samples = 0

    def accumulate(self, pred_coords: torch.Tensor, target_coords: torch.Tensor, mode: str = "train"):
        if mode != "train":
            return
        errors = torch.norm(pred_coords - target_coords, dim=2).detach().cpu().numpy()
        self.accumulated_errors += errors.sum(axis=0)
        self.num_samples += pred_coords.shape[0]

    def update_if_needed(self, current_epoch, logger=None):
        should_update = (
            current_epoch >= self.start_epoch
            and current_epoch % self.update_interval == 0
            and self.num_samples > 0
        )

        if should_update:
            average_errors = self.accumulated_errors / self.num_samples
            inverse_errors = 1.0 / (average_errors + self.eps)
            normalized = inverse_errors / np.mean(inverse_errors)
            clipped = np.clip(normalized, self.min_weight, self.max_weight)
            self.current_weights = clipped * self.num_landmarks / np.sum(clipped)
            message = f"Updated landmark weights: {self.current_weights}"
        elif current_epoch % self.update_interval == 0 and self.num_samples > 0:
            message = f"Landmark weights unchanged: {self.current_weights}"
        else:
            message = None

        if message:
            logger.info(message) if logger else print(message)

        self.accumulated_errors[:] = 0.0
        self.num_samples = 0
        return self.current_weights

    def get_weights_tensor(self, device):
        return torch.tensor(self.current_weights, device=device).unsqueeze(0)
