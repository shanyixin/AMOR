"""Model checkpoint helpers."""

import os
from pathlib import Path

import torch

from model import GLCNet


def save_net(opt, logger, model, optimizer, epoch: int, iter_step: int, is_best: bool = False):
    checkpoint_dir = Path(opt["path"]["checkpoint"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_best:
        model_path = checkpoint_dir / "best_gen_G.pth"
        optimizer_path = checkpoint_dir / "best_opt.pth"
    else:
        model_path = checkpoint_dir / f"I{iter_step}_E{epoch}_gen_G.pth"
        optimizer_path = checkpoint_dir / f"I{iter_step}_E{epoch}_opt.pth"

    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, model_path)
    torch.save(
        {
            "epoch": epoch,
            "iter": iter_step,
            "scheduler": None,
            "optimizer": optimizer.state_dict(),
        },
        optimizer_path,
    )

    label = "BEST model" if is_best else "model"
    logger.info("Saved %s to [%s].", label, model_path)


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def upload_model(opt, logger, device):
    checkpoint_path = opt["path"]["resume_state"]
    if checkpoint_path is None:
        logger.info("No checkpoint path was provided.")
        return None

    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint does not exist: %s", checkpoint_path)
        return None

    model = GLCNet(number_classes=opt["datasets"]["test"]["num_landmarks"]).to(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint)
    logger.info("Loaded %s from [%s].", model.__class__.__name__, checkpoint_path)
    return model


def calculate_dice(pred, target, threshold=0.004):
    pred_mask = (pred < threshold).float()
    target_mask = (target < threshold).float()
    intersection = (pred_mask * target_mask).sum()
    dice_score = (2 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-5)
    return dice_score.item()
