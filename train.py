"""Train the 3D shoulder landmark localization model."""

import argparse
import logging
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.logger as Logger
from model import GLCNet
from utils.landmark_adaptive_weight import AdaptiveLandmarkWeights
from utils.loss_function.boundary_aware_loss import bmc_loss
from utils.model_tool import save_net


POINT_GROUPS = {
    "scapula": [
        "ss", "sm", "si", "AA", "AP", "C", "gc", "SCP6", "SCP12",
        "bi3", "bi9", "bi12", "bi2", "bi4", "bi5", "bi6", "bi7", "bi8",
    ],
    "humerus": ["HC", "HI", "HS", "HP", "HA", "HBGS", "HBGI"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D shoulder landmark model.")
    parser.add_argument("-c", "--config", type=str, default="config/train.json", help="Path to a JSON config.")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "test"], default="train")
    return parser.parse_args()


def resolve_landmarks(dataset_opt):
    points_part = dataset_opt["points_part"]
    if points_part == "scapula":
        from utils.dataset.dataset_all_3D_Scapula import Shoulder_Joint_Whole as DatasetClass

        return POINT_GROUPS["scapula"], DatasetClass
    if points_part == "humerus":
        from utils.dataset.dataset_all_3D_Humerus import Shoulder_Joint_Whole as DatasetClass

        return POINT_GROUPS["humerus"], DatasetClass
    raise ValueError("datasets.train.points_part must be either 'scapula' or 'humerus'.")


def build_dataloaders(opt):
    dataset_opt = opt["datasets"]["train"]
    points, dataset_class = resolve_landmarks(dataset_opt)
    dataset_opt["num_landmarks"] = len(points)

    train_set = dataset_class(
        points,
        dataset_opt["image_size"],
        dataset_opt["trainroot"],
        dataset_opt["cacheroot"],
        dataset_opt["heat_sigma"],
        dataset_type=0,
        transform=False,
        rebuild_cache=dataset_opt["rebuild_cache"],
    )
    valid_set = dataset_class(
        points,
        dataset_opt["image_size"],
        dataset_opt["validroot"],
        dataset_opt["cacheroot"],
        dataset_opt["heat_sigma"],
        dataset_type=1,
        transform=False,
        rebuild_cache=dataset_opt["rebuild_cache"],
    )

    num_workers = int(dataset_opt.get("num_workers") or 0)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=dataset_opt["batch_size"],
        shuffle=dataset_opt["use_shuffle"],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=dataset_opt["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader


def build_loss(opt):
    return bmc_loss


def load_training_state(opt, logger, model, optimizer, device):
    resume_state = opt["path"]["resume_state"]
    if resume_state is None:
        logger.info("Starting training from scratch.")
        return 0, 0

    model_path = resume_state + "_gen_G.pth"
    optimizer_path = resume_state + "_opt.pth"
    if not os.path.exists(model_path) or not os.path.exists(optimizer_path):
        logger.error("Checkpoint files were not found; starting from scratch.")
        return 0, 0

    model.load_state_dict(torch.load(model_path, map_location=device))
    checkpoint = torch.load(optimizer_path, map_location=device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    logger.info("Resumed training from epoch %s, step %s.", checkpoint["epoch"], checkpoint["iter"])
    return checkpoint["epoch"], checkpoint["iter"]


def main():
    args = parse_args()
    opt = Logger.dict_to_nonedict(Logger.parse(args))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt["path"]["log"], "train", level=logging.INFO, screen=False)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))

    train_loader, valid_loader = build_dataloaders(opt)
    logger.info("Training and validation datasets are ready.")

    device = torch.device("cuda:0" if opt["gpu_ids"] and torch.cuda.is_available() else "cpu")
    model = GLCNet(number_classes=opt["datasets"]["train"]["num_landmarks"]).to(device)
    logger.info("Model initialized on %s.", device)

    criterion = build_loss(opt)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt["train"]["optimizer"]["lr"],
        weight_decay=opt["train"]["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-7,
    )

    num_landmarks = opt["datasets"]["train"]["num_landmarks"]
    landmark_weight_scheduler = AdaptiveLandmarkWeights(
        num_landmarks=num_landmarks,
        start_epoch=opt["train"]["pt_weight_scheduler"]["start_epoch"],
        update_interval=opt["train"]["pt_weight_scheduler"]["update_interval"],
        min_weight=opt["train"]["pt_weight_scheduler"]["clip_min"],
        max_weight=opt["train"]["pt_weight_scheduler"]["clip_max"],
    )

    current_epoch, current_step = load_training_state(opt, logger, model, optimizer, device)
    total_epochs = opt["train"]["n_epoch"]
    writer_root = os.path.join("runs", "experiments", f"{opt['name']}_{datetime.now().strftime('%y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=writer_root)

    remaining_steps = max(total_epochs - current_epoch, 0) * len(train_loader)
    progress = tqdm(total=remaining_steps, desc="Training", position=0, leave=True)

    try:
        while current_epoch < total_epochs:
            current_epoch += 1
            model.train()
            train_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                iter_start_time = time.time()
                images, labels, label_points, boundary_map, _ = batch
                images = images.to(device)
                labels = labels.to(device).squeeze(1).float()
                label_points = label_points.to(device)
                boundary_map = boundary_map.to(device).squeeze(1).float()

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images).squeeze(1).float()
                heatmap_loss, prior_loss, regularization_loss, peak_loss, coord_loss, total_loss = criterion(
                    outputs, labels, label_points, boundary_map, landmark_weight_scheduler, "train"
                )
                total_loss.backward()
                optimizer.step()

                current_step += 1
                train_loss += total_loss.item()
                writer.add_scalar("Loss/train", total_loss.item(), current_step)
                writer.add_scalar("Loss/train_heatmap", heatmap_loss.item(), current_step)
                writer.add_scalar("Loss/train_prior", prior_loss.item(), current_step)
                writer.add_scalar("Loss/train_regularization", regularization_loss.item(), current_step)
                writer.add_scalar("Loss/train_peak", peak_loss.item(), current_step)
                writer.add_scalar("Loss/train_coord", coord_loss.item(), current_step)

                progress.update(1)
                progress.set_postfix(
                    {
                        "epoch": f"{current_epoch}/{total_epochs}",
                        "step": f"{step}/{len(train_loader)}",
                        "loss": f"{total_loss.item():.8f}",
                    }
                )

                if current_step % opt["train"]["print_freq"] == 0:
                    logger.info(
                        "Epoch [%s/%s], step [%s/%s], loss %.8f, heatmap %.8f, prior %.8f, "
                        "regularization %.8f, peak %.8f, coord %.8f, time %.2fs",
                        current_epoch,
                        total_epochs,
                        step,
                        len(train_loader),
                        total_loss.item(),
                        heatmap_loss.item(),
                        prior_loss.item(),
                        regularization_loss.item(),
                        peak_loss.item(),
                        coord_loss.item(),
                        time.time() - iter_start_time,
                    )

            scheduler.step(current_epoch)

            if current_epoch % opt["train"]["val_freq"] == 0:
                validate(model, valid_loader, criterion, landmark_weight_scheduler, device, writer, logger, current_epoch)

            logger.info(
                "Epoch [%s/%s] finished. Mean training loss: %.8f",
                current_epoch,
                total_epochs,
                train_loss / max(len(train_loader), 1),
            )

            if current_epoch % opt["train"]["save_checkpoint_epoch"] == 0:
                save_net(opt, logger, model, optimizer, current_epoch, current_step)

            landmark_weight_scheduler.update_if_needed(current_epoch, logger)
    finally:
        progress.close()
        writer.close()

    logger.info("Training complete.")


def validate(model, valid_loader, criterion, landmark_weight_scheduler, device, writer, logger, current_epoch: int):
    model.eval()
    totals = {
        "loss": 0.0,
        "heatmap": 0.0,
        "prior": 0.0,
        "regularization": 0.0,
        "peak": 0.0,
        "coord": 0.0,
    }

    with torch.no_grad():
        for batch in valid_loader:
            images, labels, label_points, boundary_map, _ = batch
            images = images.to(device)
            labels = labels.to(device).squeeze(1).float()
            label_points = label_points.to(device)
            boundary_map = boundary_map.to(device).squeeze(1).float()
            outputs = model(images).squeeze(1).float()

            heatmap_loss, prior_loss, regularization_loss, peak_loss, coord_loss, total_loss = criterion(
                outputs, labels, label_points, boundary_map, landmark_weight_scheduler, "valid"
            )
            totals["loss"] += total_loss.item()
            totals["heatmap"] += heatmap_loss.item()
            totals["prior"] += prior_loss.item()
            totals["regularization"] += regularization_loss.item()
            totals["peak"] += peak_loss.item()
            totals["coord"] += coord_loss.item()

    count = max(len(valid_loader), 1)
    averages = {key: value / count for key, value in totals.items()}
    writer.add_scalar("Loss/valid", averages["loss"], current_epoch)
    logger.info(
        "Validation epoch %s: loss %.8f, heatmap %.8f, prior %.8f, regularization %.8f, peak %.8f, coord %.8f",
        current_epoch,
        averages["loss"],
        averages["heatmap"],
        averages["prior"],
        averages["regularization"],
        averages["peak"],
        averages["coord"],
    )


if __name__ == "__main__":
    main()

