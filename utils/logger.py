"""Configuration and logging helpers."""

import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Union


class NoneDict(dict):
    """Dictionary that returns ``None`` for missing keys."""

    def __missing__(self, key: str):
        return None


def mkdirs(paths):
    if isinstance(paths, (str, os.PathLike)):
        Path(paths).mkdir(parents=True, exist_ok=True)
        return

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def _load_json_with_line_comments(config_path):
    json_lines = []
    with open(config_path, "r", encoding="utf-8") as handle:
        for line in handle:
            json_lines.append(line.split("//", 1)[0])
    return json.loads("".join(json_lines), object_pairs_hook=OrderedDict)


def parse(args):
    """Load a JSON config and prepare experiment output directories."""

    opt = _load_json_with_line_comments(args.config)
    opt["phase"] = args.phase

    experiment_root = Path("experiments") / f"{opt['name']}_{get_timestamp()}"
    opt["path"]["experiments_root"] = str(experiment_root)

    for key, path in list(opt["path"].items()):
        if path is None or "resume" in key or "experiments" in key:
            continue
        resolved = experiment_root / str(path)
        mkdirs(resolved)
        opt["path"][key] = str(resolved)

    gpu_ids = opt.get("gpu_ids")
    if gpu_ids:
        opt["gpu_ids"] = [int(gpu_id.strip()) for gpu_id in str(gpu_ids).split(",") if gpu_id.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in opt["gpu_ids"])
    else:
        opt["gpu_ids"] = []
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    opt["distributed"] = len(opt["gpu_ids"]) > 1
    return opt


def setup_logger(
    logger_name,
    root,
    phase,
    level=logging.INFO,
    screen=False,
):
    """Configure a file logger for a training or inference phase."""

    mkdirs(root)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(Path(root) / f"{phase}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if screen:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        return NoneDict({key: dict_to_nonedict(value) for key, value in opt.items()})
    if isinstance(opt, list):
        return [dict_to_nonedict(value) for value in opt]
    return opt


def dict2str(opt, indent_l=1):
    lines = []
    for key, value in opt.items():
        indent = " " * (indent_l * 2)
        if isinstance(value, dict):
            lines.append(f"{indent}{key}:[")
            lines.append(dict2str(value, indent_l + 1).rstrip())
            lines.append(f"{indent}]")
        else:
            lines.append(f"{indent}{key}: {value}")
    return "\n".join(lines) + "\n"
