# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import tempfile
import time
import uuid
from typing import Optional, Dict, Callable, TYPE_CHECKING
import gc


import torch
import torchvision
from torchvision.transforms import functional as tv_F

import wandb

from smd.callbacks.callback import Callback
from smd.callbacks.sampling.base import sanitize_stem
from smd.callbacks.sampling.rendering.decoder import describe_decoder
from smd.callbacks.sampling.rendering.gather import gather_samples
from smd.callbacks.sampling.rendering.protocol import VisualizationItem
from smd.callbacks.wandb_utils import log_at_iteration
from smd.configs.config_utils import serialize_config
from smd.utils import basic_utils

from smd.utils.distributed import is_rank0, rank0_only, synchronize, world_size
from smd.utils import logging_utils as logger

if TYPE_CHECKING:
    from smd.configs.config import BaseConfig
    from smd.methods import SMDModel


_WANDB_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _generate_wandb_run_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(frozen=True)
class WandbRunIdentity:
    run_id: str
    source: str


def _validate_wandb_run_id(value: str, *, source: str) -> str:
    run_id = value.strip()
    if not _WANDB_RUN_ID_PATTERN.fullmatch(run_id):
        raise RuntimeError(
            f"Invalid W&B run id from {source}: {value!r}. "
            "Expected 1-128 ASCII letters, digits, underscores, or hyphens."
        )
    return run_id


def _atomic_write_wandb_run_id(path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{run_id}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _create_wandb_run_id_if_absent(path: Path, run_id: str) -> bool:
    """Create the identity sidecar without replacing a concurrently-created file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{run_id}\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return True


def resolve_wandb_run_identity(
    save_path: str | os.PathLike[str],
    *,
    resume: bool = True,
    generate_id: Callable[[], str] | None = None,
) -> WandbRunIdentity:
    """Resolve one durable W&B run ID for a logical SMD output directory."""
    generate_id = _generate_wandb_run_id if generate_id is None else generate_id
    root = Path(save_path)
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "wandb_id.txt"
    environment_id_raw = os.environ.get("WANDB_RUN_ID")
    environment_id = (
        _validate_wandb_run_id(environment_id_raw, source="WANDB_RUN_ID") if environment_id_raw is not None else None
    )

    try:
        sidecar_text = identity_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        sidecar_text = None
    except OSError as exc:
        raise RuntimeError(
            f"Could not read existing W&B identity sidecar {identity_path}; "
            "refusing to replace a possibly valid legacy run ID."
        ) from exc

    if sidecar_text is not None:
        try:
            sidecar_id = _validate_wandb_run_id(sidecar_text, source=str(identity_path))
        except RuntimeError as exc:
            if environment_id is None and resume:
                raise RuntimeError(
                    f"W&B identity sidecar {identity_path} is invalid and WANDB_RUN_ID is unset."
                ) from exc
            fresh_id = environment_id or _validate_wandb_run_id(
                generate_id(),
                source="wandb.util.generate_id()",
            )
            source = "environment_repair" if environment_id is not None else "generated_fresh"
            logger.warning(f"Replacing invalid W&B identity sidecar {identity_path} from {source}.")
            _atomic_write_wandb_run_id(identity_path, fresh_id)
            return WandbRunIdentity(fresh_id, source)

        if not resume:
            fresh_id = environment_id or _validate_wandb_run_id(
                generate_id(),
                source="wandb.util.generate_id()",
            )
            source = "environment_fresh" if environment_id is not None else "generated_fresh"
            logger.info(
                f"Starting a fresh W&B run id={fresh_id!r}; replacing persisted id "
                f"{sidecar_id!r} because trainer.resume=False."
            )
            _atomic_write_wandb_run_id(identity_path, fresh_id)
            return WandbRunIdentity(fresh_id, source)
        if environment_id is not None and environment_id != sidecar_id:
            logger.warning(
                f"WANDB_RUN_ID={environment_id!r} differs from persisted id {sidecar_id!r}; "
                "the persisted output-directory identity wins."
            )
        return WandbRunIdentity(sidecar_id, "sidecar")

    if environment_id is not None:
        run_id = environment_id
        source = "environment"
    else:
        run_id = _validate_wandb_run_id(generate_id(), source="wandb.util.generate_id()")
        source = "generated"

    if _create_wandb_run_id_if_absent(identity_path, run_id):
        return WandbRunIdentity(run_id, source)

    # Another process or a temporarily stale directory lookup won the create
    # race. Its persisted identity is authoritative.
    try:
        concurrent_id = _validate_wandb_run_id(
            identity_path.read_text(encoding="utf-8"),
            source=str(identity_path),
        )
    except OSError as exc:
        raise RuntimeError(f"W&B identity sidecar {identity_path} appeared concurrently but cannot be read.") from exc
    if environment_id is not None and environment_id != concurrent_id:
        logger.warning(
            f"Concurrent W&B identity {concurrent_id!r} differs from WANDB_RUN_ID={environment_id!r}; "
            "the persisted output-directory identity wins."
        )
    return WandbRunIdentity(concurrent_id, "sidecar_race")


def to_wandb(
    tensor: torch.Tensor,
    rgb_range: float = 255.0,
    normalized: bool = False,
    max_plot_img: int = 16,
    max_plot_vid: int = 2,
    fps: int = 16,
    channel_before_time: bool = True,
    caption: str | None = None,
    vid_format: str = "mp4",
    image_file_type: str = "jpg",
) -> wandb.Image | wandb.Video:
    """
    Convert a tensor to a wandb.Image or wandb.Video.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B,C,H,W], [B,T,C,H,W], or [B,T,C,H,W,D].
        rgb_range (float, optional): Output target RGB range (can almost definitely be kept as 255).
            Defaults to 255.0.
        normalized (bool, optional): Whether the tensor is normalized to [0,1]. Defaults to False which assumes [-1,1] range.
        max_plot_img (int, optional): Max number of images to plot. Defaults to 16.
        max_plot_vid (int, optional): Max number of videos to plot. Defaults to 2.
        fps (int, optional): Frames per second. Defaults to 8.
        channel_before_time (bool, optional): Whether the tensor is in the format [B,C,T,..]. Set False if the [B,T,C,..] format is used.
        caption (str, optional): Caption for the image or video. Defaults to None.
        vid_format (str, optional): Format of the video file. Defaults to "mp4".
        image_file_type (str, optional): File type for W&B static images. Defaults to "jpg" to keep uploads small.

    Returns:
        wandb.Image | wandb.Video: Format a tensor for logging to W&B.
    """

    if tensor.ndim == 5:
        max_plot = max_plot_vid
        if channel_before_time:
            tensor = tensor.permute(0, 2, 1, 3, 4)
    elif tensor.ndim == 4:
        max_plot = max_plot_img
    else:
        raise ValueError(f"Tensor must be 4 or 5 dimensional, but got {tensor.ndim} dimensions")

    # slice and adjust range
    if normalized:
        factor = rgb_range
        offset = 0.0
    else:
        factor = rgb_range / 2.0
        offset = rgb_range / 2.0
    tensor = tensor[:max_plot].mul(factor).add(offset).clip_(0, rgb_range).to(torch.uint8)

    # convert to wandb.Image or wandb.Video
    assert tensor.shape[-3] == 3, "Make sure that the data is in ..., C, H, W format"
    if tensor.ndim == 5:
        return wandb.Video(tensor.cpu().numpy(), fps=fps, format=vid_format, caption=caption)
    else:
        image_grid = torchvision.utils.make_grid(tensor, nrow=4, pad_value=1)
        image_grid = tv_F.to_pil_image(image_grid).convert("RGB")
        return wandb.Image(image_grid, caption=caption, file_type=image_file_type)


@rank0_only
def init_wandb(config: BaseConfig):
    # wandb login
    wandb_credential = config.log_config.wandb_credential
    if os.path.isfile(wandb_credential):
        os.environ["WANDB_API_KEY"] = open(wandb_credential, encoding="utf-8").read().strip("\n")
        logger.info(f"Loading WANDB_API_KEY from {wandb_credential}")

    wandb_config = config.log_config

    resume_training = bool(getattr(getattr(config, "trainer", None), "resume", True))
    logger.info(f"wandb_config.save_path: {wandb_config.save_path}")
    identity = resolve_wandb_run_identity(wandb_config.save_path, resume=resume_training)
    logger.info(
        f"Using W&B {wandb_config.name} id={identity.run_id} "
        f"(identity_source={identity.source}, trainer_resume={resume_training})"
    )

    # Get config as plain dict
    config_resolved = serialize_config(config, return_type="dict")
    config_resolved["runtime"] = {
        "retry_attempt": os.environ.get("TORCHELASTIC_RESTART_COUNT"),
        "max_retries": os.environ.get("SMD_TRAIN_MAX_RETRIES"),
        "wandb_name": wandb_config.name,
        "wandb_run_id": identity.run_id,
        "wandb_run_id_source": identity.source,
    }

    # Keep the durable run identity under save_path, while allowing W&B media
    # and its frequently updated event files to use node-local storage.
    wandb_dir = os.environ.get("WANDB_DIR") or wandb_config.save_path
    os.makedirs(wandb_dir, exist_ok=True)

    # Initialize the wandb library.
    wandb.init(
        id=identity.run_id,
        project=wandb_config.project,
        group=wandb_config.group,
        name=wandb_config.name,
        config=config_resolved,
        dir=wandb_dir,
        resume="allow",
        mode=wandb_config.wandb_mode,
    )
    wandb.define_metric("optimizer/iteration")
    wandb.define_metric("*", step_metric="optimizer/iteration")

    # Save a copy of code to a wandb Artifact (this can be slow)
    # Make code upload optional to avoid distributed training delays
    upload_code = basic_utils.str2bool(os.getenv("WANDB_UPLOAD_CODE", "false"))
    if upload_code:
        logger.info("Uploading code to wandb (this may take a few minutes)...")
        wandb.run.log_code(".")
        logger.info("Code upload to wandb completed")
    else:
        logger.info("Wandb code upload disabled (set WANDB_UPLOAD_CODE=true to enable)")


@dataclass
class _LossDictRecord:
    loss_dict: dict = field(default_factory=dict)
    iter_count_dict: dict = field(default_factory=dict)

    def add(self, loss_dict: Optional[Dict[str, torch.Tensor]]) -> None:
        if loss_dict is not None:
            for loss_name, loss_val in loss_dict.items():
                if isinstance(loss_val, (int, float)):
                    val = float(loss_val)
                elif isinstance(loss_val, torch.Tensor):
                    val = loss_val.float().item()
                else:
                    continue
                self.loss_dict[loss_name] = self.loss_dict.get(loss_name, 0.0) + val
                self.iter_count_dict[loss_name] = self.iter_count_dict.get(loss_name, 0) + 1

    def reset(self) -> None:
        self.loss_dict = {}
        self.iter_count_dict = {}

    def gather_dict(self, dictionary: Dict[str, float | int]) -> Dict[str, float | int]:
        n_ranks = world_size()
        if n_ranks > 1:
            dict_list = [None for _ in range(n_ranks)]
            torch.distributed.all_gather_object(dict_list, dictionary)
            # from list of dicts to dict of summed values
            dictionary = {}
            for d in dict_list:
                for key, value in d.items():
                    dictionary[key] = dictionary.get(key, 0.0) + value
        return dictionary

    def get_stat(self) -> Dict[str, float]:
        # number of ranks that logged this loss
        rank_dict = self.gather_dict({k: 1 for k in self.loss_dict.keys()})
        # number of times this loss was computed
        count_dict = self.gather_dict(self.iter_count_dict)
        # sum of all losses
        loss_dict = self.gather_dict(self.loss_dict)

        avg_loss_dict = {}
        rows = []
        for loss_name, loss_val in loss_dict.items():
            count = count_dict.get(loss_name, 0)
            ranks = rank_dict.get(loss_name, 1)
            iter_count = count / ranks
            avg_loss = (loss_val / count) * (ranks / world_size()) if count > 0 else 0.0
            stat_name = f"avg_{loss_name}"
            rows.append((stat_name, avg_loss, iter_count))
            avg_loss_dict[loss_name] = avg_loss

        name_width = max((len(name) for name, _, _ in rows), default=16)
        for stat_name, avg_loss, iter_count in rows:
            abs_val = abs(avg_loss)
            value_str = f"{avg_loss:.3e}" if (0.0 < abs_val < 1e-4 or abs_val >= 1e4) else f"{avg_loss:.4f}"
            logger.info(f"{stat_name:<{name_width}} : {value_str:>12}  iter count: {iter_count:.1f}")
        self.reset()
        return avg_loss_dict


class WandbCallback(Callback):
    """
    The callback gets precision for data from model
    """

    def __init__(
        self,
        validation_logging_step: int = 1,
        sample_logging_iter: Optional[int] = None,
        sample_grid_size: int = 4,
        fps: int = 16,
    ):
        self.validation_logging_step = validation_logging_step
        self.sample_logging_iter = sample_logging_iter
        self.val_sample_map = None
        self.sample_grid_size = sample_grid_size
        self.fps = fps
        self.loss_dict_record = _LossDictRecord()
        self.val_loss_dict_record = _LossDictRecord()
        self._unconditional_metric_map = {
            "attraction_loss": "unconditional_attraction_loss",
            "total_loss": "unconditional_total_loss",
        }

    def on_app_begin(self) -> None:
        assert hasattr(self, "config"), "Missing config in WandbCallback."
        init_wandb(self.config)
        if self.sample_logging_iter is None:
            self.sample_logging_iter = self.config.trainer.logging_iter
        synchronize()

    @rank0_only
    def on_optimizer_step_begin(self, model: SMDModel, iteration: int = 0) -> None:
        assert hasattr(self, "config"), "Missing config in WandbCallback."
        if iteration % self.config.trainer.logging_iter == 0:
            for name, scheduler in model.scheduler_dict.items():
                log_at_iteration(
                    {f"optimizer/lr_{name}": scheduler.get_last_lr()[0]},
                    iteration=iteration,
                )

    def collect_sample_latents(
        self, model: SMDModel, data_batch: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor | Callable]
    ) -> dict[str, torch.Tensor]:
        """Gather the latents worth visualizing for this step.

        Returns *latents*, never pixels: decoding happens in the isolated worker.
        Every rank must call this — it runs collectives.
        """
        if "gen_rand" not in output_batch:
            return {}  # Sparse/non-visual models may not produce gen_rand

        gen_rand = output_batch["gen_rand"]
        if isinstance(gen_rand, dict):
            synchronize()
            gen_rand = {k: (v() if callable(v) else v) for k, v in gen_rand.items()}
            synchronize()
        elif isinstance(gen_rand, Callable):
            synchronize()
            gen_rand = gen_rand()
            synchronize()

        latents: dict[str, torch.Tensor] = {}
        if isinstance(gen_rand, dict):
            for name, value in gen_rand.items():
                latents[f"student/generation/{name}"] = value
        else:
            latents["student/generation"] = gen_rand

        if "real" in data_batch:
            latents["data/real"] = data_batch["real"]

        # Multistep teacher generation, for reference against the student.
        if self.config.trainer.visualize_teacher:
            assert "input_rand" in output_batch, "We need to know the noise to visualize teacher generation"
            latents["teacher/generation"] = model.sample(
                model.teacher,
                output_batch["input_rand"][0:1],
                data_batch["condition"][0:1],
                data_batch["neg_condition"][0:1],
            )

        if logger.LOG_LEVEL == "DEBUG" and "gen_rand_train" in output_batch:
            latents["student/generation_train"] = output_batch["gen_rand_train"]

        return {name: value for name, value in latents.items() if isinstance(value, torch.Tensor)}

    def log_sample_map(
        self,
        model: SMDModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        suffix: str = "",
        iteration: int = 0,
        group: str = "train",
    ) -> None:
        """Queue this step's media for the job-wide render event.

        Historically this decoded through the VAE inline. Now only latents are
        collected; CallbackDict flushes all producers through one isolated,
        timeout-bounded worker after the hook completes.
        """
        latents = self.collect_sample_latents(model, data_batch, output_batch)

        items = []
        for name in sorted(latents):
            # Collective: identical key order on every rank.
            gathered = gather_samples(latents[name], limit=self.sample_grid_size)
            if gathered is None:
                continue
            key = f"{group}_media/{name}{suffix}"
            items.append(
                VisualizationItem(
                    key=key,
                    stem=sanitize_stem(key),
                    caption=self._sample_caption(data_batch),
                    latents=gathered,
                    fps=self.fps,
                )
            )

        net = getattr(model, "net_inference", None) or model.net
        self.trainer.preview_service.enqueue(
            label="train_media",
            items=items,
            iteration=iteration,
            decoder=describe_decoder(net) if is_rank0() else None,
        )
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _sample_caption(data_batch: dict[str, torch.Tensor]) -> str | None:
        raw = data_batch.get("condition_raw")
        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], str):
            return "\n".join(str(text) for text in raw[:4])
        return None

    def log_stats(self, loss_dict_record: _LossDictRecord, iteration: int = 0, group: str = "train") -> None:
        logger.info(f"logging {group} stats at iteration {iteration}" + "-" * 20)
        # Collect distributed statistics
        avg_loss_dict = loss_dict_record.get_stat()
        stats = {self._metric_namespace(name): val for name, val in avg_loss_dict.items()}
        # W&B's internal _step remains monotonic across checkpoint rollback;
        # optimizer/iteration is the real training x-axis and may replay.
        log_at_iteration(stats, iteration=iteration)

    @staticmethod
    def _is_segment_metric(name: str) -> bool:
        parts = name.split("_", 2)
        return len(parts) == 3 and parts[0] == "seg" and parts[1].isdigit()

    def _loss_dict_for_logging(
        self,
        loss_dict: Optional[Dict[str, torch.Tensor]],
        output_batch: dict[str, torch.Tensor | Callable | bool],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if loss_dict is None:
            return None
        is_unconditional_step = output_batch.get("is_unconditional_step") if isinstance(output_batch, dict) else None
        if is_unconditional_step is not True:
            return loss_dict
        return {
            logged_name: loss_dict[source_name]
            for source_name, logged_name in self._unconditional_metric_map.items()
            if source_name in loss_dict
        }

    @classmethod
    def _metric_namespace(cls, name: str) -> str:
        """Route train metrics into stable W&B sections without duplicating keys."""
        if name.startswith(("diagnostic/", "loss/", "loss_raw/")):
            return name
        if name.startswith("causal_wan/"):
            return f"diagnostic/{name}"
        if cls._is_segment_metric(name):
            _, seg_idx, metric = name.split("_", 2)
            section = cls._metric_section(metric, segment=True)
            return f"{section}/segments/{metric}/seg{seg_idx}"
        section = cls._metric_section(name, segment=False)
        return f"{section}/{name}"

    @staticmethod
    def _metric_section(metric: str, *, segment: bool) -> str:
        diagnostic_tokens = (
            "norm",
            "smoothness",
            "spread",
            "estimator",
            "mode",
            "beta",
            "weight",
            "reference",
            "distance",
            "time",
            "skip_",
            "selected_segment",
            "selected_segments",
            "prefix_rollout",
            "total_segments",
            "phase",
            "nfe",
            "ratio",
        )
        if any(token in metric for token in diagnostic_tokens):
            return "diagnostic"
        if "raw" in metric or metric.endswith("_term") or segment:
            return "loss_raw"
        return "loss"

    def on_training_step_end(
        self,
        model: SMDModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        self.loss_dict_record.add(self._loss_dict_for_logging(loss_dict, output_batch))
        time_start = time.perf_counter()
        logged = False
        if iteration % self.config.trainer.logging_iter == 0 or iteration == 1:
            self.log_stats(self.loss_dict_record, iteration=iteration, group="train")
            logged = True
        if iteration % self.sample_logging_iter == 0 or iteration == 1:
            self.log_sample_map(model, data_batch, output_batch, iteration=iteration, group="train")
            logged = True
        if logged:
            time_taken = time.perf_counter() - time_start
            logger.info(f"WandB logging complete after {time_taken:.2f} seconds")

    def on_validation_step_end(
        self,
        model: SMDModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        self.val_loss_dict_record.add(self._loss_dict_for_logging(loss_dict, output_batch))

        if step % self.validation_logging_step == 0:
            self.log_sample_map(
                model, data_batch, output_batch, suffix=f"_{step}", iteration=iteration, group=f"val{idx}"
            )

    def on_validation_end(self, model: SMDModel, iteration: int = 0, idx: int = 0) -> None:
        self.log_stats(self.val_loss_dict_record, iteration=iteration, group=f"val{idx}")
