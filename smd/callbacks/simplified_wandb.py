# SPDX-License-Identifier: Apache-2.0
"""SimplifiedDistill scalar logging, with explicit namespaces and observation counts."""

from dataclasses import dataclass, field
import math
import torch

from smd.callbacks.wandb import WandbCallback
from smd.callbacks.wandb_utils import log_at_iteration
from smd.methods.simplified_distill.metrics import metric_name, uses_last_value
from smd.utils.distributed import is_rank0, world_size
import smd.utils.logging_utils as logger


@dataclass
class MetricRecord:
    sums: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)

    def add(self, values):
        for name, value in (values or {}).items():
            name = metric_name(name)
            value = float(value.detach()) if torch.is_tensor(value) else float(value)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite logged metric {name}={value}")
            if uses_last_value(name):
                self.last[name] = value
            else:
                self.sums[name] = self.sums.get(name, 0.0) + value
                self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self):
        self.sums.clear()
        self.counts.clear()
        self.last.clear()

    @staticmethod
    def merge(records):
        result = {}
        keys = dict.fromkeys(key for record in records for key in record[0])
        for key in keys:
            count = sum(counts.get(key, 0) for _, counts, _ in records)
            result[key] = sum(sums.get(key, 0.0) for sums, _, _ in records) / count
        last_keys = dict.fromkeys(key for _, _, last in records for key in last)
        for key in last_keys:
            values = [last[key] for _, _, last in records if key in last]
            if key == "progress/conditions_consumed":
                result[key] = sum(values)
            else:
                if len(set(values)) != 1:
                    raise ValueError(f"Ranks disagree on {key}: {values}")
                result[key] = values[0]
        return result

    def get_stat(self):
        record = (self.sums, self.counts, self.last)
        records = [record]
        if world_size() > 1:
            records = [None] * world_size()
            torch.distributed.all_gather_object(records, record)
        result = self.merge(records)
        self.reset()
        return result


class SimplifiedWandbCallback(WandbCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_dict_record = MetricRecord()
        self.val_loss_dict_record = MetricRecord()

    def on_optimizer_step_begin(self, model, iteration=0):
        if is_rank0() and iteration % self.config.trainer.logging_iter == 0:
            log_at_iteration(
                {
                    f"optimizer/lr/{name}": scheduler.get_last_lr()[0]
                    for name, scheduler in model.scheduler_dict.items()
                },
                iteration=iteration,
            )

    def on_training_step_end(self, model, data_batch, output_batch, loss_dict, iteration=0):
        loss_dict = dict(loss_dict)
        cycle = model.training_batch_reuse
        loss_dict["progress/conditions_consumed"] = cycle.samples_consumed
        next_segment = getattr(model, "next_segment_index", lambda _: cycle.segment_index)
        loss_dict["progress/next_segment_index"] = next_segment(iteration)
        super().on_training_step_end(model, data_batch, output_batch, loss_dict, iteration=iteration)

    def log_stats(self, loss_dict_record, iteration=0, group="train"):
        stats = loss_dict_record.get_stat()  # Every rank participates in the reduction.
        if is_rank0():
            payload = {f"{group}/{name}": value for name, value in stats.items()}
            logger.info(f"{group} metrics at optimizer iteration {iteration}: {payload}")
            log_at_iteration(payload, iteration=iteration)

    def on_load_checkpoint_end(self, model, iteration=0):
        # A resumed log window starts at the restored iteration, with no stale observations.
        self.loss_dict_record.reset()
        self.val_loss_dict_record.reset()
