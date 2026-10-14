# SPDX-License-Identifier: Apache-2.0
"""Metric names and reductions shared by every SimplifiedDistill execution path."""

import torch

LOSS_NAMES = ("imitation", "teacher_repulsion")
SECTIONS = ("loss/", "loss_raw/", "diagnostic/", "segments/", "chunks/", "weights/", "enabled/", "progress/")


def metric_name(name):
    # total_loss is the SMD training API key, not another published alias.
    if name == "total_loss":
        return "loss/total"
    if not name.startswith(SECTIONS):
        raise ValueError(f"Undeclared SimplifiedDistill metric namespace: {name}")
    return name


def uses_last_value(name):
    return name.startswith(("progress/", "weights/", "enabled/"))


def reduce_metrics(records, *, segments=False):
    """Sum normalized segment contributions; mean microbatch objectives and observations."""
    if not records:
        raise ValueError("Cannot reduce an empty metric sequence.")
    result = {}
    for name in dict.fromkeys(key for record in records for key in record):
        metric_name(name)
        values = [record[name].detach() for record in records if name in record]
        if uses_last_value(name):
            result[name] = values[-1]
        elif segments and name.startswith("loss/"):
            result[name] = torch.stack(values).sum()
        else:
            result[name] = values[0] if len(values) == 1 else torch.stack(values).mean()
    return result


def objective_metrics(raw, weights, *, segment_index):
    """Raw distances and signed contributions; disabled raw terms were not measured."""
    result = {}
    for name, sign in zip(LOSS_NAMES, (1.0, -1.0)):
        value, weight = raw[name].detach(), getattr(weights, name)
        result[f"loss/{name}"] = sign * value * weight * weights.normalization
        result[f"weights/{name}"] = value.new_tensor(weight)
        result[f"enabled/{name}"] = value.new_tensor(float(weight > 0))
        if weight > 0:
            result[f"loss_raw/{name}"] = value
            result[f"segments/{segment_index}/loss_raw/{name}"] = value
    return result
