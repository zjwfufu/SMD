"""Frozen paired-teacher correction on the student trajectory state."""
import torch
import torch.nn.functional as F

def paired_teacher_state_correction_loss(
    proxy_state: torch.Tensor,
    reference_velocity: torch.Tensor,
    strong_velocity: torch.Tensor,
    *,
    normalize: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach a frozen weak-to-strong velocity correction only to ``proxy_state``."""
    if proxy_state.shape != reference_velocity.shape or proxy_state.shape != strong_velocity.shape:
        raise ValueError(
            "Paired-teacher state correction requires equal proxy/velocity shapes, got "
            f"{tuple(proxy_state.shape)}, {tuple(reference_velocity.shape)}, {tuple(strong_velocity.shape)}."
        )
    direction = (reference_velocity.detach().float() - strong_velocity.detach().float()).detach()
    dims = tuple(range(1, direction.ndim))
    raw_rms = (direction.square().mean(dim=dims, keepdim=True) + eps).sqrt()
    if normalize:
        direction = direction / raw_rms
    direction_rms = direction.square().mean(dim=dims).sqrt()
    proxy_fp32 = proxy_state.float()
    pseudo_target = (proxy_fp32.detach() + direction).detach()
    loss = 0.5 * F.mse_loss(proxy_fp32, pseudo_target, reduction="mean")
    return loss, raw_rms.reshape(proxy_state.shape[0]).mean().detach(), direction_rms.mean().detach()

@torch.no_grad()
def strong_cfg_residual_targets(teacher, state, sigma, condition, negative, *, precision, skip_layers):
    if negative is None:
        raise ValueError("Strong CFG residual requires a negative condition.")
    def forward(cond, **kwargs):
        value = teacher(state.to(dtype=precision), sigma, condition=cond, fwd_pred_type="flow", **kwargs)
        return (value[0] if isinstance(value, (tuple, list)) else value).detach()
    strong = forward(condition)
    reference = forward(negative, **({"skip_layers": skip_layers} if skip_layers is not None else {}))
    return strong, reference
