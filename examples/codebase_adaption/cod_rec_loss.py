"""REC-OneSide-NoIS policy loss for the CoD reproduction (arXiv 2606.20002, user 2026-10-18).

Selected only via ``--loss-type custom_loss --custom-loss-function-path
examples.codebase_adaption.cod_rec_loss.rec_onesided_noIS_policy_loss``; nothing else imports it.

Per-token gradient (paper App. A, Eq. 2): grad log pi(i) * A_i * M_i, no importance-sampling
weight, with the one-side mask
    M_i = 1(A_i > 0, ratio_i <= 1 + eps_high) + 1(A_i < 0, ratio_i >= 1 - eps_low)
where ratio_i = pi(i) / pi_old(i) and eps_low/eps_high come from --eps-clip/--eps-clip-high.
The adaptive RED re-weighting (Eq. 3) is applied to the advantages upstream in
codebase_advantage (it needs the whole rollout batch), so this module is the plain REC term.

Everything else (log-prob computation, KL term, entropy, TIS/mismatch metrics, reducers) is the
stock ``policy_loss_function``: only its per-token PPO surrogate is swapped for the call.
"""

from __future__ import annotations

import threading

import torch

from miles.backends.training_utils.loss_hub import losses as _losses

_SWAP_LOCK = threading.Lock()


def rec_onesided_noIS_surrogate(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    """Drop-in for math_utils.compute_policy_loss: returns (per-token loss, masked-fraction).

    ppo_kl = log pi_old - log pi, so -ppo_kl carries grad log pi; the mask is detached.
    """
    log_ratio = -ppo_kl
    ratio = log_ratio.detach().exp()
    keep = ((advantages > 0) & (ratio <= 1.0 + eps_clip_high)) | ((advantages < 0) & (ratio >= 1.0 - eps_clip))
    mask = keep.to(log_ratio.dtype)
    pg_losses = -log_ratio * advantages * mask
    clipfrac = ((advantages != 0) & ~keep).to(log_ratio.dtype)
    return pg_losses, clipfrac


def rec_onesided_noIS_policy_loss(args, batch, logits, sum_of_sample_mean):
    """``LossFunction`` entry point: stock policy loss with the REC-OneSide-NoIS surrogate."""
    with _SWAP_LOCK:
        orig = _losses.compute_policy_loss
        _losses.compute_policy_loss = rec_onesided_noIS_surrogate
        try:
            return _losses.policy_loss_function(args, batch, logits, sum_of_sample_mean)
        finally:
            _losses.compute_policy_loss = orig
