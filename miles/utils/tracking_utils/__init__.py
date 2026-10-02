import logging

from .base import TrackingManager

logger = logging.getLogger(__name__)
_manager = TrackingManager()


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    # Set only by decoupled multi-policy training (miles/ray/decoupled_train.py) so a second
    # policy's metrics do not overwrite the first's; step keys stay unprefixed.
    if prefix := getattr(args, "tracking_key_prefix", None):
        metrics = {k if k == step_key else f"{prefix}{k}": v for k, v in metrics.items()}
    _manager.log(metrics, step=step)


def finish_tracking():
    _manager.finish()
