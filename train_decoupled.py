"""Entry point for decoupled multi-policy training (see miles/ray/decoupled_train.py).

Same arguments as train.py plus ``--decoupled-config <yaml>`` (consumed here, before the
regular parser). The loop mirrors train.py's order — generate, offload rollout, train, save,
offload train, onload weights, update weights, onload kv, eval — applied to every train group.
Groups train concurrently (disjoint GPUs); a group whose share of a rollout is too small to
fill its DP ranks skips that train step but still saves / syncs with the others.
"""

import asyncio
import logging
import os
import sys

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.decoupled_train import (
    DecoupledRolloutManager,
    check_layout,
    create_decoupled_training_models,
    load_decoupled_config,
)
from miles.ray.placement_group import create_placement_groups
from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import finish_tracking, init_tracking

logger = logging.getLogger(__name__)


def _pop_decoupled_config_arg() -> str:
    argv = sys.argv
    assert "--decoupled-config" in argv, "train_decoupled.py requires --decoupled-config <yaml>"
    i = argv.index("--decoupled-config")
    path = argv[i + 1]
    del argv[i : i + 2]
    return path


def _define_prefixed_metrics(args, specs) -> None:
    """Give prefixed groups' wandb keys the same step axes as the unprefixed ones."""
    if not getattr(args, "use_wandb", False):
        return
    import wandb

    for spec in specs[1:]:
        p = spec.tracking_prefix if spec.tracking_prefix is not None else f"{spec.name}/"
        wandb.define_metric(f"{p}train/*", step_metric="train/step")
        wandb.define_metric(f"{p}rollout/*", step_metric="rollout/step")
        wandb.define_metric(f"{p}perf/*", step_metric="rollout/step")


async def train(args, route_key, specs):
    configure_logger()
    assert args.num_rollout and args.num_rollout > 0, "decoupled training needs --num-rollout"
    assert not args.use_critic, "decoupled training does not support a critic"
    check_layout(args, specs)

    pgs = create_placement_groups(args)
    init_tracking(args)
    _define_prefixed_metrics(args, specs)

    rollout_manager = DecoupledRolloutManager.options(
        num_cpus=1, num_gpus=0, **(compute_ray_pin_head_options() if args.pin_rollout_manager_to_head else {})
    ).remote(args, pgs["rollout"])
    await rollout_manager.configure_groups.remote(route_key, specs)
    if args.offload_rollout:
        await rollout_manager.offload.remote()

    groups = await create_decoupled_training_models(args, pgs, rollout_manager, specs)

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()
    for g in groups.values():
        await g.update_weights()
    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        refs = await rollout_manager.generate_decoupled.remote(rollout_id)

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

        trained = [name for name, r in refs.items() if r is not None]
        skipped = [name for name, r in refs.items() if r is None]
        if skipped:
            logger.warning(f"[decoupled] rollout {rollout_id}: no train step for groups {skipped}")
        await asyncio.gather(*(groups[name].train(rollout_id, refs[name]) for name in trained))

        save_num_rollout = None if os.environ.get("MILES_DISABLE_FINAL_SAVE") else args.num_rollout
        if should_run_periodic_action(rollout_id, args.save_interval, None, save_num_rollout):
            for g in groups.values():
                await g.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
            if args.rollout_global_dataset:
                await rollout_manager.save.remote(rollout_id)

        for g in groups.values():
            if args.offload_train:
                await g.offload()
            else:
                await g.clear_memory()
        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
        for g in groups.values():
            await g.update_weights()
        if args.offload_rollout:
            await rollout_manager.onload_kv.remote()

        if should_run_periodic_action(rollout_id, args.eval_interval, None):
            await rollout_manager.eval.remote(rollout_id)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    cfg_path = _pop_decoupled_config_arg()
    route_key, specs = load_decoupled_config(cfg_path)
    args = parse_args()
    try:
        asyncio.run(train(args, route_key, specs))
    finally:
        finish_tracking()
