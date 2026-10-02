"""Decoupled multi-policy training (opt-in; entry point ``train_decoupled.py``).

Trains N independent policies in one miles job. Each policy ("train group") owns a disjoint
slice of the colocated GPUs, its own Megatron train group, its own SGLang model (from
``--sglang-config``, all with ``update_weights: true``) and its own checkpoint dirs. Samples
are routed to a group by a metadata key (e.g. ``metadata["phase"]`` == "act" / "write").

Nothing here changes the single-policy path: ``train.py``, ``RolloutManager`` and the train
actors are used as-is. The pieces are
  - ``DecoupledRolloutManager``: a ``RolloutManager`` subclass that adds per-group data
    splitting and per-model engine lookup (the base class allows only one updatable model);
  - ``GroupManagerProxy``: handed to each train group in place of the rollout manager, so the
    unmodified train actors' calls (set_train_parallel_config, get_updatable_engines_and_lock,
    clear_updatable_has_new_engines, recover_updatable_engines) resolve to that group's model;
  - ``clone_args_for_group``: per-group copy of the parsed args (GPU count, DP size, ckpt dirs).

Config (YAML, path given to ``train_decoupled.py --decoupled-config``)::

    route_key: phase                 # metadata key used to route samples
    groups:
      - name: act                    # first group keeps the main --load/--save/--save-hf
        sglang_model: default        # model name in --sglang-config
        route_value: act
        num_gpus: 6
      - name: write
        sglang_model: writer
        route_value: write
        num_gpus: 2
        load: /path/ckpt_writer      # required for every group but the first
        save: /path/ckpt_writer
        save_hf: /path/hf_writer/iter_{rollout_id}   # optional
        tracking_prefix: "write/"    # optional, default "<name>/"

Group GPU slices are laid out in config order over the colocated placement group, and the
sglang models must use the same order and GPU counts so each group's engines sit on its GPUs.
"""

import asyncio
import copy
import dataclasses
import logging
import os
import time

import ray
import yaml

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.ray.rollout.metrics import log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_manager import EnginesAndLock, RolloutManager
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.rollout.base_types import RolloutFnTrainInput, call_rollout_fn
from miles.rollout.inference_rollout.compatibility import call_rollout_function

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GroupSpec:
    name: str
    sglang_model: str
    route_value: str
    num_gpus: int
    load: str | None = None
    save: str | None = None
    save_hf: str | None = None
    tracking_prefix: str | None = None
    gpu_base: int = 0  # filled in by load_decoupled_config


def load_decoupled_config(path: str) -> tuple[str, list[GroupSpec]]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    route_key = cfg.get("route_key", "phase")
    specs = [GroupSpec(**g) for g in cfg["groups"]]
    assert len(specs) >= 2, "decoupled training needs at least two groups"
    assert len({s.name for s in specs}) == len(specs), "group names must be unique"
    assert len({s.sglang_model for s in specs}) == len(specs), "each group needs its own sglang model"
    base = 0
    for i, s in enumerate(specs):
        s.gpu_base = base
        base += s.num_gpus
        if i > 0:
            assert s.load and s.save, f"group {s.name}: load and save are required for non-first groups"
    return route_key, specs


def check_layout(args, specs: list[GroupSpec]) -> None:
    """Group GPU slices must tile the colocated actor GPUs and match the sglang models 1:1."""
    from miles.backends.sglang_utils.sglang_config import SglangConfig

    total = args.actor_num_nodes * args.actor_num_gpus_per_node
    assert args.colocate, "decoupled training is implemented for --colocate only"
    assert args.actor_num_nodes == 1, "decoupled training is implemented for a single node only"
    assert sum(s.num_gpus for s in specs) == total, f"group GPUs {[s.num_gpus for s in specs]} != {total}"
    assert args.sglang_config, "decoupled training needs --sglang-config with one model per group"
    models = SglangConfig.from_yaml(args.sglang_config).models
    assert [m.name for m in models] == [s.sglang_model for s in specs], (
        f"sglang models {[m.name for m in models]} must match groups {[s.sglang_model for s in specs]} in order"
    )
    for m, s in zip(models, specs, strict=True):
        n = sum(g.num_gpus for g in m.server_groups)
        assert n == s.num_gpus, f"sglang model {m.name} has {n} GPUs, group {s.name} has {s.num_gpus}"
        assert m.update_weights is True, f"sglang model {m.name} must set update_weights: true explicitly"


def clone_args_for_group(args, spec: GroupSpec, is_first: bool):
    """Per-group copy of the driver args. Mirrors what parse_args derives from the GPU count."""
    g = copy.deepcopy(args)
    n = spec.num_gpus
    g.actor_num_gpus_per_node = n
    g.world_size = n
    model_parallel = g.tensor_model_parallel_size * g.pipeline_model_parallel_size * g.context_parallel_size
    assert n % model_parallel == 0, f"group {spec.name}: {n} GPUs not divisible by TP*PP*CP={model_parallel}"
    g.data_parallel_size = n // model_parallel
    # Megatron's microbatch calculator (initialize.py) requires gbs % (mbs * dp) == 0. With
    # --use-dynamic-global-batch-size every rollout ships its own gbs, so the static value only
    # feeds that check and the (constant-LR) scheduler increments; round it up to fit.
    unit = g.micro_batch_size * g.data_parallel_size
    if g.global_batch_size % unit:
        new_gbs = -(-g.global_batch_size // unit) * unit
        logger.info(f"[decoupled] group {spec.name}: static global_batch_size {g.global_batch_size} -> {new_gbs}")
        g.global_batch_size = new_gbs
    if not is_first:
        g.load, g.save = spec.load, spec.save
        g.save_hf = spec.save_hf  # never inherit the first group's HF export dir
        # Same first-launch fallback parse_args applies to the main --load.
        if not os.path.exists(os.path.join(g.load, "latest_checkpointed_iteration.txt")):
            g.no_load_optim = True
            g.no_load_rng = True
            g.finetune = True
            g.load = g.ref_load
            if g.ref_ckpt_step is not None:
                g.ckpt_step = g.ref_ckpt_step
    g.tracking_key_prefix = spec.tracking_prefix if spec.tracking_prefix is not None else (
        "" if is_first else f"{spec.name}/"
    )
    g.decoupled_group_name = spec.name
    return g


def _subset_train_data(data: dict, idx: list[int], n: int) -> dict:
    """Select rows idx from every per-sample list of a train-data dict (other keys copied)."""
    out = {}
    for k, v in data.items():
        out[k] = [v[i] for i in idx] if isinstance(v, list) and len(v) == n else v
    return out


@ray.remote
class DecoupledRolloutManager(RolloutManager.__ray_actor_class__):
    """RolloutManager plus per-group data routing and per-model engine lookup."""

    def configure_groups(self, route_key: str, specs: list[GroupSpec]) -> None:
        self._route_key = route_key
        self._group_specs = specs
        self._group_parallel: dict[str, dict] = {}

    def set_group_train_parallel_config(self, group: str, config: dict) -> None:
        self._group_parallel[group] = config

    # -------------------------- per-model engines -----------------------------

    async def get_model_engines_and_lock(self, model_name: str, gpu_base: int) -> EnginesAndLock:
        srv = self.servers[model_name]
        assert srv.update_weights, f"sglang model {model_name} is not updatable"
        await srv.wait_all_engines_alive()
        # Engine offsets are rollout-slot indices; the group's train ranks start at 0 on its
        # first GPU, so rebase them for the colocated IPC rank -> engine mapping.
        return EnginesAndLock(
            rollout_engines=[e.actor_handle for e in srv.engines],
            rollout_engine_lock=self.rollout_engine_lock,
            has_new_engines=srv.has_new_engines,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=[o - gpu_base for o in srv.engine_gpu_offsets],
        )

    def clear_model_has_new_engines(self, model_name: str) -> None:
        self.servers[model_name].clear_has_new_engines()

    async def recover_model_engines(self, model_name: str) -> None:
        self._health_monitoring_pause()
        if self.rollout_id == -1:
            return
        await self.servers[model_name].recover()

    # -------------------------- data generation -----------------------------

    async def generate_decoupled(self, rollout_id) -> dict:
        """One rollout -> {group name: per-DP-rank data refs, or None if the group has too few samples}."""
        start_time = time.time()
        self.rollout_id = rollout_id
        self._health_monitoring_resume()
        if self.use_experimental_refactor:
            out = await asyncio.to_thread(
                call_rollout_function, self.generate_rollout, RolloutFnTrainInput(rollout_id=rollout_id)
            )
        else:
            out = await asyncio.to_thread(
                call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
            )
        metrics, samples = out.metrics, out.samples
        while samples and isinstance(samples[0], list):
            samples = [s for group in samples for s in group]

        # Route, then trim each group to its own DP multiple (the base path trims the whole
        # batch once against the single DP size). Trimming before reward post-processing keeps
        # the base order: post-processing sees exactly the samples that are trained.
        kept, kept_meta, group_ranges = [], {}, {}
        for spec in self._group_specs:
            mine = [s for s in samples if (s.metadata or {}).get(self._route_key) == spec.route_value]
            dp = self._group_parallel[spec.name]["dp_size"]
            if len(mine) < dp:
                logger.warning(f"[decoupled] rollout {rollout_id} group {spec.name}: {len(mine)} samples < dp {dp}, skipped")
                continue
            mine, meta = postprocess_rollout_data(self.args, mine, train_parallel_config=self._group_parallel[spec.name])
            group_ranges[spec.name] = (len(kept), len(kept) + len(mine))
            kept_meta[spec.name] = meta
            kept.extend(mine)
        routed = sum(1 for s in samples if (s.metadata or {}).get(self._route_key) in {g.route_value for g in self._group_specs})
        if routed != len(samples):
            logger.warning(f"[decoupled] rollout {rollout_id}: {len(samples) - routed} samples matched no group, dropped")
        logger.info(
            f"[decoupled] rollout {rollout_id}: "
            + ", ".join(f"{k}={b - a}" for k, (a, b) in group_ranges.items())
        )
        if not kept:
            return {spec.name: None for spec in self._group_specs}

        save_debug_rollout_data(self.args, kept, rollout_id=rollout_id, evaluation=False)
        log_rollout_data(rollout_id, self.args, kept, metrics, time.time() - start_time)
        data = convert_samples_to_train_data(
            self.args,
            kept,
            # Placeholder for the converter's dynamic-gbs check; each group's own value is set below.
            metadata={"dynamic_global_batch_size": len(kept)} if self.args.use_dynamic_global_batch_size else {},
            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
            custom_reward_post_process_func=self.custom_reward_post_process_func,
        )
        refs = {}
        for spec in self._group_specs:
            if spec.name not in group_ranges:
                refs[spec.name] = None
                continue
            a, b = group_ranges[spec.name]
            sub = _subset_train_data(data, list(range(a, b)), len(kept))
            if "dynamic_global_batch_size" in kept_meta[spec.name]:
                sub["dynamic_global_batch_size"] = kept_meta[spec.name]["dynamic_global_batch_size"]
            refs[spec.name] = split_train_data_by_dp(self.args, sub, self._group_parallel[spec.name]["dp_size"])
        return refs


@ray.remote(num_cpus=0)
class GroupManagerProxy:
    """Stands in for the rollout manager inside one train group (driver and train actors)."""

    def __init__(self, manager, group: str, model_name: str, gpu_base: int):
        self._m, self._group, self._model, self._base = manager, group, model_name, gpu_base

    async def set_train_parallel_config(self, config: dict):
        return await self._m.set_group_train_parallel_config.remote(self._group, config)

    async def get_updatable_engines_and_lock(self):
        return await self._m.get_model_engines_and_lock.remote(self._model, self._base)

    async def clear_updatable_has_new_engines(self):
        return await self._m.clear_model_has_new_engines.remote(self._model)

    async def recover_updatable_engines(self):
        return await self._m.recover_model_engines.remote(self._model)


async def create_decoupled_training_models(args, pgs, rollout_manager, specs: list[GroupSpec]) -> dict:
    """One RayTrainGroup per spec on its GPU slice; returns {name: group}. Inits sequentially so
    each group's rank 0 has bound its MASTER_PORT before the next group picks one."""
    from miles.ray.placement_group import allocate_train_group

    pg, bundle_idx, gpu_ids = pgs["actor"]
    groups, start_ids = {}, {}
    for i, spec in enumerate(specs):
        gargs = clone_args_for_group(args, spec, is_first=(i == 0))
        lo, hi = spec.gpu_base, spec.gpu_base + spec.num_gpus
        group = allocate_train_group(
            args=gargs,
            num_nodes=1,
            num_gpus_per_node=spec.num_gpus,
            pg=(pg, bundle_idx[lo:hi], gpu_ids[lo:hi]),
            role="actor",
            with_ref=gargs.kl_coef != 0 or gargs.use_kl_loss,
        )
        ids = await group.init()
        assert len(set(ids)) == 1, f"group {spec.name}: ranks disagree on start rollout {ids}"
        start_ids[spec.name] = ids[0]
        proxy = GroupManagerProxy.remote(rollout_manager, spec.name, spec.sglang_model, spec.gpu_base)
        await group.set_rollout_manager(proxy)
        groups[spec.name] = group
    assert len(set(start_ids.values())) == 1, f"groups resume from different rollouts: {start_ids}"
    if args.start_rollout_id is None:
        args.start_rollout_id = next(iter(start_ids.values()))
    if args.rollout_global_dataset:
        await rollout_manager.load.remote(args.start_rollout_id - 1)
    return groups
