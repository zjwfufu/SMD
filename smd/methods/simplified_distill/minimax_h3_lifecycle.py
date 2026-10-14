# SPDX-License-Identifier: Apache-2.0
"""H3 FSDP mixed precision, reverse-tied teacher and stable optimizer ownership."""
import torch


class H3LifecycleMixin:
    def build_student(self):
        # Process-group construction is collective: do it before rank0-first I/O.
        from smd.networks.MiniMaxH3.context_parallel import get_context_parallel_state
        get_context_parallel_state(int(self.config.net.get("context_parallel_size", 1)))
        self.net = self._coordinated_instantiate(
            self.config.net,
            all_rank_read=not bool(self.config.fsdp_meta_init),
        )
        self.net.to(dtype=self.precision_fsdp)
        self.net.train().requires_grad_(True)
        self._build_preprocessors(self.net)
        self.initialize_student()

    def _build_reverse_tied_teacher(self):
        import smd.utils.logging_utils as logger
        from smd.utils import instantiate
        from smd.utils.distributed.tie_weights import materialize_independent_params, reverse_tie_frozen_base

        logger.info("tie_backbone: building MiniMax H3 meta-skeleton teacher")
        if self.precision != self.precision_fsdp:
            raise ValueError(
                "tie_backbone reverse-tie requires precision == precision_fsdp "
                f"(got precision={self.precision}, precision_fsdp={self.precision_fsdp})."
            )
        with torch.device("meta"):
            self.teacher = instantiate(self.teacher_config)
        self.teacher.to(dtype=self.precision_fsdp)
        result = reverse_tie_frozen_base(
            self.backend.get_transformer(self.teacher),
            self.backend.get_transformer(self.net),
        )
        self._h3_deferred_teacher_materialization = any(parameter.is_meta for parameter in self.net.parameters())
        if not self._h3_deferred_teacher_materialization:
            materialize_independent_params(self.teacher, self.net, requires_grad=False, device=self.device)
        self.teacher.eval().requires_grad_(False)
        self._teacher_reverse_result = result
        self._reverse_tied = True
        logger.info(
            f"tie_backbone reverse-tie: identity_modules={len(result.identity_modules)} "
            f"base_layer_binds={len(result.base_layer_binds)} independent={len(result.independent_params)} "
            f"deferred_materialization={self._h3_deferred_teacher_materialization}"
        )

    @staticmethod
    def _broadcast_meta_parameters(module: torch.nn.Module, names: list[str] | None = None) -> None:
        modules = dict(module.named_modules())
        selected = names if names is not None else [name for name, _ in module.named_parameters()]
        for full_name in selected:
            module_name, _, parameter_name = full_name.rpartition(".")
            owner = modules[module_name] if module_name else module
            parameter = owner.get_parameter(parameter_name)
            if parameter.is_meta:
                parameter = torch.nn.Parameter(
                    torch.empty(parameter.shape, device=torch.cuda.current_device(), dtype=parameter.dtype),
                    requires_grad=False,
                )
                setattr(owner, parameter_name, parameter)
            torch.distributed.broadcast(parameter.data, src=0)
        remaining = [name for name, parameter in module.named_parameters() if parameter.is_meta]
        if remaining:
            raise RuntimeError(
                f"MiniMax H3 module still has {len(remaining)} meta parameters after rank0 broadcast: {remaining[:5]}"
            )

    @property
    def fsdp_no_sync_keys(self) -> set | None:
        keys = set(super().fsdp_no_sync_keys or ())
        if self.config.fsdp_meta_init:
            keys.discard("net")
        return keys or None

    def on_train_begin(self, is_fsdp: bool = False) -> None:
        if not (is_fsdp and self._reverse_tied):
            super().on_train_begin(is_fsdp=is_fsdp)
            return

        teacher = self.teacher
        self.teacher = None
        try:
            super().on_train_begin(is_fsdp=is_fsdp)
        finally:
            self.teacher = teacher
        self.pivot.video_proj_out.to(dtype=torch.float32)
        self.pivot.audio_proj_out.to(dtype=torch.float32)

    def post_fsdp_wrap(self):
        if self.config.fsdp_meta_init:
            teacher_transformer = self.backend.get_transformer(self.teacher)
            self._broadcast_meta_parameters(
                teacher_transformer,
                list(self._teacher_reverse_result.independent_params),
            )
            self._h3_deferred_teacher_materialization = False
        super().post_fsdp_wrap()
        self.net.restore_unsharded_transformer_tensors(self.device)
        self.net.validate_official_runtime_contract()
        self.teacher.validate_official_runtime_contract()
        for name in ("video_proj_out", "audio_proj_out"):
            dtype = getattr(self.pivot, name).weight.dtype
            if dtype != torch.float32:
                raise RuntimeError(f"MiniMax H3 Pivot {name} must remain FP32, got {dtype}.")

    def init_optimizers(self) -> None:
        super().init_optimizers()
        names_by_id = {id(parameter): name for name, parameter in self.net.named_parameters()}
        self._net_optimizer_param_names = [
            [names_by_id[id(parameter)] for parameter in group["params"]] for group in self.net_optimizer.param_groups
        ]
        # Reuse the existing trainable-only checkpoint rebind protocol without
        # changing shared checkpointer infrastructure.
        self.net_optimizer._smd_checkpoint_rebind_param_names = self._net_optimizer_param_names
        self._stable_net_optimizer_parameters = [list(group["params"]) for group in self.net_optimizer.param_groups]

    @staticmethod
    def _local_tensor(value: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        return value._local_tensor if isinstance(value, DTensor) else value

    def _transfer_net_gradients_to_stable_parameters(self) -> None:
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor, distribute_tensor

        for module in self.net.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        current = dict(self.net.named_parameters())
        for group, names, stable_parameters in zip(
            self.net_optimizer.param_groups,
            self._net_optimizer_param_names,
            self._stable_net_optimizer_parameters,
        ):
            previous_parameters = list(group["params"])
            group["params"] = stable_parameters
            for name, previous, stable in zip(names, previous_parameters, stable_parameters):
                if previous is not stable:
                    state = self.net_optimizer.state.pop(previous, None)
                    if state is not None:
                        if stable in self.net_optimizer.state:
                            raise RuntimeError(f"MiniMax H3 optimizer state already exists for {name!r}.")
                        self.net_optimizer.state[stable] = state
                source = current[name]
                if source.grad is None:
                    stable.grad = None
                    continue
                source_grad = self._local_tensor(source.grad)
                if isinstance(stable, DTensor):
                    local_shape = tuple(stable.to_local().shape)
                    if tuple(source_grad.shape) == tuple(stable.shape):
                        stable.grad = distribute_tensor(
                            source_grad.to(device=self.device),
                            device_mesh=stable.device_mesh,
                            placements=stable.placements,
                        ).to(device=stable.device)
                    elif tuple(source_grad.shape) == local_shape:
                        # from_local follows the CUDA mesh even for a CPU shard;
                        # restore the optimizer parameter's offload device.
                        stable.grad = DTensor.from_local(
                            source_grad.to(device=stable.device),
                            device_mesh=stable.device_mesh,
                            placements=stable.placements,
                            run_check=False,
                            shape=stable.shape,
                            stride=stable.stride(),
                        ).to(device=stable.device)
                    else:
                        raise RuntimeError(
                            f"MiniMax H3 gradient shape mismatch for {name!r}: source={tuple(source_grad.shape)}, "
                            f"stable_global={tuple(stable.shape)}, stable_local={local_shape}."
                        )
                else:
                    stable.grad = source_grad.to(device=stable.device)

    def optimizers_schedulers_step(self, iteration: int) -> None:
        self._transfer_net_gradients_to_stable_parameters()
        super().optimizers_schedulers_step(iteration)

    @property
    def optimizer_dict(self):
        current = dict(self.net.named_parameters())
        for group, names in zip(self.net_optimizer.param_groups, self._net_optimizer_param_names):
            old_parameters = group["params"]
            new_parameters = [current[name] for name in names]
            for old, new in zip(old_parameters, new_parameters):
                if old is new:
                    continue
                state = self.net_optimizer.state.pop(old, None)
                if state is not None:
                    self.net_optimizer.state[new] = state
            group["params"] = new_parameters
        return super().optimizer_dict
