# SPDX-License-Identifier: Apache-2.0
"""Fixed raw inputs, independent of the advancing training iterator."""

from omegaconf import OmegaConf


def dataset_cases(dataset, indices):
    """Read named production-loader cases in file order, including paired media.

    ``loader`` is an import path rather than a nested LazyCall: constructing a
    callback must not construct a training dataloader or start worker processes.
    """
    config = dict(OmegaConf.to_container(dataset, resolve=True) if OmegaConf.is_config(dataset) else dataset)
    target = config.pop("loader")
    if "MiniMaxH3" in target:
        if target.endswith("MiniMaxH3PromptCacheLoader"):
            from smd.datasets.minimax_h3_prompt_cache import MiniMaxH3PromptCacheDataset, _collate_minimax_h3_cache

            source = MiniMaxH3PromptCacheDataset(config["dataset_path"])
            collate = _collate_minimax_h3_cache
        else:
            import inspect

            from smd.datasets.minimax_h3_ref2va_cache import MiniMaxH3Ref2VACacheDataset, _collate

            accepted = inspect.signature(MiniMaxH3Ref2VACacheDataset).parameters
            source = MiniMaxH3Ref2VACacheDataset(
                config["dataset_path"], **{k: v for k, v in config.items() if k in accepted}
            )
            collate = _collate
        return [collate([source[index]]) for index in indices]
    from smd.inference.formal_dataset_cases import select_formal_case

    config["_target_"] = target
    return [select_formal_case(OmegaConf.create(config), index)[0] for index in indices]
