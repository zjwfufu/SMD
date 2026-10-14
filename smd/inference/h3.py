# SPDX-License-Identifier: Apache-2.0
"""Inference from an SMD H3 student adapter and production cached conditions."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

TARGETS = ('to_q', 'to_k', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2')


def launch_if_needed(args):
    """Honor an existing torchrun launch, or launch the requested GPU count."""
    launched = 'LOCAL_RANK' in os.environ
    if launched and 'WORLD_SIZE' not in os.environ:
        raise ValueError('LOCAL_RANK requires WORLD_SIZE from torchrun')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    requested = args.n_gpus if args.n_gpus is not None else (world if launched else 1)
    if requested < 1 or world < 1:
        raise ValueError('H3 GPU count must be positive')
    if not launched and world != 1:
        raise ValueError('WORLD_SIZE requires a torchrun launch with LOCAL_RANK')
    if launched and requested != world:
        raise ValueError(f'--n-gpus={requested} differs from torchrun WORLD_SIZE={world}')
    visible = torch.cuda.device_count()
    if requested > visible:
        raise ValueError(f'H3 requested {requested} GPUs, but only {visible} are visible; check CUDA_VISIBLE_DEVICES')
    if not launched and requested > 1:
        entrypoint = Path(sys.argv[0]).resolve()
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        '--nnodes=1', f'--nproc-per-node={requested}', '--max-restarts=0',
                        str(entrypoint), *sys.argv[1:]], check=True)
        return True
    return False


def validate_sampling(metadata, experiment=None):
    if metadata.get('network_family') != 'minimax_h3':
        raise ValueError('H3 inference requires an exported H3 sampling.json')
    sampler = metadata['sampler']
    if sampler.get('type') != 'minimax_h3_joint_av_euler_student':
        raise ValueError('Unsupported H3 sampler; export the matching training bundle')
    mode = sampler.get('condition_mode')
    if mode not in ('t2va', 'ref2va'):
        raise ValueError('H3 condition_mode must be t2va or ref2va')
    subfolder = 'transformer' if mode == 't2va' else 'transformer_ref'
    if sampler.get('transformer_subfolder') != subfolder:
        raise ValueError('H3 transformer and condition mode disagree')
    steps = sampler.get('student_sample_steps')
    if type(steps) is not int or steps <= 0:
        raise ValueError('H3 student_sample_steps must be a positive integer')
    if experiment is not None and experiment != f'h3_{mode}_{steps}step':
        raise ValueError('Experiment mode/NFE differs from the exported H3 student')
    for key in ('video_shift', 'audio_shift', 'final_step_size_scale'):
        value = sampler.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid H3 {key}')
    if sampler['final_step_size_scale'] > 1:
        raise ValueError('H3 final step scale must be at most one')
    timestep = sampler.get('condition_video_timestep')
    if not isinstance(timestep, (int, float)) or not 0 <= timestep <= 1:
        raise ValueError('Invalid H3 reference video timestep')
    if sampler.get('fps') != 24 or sampler.get('guidance_distilled') is not True:
        raise ValueError('H3 requires 24 fps and guidance-distilled sampling')
    if not sampler.get('model_id') or not sampler.get('model_revision'):
        raise ValueError('H3 sampling metadata must identify its base and revision')
    if metadata.get('model_id') != sampler['model_id']:
        raise ValueError('H3 base identities disagree')
    return sampler


def validate_adapter(state, *, rank, alpha):
    if type(rank) is not int or rank <= 0 or not math.isfinite(alpha) or alpha <= 0 or int(alpha) != alpha:
        raise ValueError('H3 adapter rank and alpha must be positive integers')
    pairs = {}
    for name, value in state.items():
        if name.endswith('.lora_A.weight'):
            module, side = name[:-len('.lora_A.weight')], 'A'
        elif name.endswith('.lora_B.weight'):
            module, side = name[:-len('.lora_B.weight')], 'B'
        else:
            raise ValueError(f'H3 supports original LoRA tensors only, got {name}')
        if not module.endswith(TARGETS) or value.ndim != 2 or value.dtype != torch.bfloat16:
            raise ValueError(f'Invalid H3 adapter matrix: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite H3 adapter matrix: {name}')
        pairs.setdefault(module, {})[side] = value
    if not pairs:
        raise ValueError('Empty H3 adapter')
    for name, pair in pairs.items():
        if set(pair) != {'A', 'B'} or pair['A'].shape[0] != rank or pair['B'].shape[1] != rank:
            raise ValueError(f'Unpaired or mismatched H3 LoRA rank: {name}')


def load_package(directory, experiment):
    from safetensors.torch import load_file

    directory = Path(directory)
    metadata = json.loads((directory / 'sampling.json').read_text())
    sampler = validate_sampling(metadata, experiment)
    adapter = json.loads((directory / 'adapter_config.json').read_text())
    if adapter.get('format') != 'smd_lora' or adapter.get('base_model') != sampler['model_id']:
        raise ValueError('H3 adapter configuration and sampling base disagree')
    if metadata.get('weights') != 'lora.safetensors':
        raise ValueError('H3 requires the exported lora.safetensors')
    path = directory / 'lora.safetensors'
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != metadata.get('weights_sha256'):
        raise ValueError('H3 adapter checksum mismatch; re-export the training checkpoint')
    state = load_file(path, device='cpu')
    validate_adapter(state, rank=adapter['r'], alpha=adapter['lora_alpha'])
    return metadata, adapter, state


def load_case(mode, root, index):
    if index < 0:
        raise ValueError('H3 dataset case index must be non-negative')
    if mode == 't2va':
        from smd.datasets.minimax_h3_prompt_cache import MiniMaxH3PromptCacheDataset, _collate_minimax_h3_cache
        dataset = MiniMaxH3PromptCacheDataset(str(root))
        collate = _collate_minimax_h3_cache
    else:
        from smd.datasets.minimax_h3_ref2va_cache import MiniMaxH3Ref2VACacheDataset, _collate
        dataset = MiniMaxH3Ref2VACacheDataset(str(root), verify_payload_hashes=True)
        collate = _collate
    if index >= len(dataset):
        raise IndexError(f'H3 case {index} is outside the {len(dataset)} cached records')
    data = collate([dataset[index]])
    data['condition'].validate()
    if data['condition'].mode != mode:
        raise ValueError('Cached condition does not match the H3 student mode')
    return data


def _apply_adapter(net, state):
    net.attach_lora()
    state = {key.replace('.lora_A.weight', '.lora_A.default.weight').replace(
        '.lora_B.weight', '.lora_B.default.weight'): value for key, value in state.items()}
    parameters = {name: value for name, value in net.transformer.named_parameters()
                  if '.lora_A.' in name or '.lora_B.' in name}
    if set(parameters) != set(state):
        raise ValueError('H3 adapter does not cover exactly the native transformer LoRA modules')
    if any(parameters[name].shape != value.shape for name, value in state.items()):
        raise ValueError('H3 adapter shapes disagree with the base transformer')
    result = net.transformer.load_state_dict(state, strict=False)
    if result.unexpected_keys or any('.lora_' in name for name in result.missing_keys):
        raise ValueError('H3 adapter loading did not consume all LoRA tensors')


def run(args):
    """All ranks sample one shared case; rank zero decodes after freeing the model."""
    from smd.callbacks.sampling.engine import copy_tree, sample_records
    from smd.callbacks.sampling.rendering.media import write_av_media
    from smd.networks.MiniMaxH3.network import MiniMaxH3
    from smd.networks.MiniMaxH3.decoder import MiniMaxH3AVDecoder
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    if args.prompt is not None:
        raise ValueError('H3 consumes cached prompts/references; use --condition-cache and --dataset-case-index')
    if args.dataset_route is not None or args.reduced_geometry or args.decode_resolution is not None:
        raise ValueError('H3 uses the selected cached case geometry without image/3D overrides')
    metadata, adapter, state = load_package(args.weights, args.experiment)
    sampler = metadata['sampler']
    mode = sampler['condition_mode']
    root = args.condition_cache or Path(f'assets/examples/h3/{mode}' + ('/release' if mode == 'ref2va' else ''))
    index = args.dataset_case_index if args.dataset_case_index is not None else 0
    try:
        data = load_case(mode, root, index)
    except FileNotFoundError as error:
        raise FileNotFoundError(f'Missing H3 condition files. Run python tools/download_resources.py --model {args.experiment}, '
                                'or provide your prepared dataset with --condition-cache.') from error
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ('sample.mp4', 'inference.json')):
        raise FileExistsError('H3 output already exists; select a new output directory')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world > 1
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    rank = dist.get_rank() if distributed else 0
    base = str(args.base) if args.base else sampler['model_id']
    transformer_base = str(args.transformer_base) if args.transformer_base else base
    try:
        # Stream single-GPU weights onto that device rather than retaining a
        # full CPU copy. Distributed inference retains CPU loading before FSDP.
        # Apply the adapter without writing merged base weights to disk.
        net = MiniMaxH3(model_id=base, model_revision=sampler['model_revision'],
            transformer_model_id=transformer_base, transformer_subfolder=sampler['transformer_subfolder'],
            pretrained_device=None if distributed else device,
            use_lora=True, lora_rank=adapter['r'], lora_alpha=int(adapter['lora_alpha']),
            gradient_checkpointing=False, context_parallel_size=world,
            ff_chunk_size=args.ff_chunk_size, qk_norm_chunk_size=args.qk_norm_chunk_size,
            video_shift=sampler['video_shift'], audio_shift=sampler['audio_shift'],
            condition_video_timestep=sampler['condition_video_timestep'])
        _apply_adapter(net, state)
        del state
        net.eval().requires_grad_(False)
        if distributed:
            net.fully_shard(mesh=init_device_mesh('cuda', (world,)),
                           mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, cast_forward_inputs=False))
            net.restore_unsharded_transformer_tensors(device)
        else:
            net.to(device=device, dtype=torch.bfloat16)
        net.validate_official_runtime_contract()
        runtime = SimpleNamespace(device=device, precision=torch.bfloat16)
        runtime.sample_student = lambda noise, condition, net: net.sample(
            noise, condition, num_steps=sampler['student_sample_steps'],
            final_step_size_scale=sampler['final_step_size_scale'])
        evaluations = []
        hook = net.register_forward_hook(lambda *_: evaluations.append(None))
        with torch.no_grad():
            records = sample_records(runtime, net, copy_tree(data, device), grid_size=1, seed=args.seed, include_real=False)
        hook.remove()
        if len(evaluations) != sampler['student_sample_steps']:
            raise RuntimeError('H3 sampler did not execute the exported number of steps')
        schedules = net.build_paired_schedule(sampler['student_sample_steps'], sampler['final_step_size_scale'])
        schedules = {key: value.tolist() for key, value in zip(('video_sigmas', 'audio_sigmas'), schedules)}
        # FSDP ownership must be released on every rank before loading decoders.
        del net, runtime
        gc.collect()
        torch.cuda.empty_cache()
        if distributed:
            dist.barrier()
        if rank == 0:
            decoder = MiniMaxH3AVDecoder(base, sampler['model_revision']).to(device)
            record = records[0]
            shapes = {}
            for key in ('student', 'reference'):
                if key not in record:
                    continue
                pair = record[key]
                if not all(torch.isfinite(value).all() for value in pair.values()):
                    raise RuntimeError('Nonfinite H3 sampled latents')
                decoded = decoder.decode_joint(copy_tree(pair, device))
                if not all(torch.isfinite(decoded[key]).all() for key in ('video', 'waveform')):
                    raise RuntimeError('Nonfinite H3 decoded audio/video')
                name = 'sample' if key == 'student' else 'reference'
                write_av_media(decoded['video'], decoded['waveform'], args.output / name,
                               fps=int(data['fps'].item()), sample_rate=decoded['sample_rate'], normalized=True)
                shapes[name] = {'video': list(decoded['video'].shape), 'audio': list(decoded['waveform'].shape),
                                'sample_rate': decoded['sample_rate']}
                del decoded
            condition = data['condition']
            identity = {'case_index': index, 'prompt': data.get('condition_raw', [None])[0], 'mode': mode,
                        'case': data.get('meta', [{}])[0],
                        'video_latent_shape': data['video_latent_shape'][0].tolist(),
                        'audio_latent_shape': data['audio_latent_shape'][0].tolist(),
                        'num_frames': int(data['num_frames'].item()), 'fps': float(data['fps'].item())}
            (args.output / 'input_case.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
            receipt = {'experiment': args.experiment, 'seed': args.seed, 'sampler': sampler,
                       'weights_sha256': metadata['weights_sha256'], 'decoded': shapes, 'finite': True,
                       'condition_tokens': int(condition.prompt_embeds.shape[1]), 'case_index': index,
                       'n_gpus': world,
                       'forward_evaluations': len(evaluations), **schedules}
            (args.output / 'inference.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(f'H3 inference saved to {args.output}', flush=True)
        if distributed:
            dist.barrier()
    finally:
        if distributed:
            dist.destroy_process_group()
