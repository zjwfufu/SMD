"""Finite, deterministic case selection using the production tar readers and collators."""
import base64
import hashlib
import io
import random
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image


def select_formal_case(loader_config, index):
    config = OmegaConf.to_container(loader_config, resolve=True)
    edit = config['_target_'].endswith('.EditImageTarLoader')
    config.update(batch_size=1, num_workers=0, sampler_start_idx=0, seed=0)
    config['shuffle_buffer' if edit else 'shuffle'] = 0 if edit else False
    loader = instantiate(config)
    dataset = loader._dataset if edit else loader.dataset
    dataset.rng = random.Random(0)
    shards = dataset.shard_urls if edit else dataset._shard_paths
    remaining = index
    for shard in sorted(shards):
        records = dataset._iter_records_from_tar(shard) if edit else dataset._iter_shard(shard)
        try:
            for record in records:
                sample = dataset._process_record(record) if edit else record
                if sample is None:
                    continue
                if remaining:
                    remaining -= 1
                    continue
                if edit:
                    batch = loader._collate_bucket_batch([sample], sample['bucket_hw'])
                    with Image.open(io.BytesIO(base64.b64decode(record['edited_img'], validate=True))) as image:
                        batch['reference_pil'] = [image.convert('RGB').copy()]
                    image_bytes = base64.b64decode(record['source_img'], validate=True)
                else:
                    batch = loader.loader.collate_fn([sample])
                    image_bytes = sample.get('cond_image_bytes')
                    if image_bytes is None:
                        raise ValueError('Visual inference cases require a condition image in the shard')
                manifest = dict(dataset_path=str(Path(shard)),
                    dataset_sha256=hashlib.sha256(Path(shard).read_bytes()).hexdigest(),
                    sample_id=sample['fname'], prompt=batch['condition'][0] if edit else '',
                    image_path=f"{shard}#{sample['fname']}",
                    image_sha256=hashlib.sha256(image_bytes).hexdigest(), input_image='input.png')
                if edit:
                    manifest.update(reference_image='reference.png',
                        reference_sha256=hashlib.sha256(base64.b64decode(record['edited_img'], validate=True)).hexdigest())
                if 'real' in batch and not edit:
                    manifest['voxel_count'] = len(batch['real'].coords)
                if 'concat_cond' in batch:
                    manifest['shape_condition'] = 'Normalized shape latent from the same object and sparse coordinates'
                return batch, manifest
        finally:
            records.close()
    raise IndexError(f'Case {index} is outside the configured tar dataset')
