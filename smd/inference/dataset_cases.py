"""Select a reproducible raw case from an experiment's own training examples."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_case(config, experiment, index=0, route=None):
    if index < 0:
        raise ValueError('Case index must be non-negative')
    loader = config.dataloader_train
    target = loader['_target_']
    manifest = {'source': 'training_dataset', 'loader': target, 'case_index': index}
    if target.endswith(('.EditImageTarLoader', '.Hunyuan3DShapeImageLoader',
                        '.TrellisShapeWDSLoader', '.TrellisTextureWDSLoader')):
        if route is not None:
            raise ValueError('Image datasets do not have reward routes')
        from smd.inference.formal_dataset_cases import select_formal_case
        data, details = select_formal_case(loader, index)
        manifest.update(details)
        return data, manifest
    if target.endswith('.RoutedPromptLoader'):
        paths = loader['route_paths']
        defaults = {'sd35_clipscore': 'clipscore', 'sd35_geneval': 'geneval',
                    'sd35_ocr': 'ocr', 'sd35_pickscore': 'preference'}
        if route is None:
            route = defaults.get(experiment)
            if route is None:
                import numpy as np
                from smd.datasets.routed_prompt import pick_route
                names = sorted(paths)
                weights = np.asarray([loader['route_weights'][n] for n in names], dtype=float)
                route = pick_route(int(loader.get('seed', 0)), 0, names, weights / weights.sum())
        if route not in paths:
            raise ValueError(f'Unknown dataset route: {route}')
        source = paths[route]
        manifest['route'] = route
    elif target.endswith('.PromptOnlyLoader'):
        if route is not None:
            raise ValueError('This dataset has no routes')
        source = loader['dataset_path']
    else:
        raise ValueError(f'Unsupported dataset loader: {target}')
    from smd.datasets.prompt_only import PromptOnlyDataset
    source_path = Path(source)
    if not source_path.is_absolute(): source_path = ROOT / source_path
    dataset = PromptOnlyDataset(data_path=str(source_path), shuffle=False, max_samples=index + 1,
        prompt_key=loader.get('prompt_key'), file_glob=loader.get('file_glob', '*'),
        allow_empty_prompt=loader.get('allow_empty_prompt', False))
    if index >= len(dataset):
        raise IndexError(f'Case {index} is outside dataset {source}')
    prompt = dataset[index]['condition']
    data = {'condition': [prompt], 'neg_condition': [config.model.neg_prompt or '']}
    if route is not None: data['route'] = route
    manifest.update(dataset_path=str(source), dataset_sha256=checksum(source_path), prompt=prompt,
                    negative_prompt=data['neg_condition'][0])
    return data, manifest


def save_input(data, manifest, output):
    """Keep the source image beside output media for an honest visual comparison."""
    if not manifest.get('input_image'): return
    from PIL import Image
    import io
    if data.get('source_pil'):
        image = data['source_pil'][0]
    else:
        image = Image.open(io.BytesIO(data['condition_images'][0]))
    image.save(output / manifest['input_image'])
    if manifest.get('reference_image'):
        if data.get('reference_pil'):
            data['reference_pil'][0].save(output / manifest['reference_image'])
            return
        raise ValueError('Reference image is missing from the selected dataset case')
