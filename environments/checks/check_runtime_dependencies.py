"""Exercise core preprocessing and media dependencies without model weights."""
from pathlib import Path
import tempfile


def check_runtime_dependencies(cuda=False):
    import av
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from torchvision.ops import nms
    from transformers import Qwen2VLImageProcessor
    from smd.inference.media import save_media

    results = {}
    # Calls the production media writer, then decodes the result independently.
    with tempfile.TemporaryDirectory(prefix='smd-media-check-') as tmp:
        output = Path(tmp)
        save_media(torch.zeros(1, 3, 32, 32), output)
        with Image.open(output / 'sample.png') as image:
            assert image.size == (32, 32)
            assert np.asarray(image).shape == (32, 32, 3)
        save_media(torch.zeros(1, 3, 4, 32, 32), output)
        with av.open(str(output / 'sample.mp4')) as container:
            frames = list(container.decode(video=0))
            assert len(frames) == 4
            assert all(f.to_ndarray(format='rgb24').shape == (32, 32, 3) for f in frames)
        results['production_png_h264_roundtrip'] = dict(frames=4, width=32, height=32)

    for device in (['cpu', 'cuda'] if cuda else ['cpu']):
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        scores = torch.tensor([.9, .5], device=device)
        assert nms(boxes, scores, .5).tolist() == [0]
    results['torchvision_compiled_ops'] = ['cpu', 'cuda'] if cuda else ['cpu']

    processor = Qwen2VLImageProcessor()
    pixels = processor(images=Image.new('RGB', (112, 112)), return_tensors='pt')
    assert pixels['pixel_values'].numel() > 0 and torch.isfinite(pixels['pixel_values']).all()
    assert pixels['image_grid_thw'].shape == (1, 3)
    results['qwen_vl_image_processor'] = dict(pixel_shape=list(pixels['pixel_values'].shape))

    # Match the exercised core profile; native MXFP4 is a separate runtime choice.
    from transformers import Mxfp4Config
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    quantizer = Mxfp4HfQuantizer(Mxfp4Config(dequantize=False), pre_quantized=True)
    quantizer.validate_environment()
    assert quantizer.quantization_config.dequantize, 'Core profile expects Lens BF16 dequantization'
    results['lens_text_encoder_policy'] = 'BF16 dequantization; native MXFP4 kernels are not enabled'

    root = Path(__file__).resolve().parents[2]
    rows = {}
    for path in sorted((root / 'assets/sd35').glob('*.parquet')):
        batch = next(pq.ParquetFile(path).iter_batches(batch_size=2))
        assert batch.num_rows > 0
        assert len(batch.to_pandas()) == batch.num_rows
        rows[path.name] = batch.num_rows
    assert len(rows) == 5, 'Expected all five routed-prompt tables'
    results['routed_prompt_arrow_pandas'] = rows
    return results
