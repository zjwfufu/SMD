"""Decode and save image, video and 3D student outputs."""
from pathlib import Path
import torch


def decode_3d(net, latent, data, *, resolution, render_size):
    if net.network_family == 'hunyuan3d_shape':
        from smd.networks.Hunyuan3DShape.weight_loader import load_vae
        from smd.callbacks.sampling.rendering.hunyuan3d import decode_and_render_shape_hy
        vae = load_vae(net.model_id).eval().to(device=latent.device, dtype=torch.float32)
        normal, clay = decode_and_render_shape_hy(latent, vae, octree_resolution=resolution or 256,
                                                  render_size=render_size)
        if normal is None or clay is None:
            raise RuntimeError('Shape decoding produced no surface')
        return {'normal': normal, 'clay': clay}
    from smd.networks.Trellis2.network import Trellis2
    from smd.callbacks.sampling.rendering.trellis import decode_and_render_shape, decode_and_render_texture
    shape_config, _ = Trellis2._resolve_from_hf(net.model_id, 'shape_dec_next_dc_f16c32_fp16')
    shape_decoder = str(Path(shape_config).with_suffix(''))
    if net.ckpt_name == net.SHAPE_CKPT:
        views, pbr, _, _ = decode_and_render_shape(latent, shape_decoder, resolution or 1024, render_size)
    else:
        texture_config, _ = Trellis2._resolve_from_hf(net.model_id, 'tex_dec_next_dc_f16c32_fp16')
        result = decode_and_render_texture(latent, data['concat_cond'], shape_decoder,
                                           str(Path(texture_config).with_suffix('')), resolution or 1024, render_size)
        if result is None:
            raise RuntimeError('Texture decoding failed')
        views, pbr = result
    if views is None:
        raise RuntimeError('Sparse decoding produced no rendered surface')
    return {name: value for name, value in [('views', views), ('pbr', pbr)] if value is not None}


def save_media(decoded, output):
    from PIL import Image
    if isinstance(decoded, dict):
        for name, panel in decoded.items():
            pixels = (panel.detach().float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(pixels).save(output / f'{name}.png')
        return
    pixels = ((decoded[0].detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
    if pixels.ndim == 3:
        Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(output / 'sample.png')
    elif pixels.ndim == 4:
        import av
        container = av.open(str(output / 'sample.mp4'), mode='w')
        stream = container.add_stream('libx264', rate=16)
        stream.width, stream.height = pixels.shape[-1], pixels.shape[-2]
        stream.pix_fmt = 'yuv420p'
        for frame in pixels.permute(1, 2, 3, 0).numpy():
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():container.mux(packet)
        container.close()
