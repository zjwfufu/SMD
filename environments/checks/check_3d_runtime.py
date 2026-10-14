"""Small CUDA operator checks before the user-run three-model acceptance."""
import argparse
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'checks': {}}
    stage = 'cuda'
    try:
        import torch
        assert torch.cuda.is_available(), 'CUDA is required'
        torch.manual_seed(42)
        report['torch'] = torch.__version__
        report['gpu'] = torch.cuda.get_device_name()
        stage = 'flash_attention_backward'
        from flash_attn import flash_attn_varlen_qkvpacked_func
        qkv = torch.randn(16, 3, 2, 32, device='cuda', dtype=torch.float16, requires_grad=True)
        cu = torch.tensor([0, 8, 16], device='cuda', dtype=torch.int32)
        output = flash_attn_varlen_qkvpacked_func(qkv, cu, 8, 0.0)
        output.float().square().mean().backward()
        assert torch.isfinite(output).all() and torch.isfinite(qkv.grad).all() and qkv.grad.abs().sum() > 0
        report['checks'][stage] = 'passed'
        stage = 'flex_gemm_sparse_convolution'
        from smd.networks.Trellis2.sparse import SparseTensor
        from smd.networks.Trellis2.sparse.conv import SparseConv3d
        xyz = torch.cartesian_prod(*[torch.arange(4, device='cuda', dtype=torch.int32)] * 3)
        coords = torch.cat([torch.zeros(64, 1, device='cuda', dtype=torch.int32), xyz], 1)
        features = torch.randn(64, 16, device='cuda', dtype=torch.float16, requires_grad=True)
        conv = SparseConv3d(16, 16, 3).cuda().half()
        actual = conv(SparseTensor(feats=features, coords=coords)).feats
        dense = features.detach().reshape(4, 4, 4, 16).permute(3, 0, 1, 2).unsqueeze(0).float()
        reference = torch.nn.functional.conv3d(dense, conv.weight.detach().permute(0, 4, 1, 2, 3).float(), conv.bias.detach().float(), padding=1)
        reference = reference.squeeze(0).permute(1, 2, 3, 0).reshape(64, 16)
        torch.testing.assert_close(actual.float(), reference, atol=0.005, rtol=0.005)
        actual.float().square().mean().backward()
        assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
        assert torch.isfinite(conv.weight.grad).all() and conv.weight.grad.abs().sum() > 0
        report['checks'][stage] = {'status': 'passed', 'max_abs_vs_fp32_dense': (actual.float() - reference).abs().max().item()}
        stage = 'cuda_rasterization'
        import nvdiffrast.torch as dr
        positions = torch.tensor([[[-0.8, -0.8, 0., 1.], [0.8, -0.8, 0., 1.], [0., 0.8, 0., 1.]]], device='cuda', requires_grad=True)
        triangles = torch.tensor([[0, 1, 2]], device='cuda', dtype=torch.int32)
        raster, _ = dr.rasterize(dr.RasterizeCudaContext(), positions, triangles, resolution=[32, 32])
        assert torch.isfinite(raster).all() and (raster[..., 3] > 0).sum() > 100
        raster[..., 0].sum().backward()
        assert torch.isfinite(positions.grad).all() and positions.grad.abs().sum() > 0
        report['checks'][stage] = 'passed'
        stage = 'cuda_mesh'
        import cumesh
        vertices = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], device='cuda')
        faces = torch.tensor([[0,2,1], [0,1,3], [0,3,2], [1,2,3]], device='cuda', dtype=torch.int32)
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.get_edges()
        v, f = mesh.read()
        assert v.shape == vertices.shape and f.shape == faces.shape and torch.isfinite(v).all()
        report['checks'][stage] = 'passed'
        torch.cuda.synchronize()
        report.update(status='passed', scope='CUDA operators only; model training and decoded outputs require separate acceptance')
    except Exception as error:
        report.update(status='failed', stage=stage, error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
