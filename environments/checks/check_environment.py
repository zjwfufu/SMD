"""Check an isolated core installation without downloading weights or running training."""
import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-prefix', type=Path)
    parser.add_argument('--cuda', action='store_true', help='Also execute a small CUDA attention/backward/update check')
    args = parser.parse_args()
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append(dict(name=name, status='passed', detail=detail))
        except Exception as error:
            checks.append(dict(name=name, status='failed', error=f'{type(error).__name__}: {error}'[-800:]))

    def isolation():
        assert sys.prefix != sys.base_prefix or (Path(sys.prefix) / 'conda-meta').is_dir(), 'Run in a dedicated Conda or virtual environment'
        if args.expected_prefix is not None:
            assert Path(sys.prefix).resolve() == args.expected_prefix.resolve(), 'Wrong interpreter prefix'
        assert sys.version_info[:2] == (3, 12), 'Core pins require Python 3.12'
        assert platform.machine() == 'x86_64' and sys.platform == 'linux'
        duplicates = {}
        for dist in metadata.distributions():
            name = dist.metadata['Name'].lower().replace('_', '-').replace('.', '-')
            duplicates.setdefault(name, set()).add(str(dist.version))
            assert name != 'fastgen', 'Legacy FastGen distribution is installed'
            direct = json.loads(dist.read_text('direct_url.json') or '{}')
            if direct.get('dir_info', {}).get('editable'):
                assert name == 'smd-reference', f'Unexpected editable package: {name}'
        assert not any(len(v) > 1 for v in duplicates.values()), 'Conflicting package metadata'
        return dict(python=sys.version, executable=sys.executable)

    def pinned_versions():
        from packaging.requirements import Requirement
        pins = ROOT / 'environments/requirements/core-cu126-py312.constraints.txt'
        count = 0
        for line in pins.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            req = Requirement(line)
            actual = metadata.version(req.name)
            assert req.specifier.contains(actual, prereleases=True), f'{req.name}: {actual} != {req.specifier}'
            count += 1
        return dict(pins_checked=count)

    def pip_check():
        run = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        return run.stdout.strip()

    def torch_check():
        import torch
        import torchvision
        assert torch.__version__ == '2.7.1+cu126', torch.__version__
        assert torch.version.cuda == '12.6', torch.version.cuda
        x = torch.arange(8, dtype=torch.float32, requires_grad=True)
        x.square().sum().backward()
        assert torch.equal(x.grad, 2 * x.detach())
        return dict(torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda)

    def cuda_check():
        import torch
        import torch.nn.functional as F
        assert torch.cuda.is_available(), 'CUDA is unavailable'
        results = []
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                q = torch.randn(1, 2, 16, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
                optimizer = torch.optim.AdamW([q], lr=0.01)
                initial = q.detach().clone()
                output = F.scaled_dot_product_attention(q, q, q)
                loss = output.float().square().mean()
                loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(q.grad).all()
                optimizer.step()
                assert not torch.equal(initial, q.detach())
                torch.cuda.synchronize()
                results.append(dict(device=device, name=torch.cuda.get_device_name(device), loss=float(loss)))
        return results

    check('isolation', isolation)
    check('pip_check', pip_check)
    check('pinned_versions', pinned_versions)
    check('torch_cpu_autograd', torch_check)
    from environments.checks.check_flash_attention import check_flash_attention
    check('flash_attention', lambda: check_flash_attention(cuda=args.cuda))
    from environments.checks.check_runtime_dependencies import check_runtime_dependencies
    check('preprocessing_media_runtime', lambda: check_runtime_dependencies(cuda=args.cuda))
    if args.cuda:
        check('cuda_attention_backward_update', cuda_check)
    for name in ['smd.trainer', 'smd.methods.simplified_distill.distillation',
                 'smd.methods.simplified_distill.causal', 'smd.datasets.edit_image_loader']:
        check(name, lambda name=name: str(importlib.import_module(name).__file__))

    def configurations():
        from smd.configs.reference import load_experiment
        catalog = json.loads((ROOT / 'smd/configs/models.json').read_text())
        for name in catalog:
            config = load_experiment(name)
            # Sparse renderer extensions have a separate environment contract.
            if name.startswith('trellis2'):
                continue
            for node in (config.model.net, config.model.teacher):
                if node is not None and node.get('_target_'):
                    module, target = node['_target_'].rsplit('.', 1)
                    getattr(importlib.import_module(module), target)
        return dict(configurations=len(catalog), scope='Resolve all configs; import non-TRELLIS network classes')

    check('configuration_and_network_imports', configurations)

    def source_ownership():
        for name, module in tuple(sys.modules.items()):
            if name == 'smd' or name.startswith('smd.'):
                filename = getattr(module, '__file__', None)
                if filename:
                    assert Path(filename).resolve().is_relative_to(ROOT), f'{name}: {filename}'
        assert not any(name == 'fastgen' or name.startswith('fastgen.') for name in sys.modules), 'Legacy package imported'
        return 'All imported SMD modules belong to this checkout; no legacy package imported'

    check('source_ownership', source_ownership)
    result = dict(status='passed' if all(c['status'] == 'passed' for c in checks) else 'failed',
                  scope='Core installation, pins, CPU and imports; optional small CUDA check. Includes FlashAttention extension loading and optional direct CUDA kernels; excludes 3D extensions and pretrained model runs.',
                  checks=checks,
                  inputs={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ['requirements.txt', 'environments/requirements/core-cu126-py312.constraints.txt',
                                       'pyproject.toml', 'environments/checks/check_environment.py', 'environments/checks/check_flash_attention.py',
                                       'environments/checks/check_runtime_dependencies.py', 'smd/inference/media.py',
                                       'environments/requirements/attention-cu126-py312.requirements.txt']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for item in checks:
        print(f'{item["name"]}: {item["status"]}')
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
