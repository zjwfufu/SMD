"""User-run, resumable 3D extension build. No model weights or training jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'LOCAL_DEV/clean_environment_3d'
INVENTORY = """import importlib.metadata as m,json,sys
print(json.dumps({'python':sys.version,'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}}))"""


def capture(command):
    return subprocess.check_output([str(x) for x in command], text=True).strip()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run_source_command(command, *, cwd, log):
    """Retry network Git operations without retrying deterministic builds."""
    command = [str(x) for x in command]
    remote = command[0] == 'git' and ('fetch' in command or 'submodule' in command)
    attempts = 3 if remote else 1
    for attempt in range(attempts):
        try:
            subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=900 if remote else None)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt + 1 == attempts:
                raise
            print(f'Git transfer failed; retry {attempt + 2}/{attempts}', file=log, flush=True)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda', required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--env-root', type=Path)
    args = parser.parse_args()
    env_root = args.env_root or Path(os.environ.get('SMD_ENV_ROOT', args.base / 'envs'))
    source = env_root / 'smd'
    target = env_root / 'smd_3d_extend'
    python = target / 'bin/python'
    stage = 'preflight'
    state = {'status': 'running', 'stages': {}, 'target': str(target)}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        baseline = json.loads(capture([source / 'bin/python', '-c', INVENTORY]))
        assert sys.version_info[:2] == (3, 12), 'Run with the smd Python 3.12 interpreter'
        assert baseline['packages']['torch'] == '2.7.1+cu126', 'Expected the validated CUDA 12.6 Torch build'
        assert baseline['packages']['flash-attn'] == '2.7.4.post1', 'Install and validate base FlashAttention first'
        toolkit = os.environ.get('CUDA_HOME')
        if not toolkit:
            candidates = [Path('/usr/local/cuda-12.6'), Path('/usr/local/cuda-12.8')]
            toolkit = next((str(p) for p in candidates if (p / 'bin/nvcc').exists()), None)
        assert toolkit, 'Set CUDA_HOME to an installed CUDA 12.x development toolkit'
        nvcc = capture([Path(toolkit) / 'bin/nvcc', '--version'])
        assert re.search(r'release 12\.', nvcc), 'Torch CUDA 12.6 requires a CUDA 12.x build toolkit'
        os.environ['CUDA_HOME'] = toolkit
        os.environ['PATH'] = str(Path(toolkit) / 'bin') + os.pathsep + os.environ['PATH']
        os.environ.setdefault('CXX', 'g++')
        os.environ.setdefault('CC', 'gcc')
        if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
            os.environ['TORCH_CUDA_ARCH_LIST'] = capture([source / 'bin/python', '-c',
            "import torch; assert torch.cuda.is_available(); print(';'.join(sorted({'.'.join(map(str,torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())})))"])
        state['toolchain'] = {'nvcc': nvcc, 'cuda_home': toolkit,
            'cxx': capture([os.environ['CXX'], '--version']).splitlines()[0],
            'arch': os.environ['TORCH_CUDA_ARCH_LIST'], 'max_jobs': os.environ.get('MAX_JOBS', '2'),
            'cxx_command': os.environ['CXX'], 'cc_command': os.environ['CC'],
            'torch_cxx11_abi': capture([source / 'bin/python', '-c', 'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)'])}
        recipe = hashlib.sha256()
        for path in [Path(__file__), ROOT / 'environments/setup_3d_environment.sh',
                     *[ROOT / 'environments/requirements' / name for name in (
                         '3d-sources.json', '3d-cu126-py312.requirements.txt',
                         'core-cu126-py312.constraints.txt', 'attention-cu126-py312.requirements.txt')]]:
            if path.is_file(): recipe.update(path.read_bytes())
        recipe.update(json.dumps([baseline, state['toolchain']], sort_keys=True).encode())
        identity = recipe.hexdigest()
        state['recipe'] = identity

        def run(name, commands, cwd=ROOT):
            nonlocal stage
            stage = name
            state['stage'] = name
            state['stages'][name] = {'status': 'running', 'log': str(OUT / f'{name}.log')}
            write(OUT / 'summary.json', state)
            print(f'{name}: running', flush=True)
            with (OUT / f'{name}.log').open('w') as log:
                for command in commands:
                    run_source_command(command, cwd=cwd, log=log)
            state['stages'][name]['status'] = 'passed'
            write(OUT / 'summary.json', state)
            print(f'{name}: passed', flush=True)

        ownership = OUT / 'clone-owner.json'
        if target.exists():
            assert ownership.exists(), 'Refusing an unmanaged smd_3d_extend environment'
            owner = json.loads(ownership.read_text())
            assert owner['target'] == str(target) and owner['source'] == str(source), 'Clone ownership mismatch'
            assert (target / 'conda-meta/history').exists() and python.exists(), 'Incomplete clone: inspect clone.log before removing this failed clone and rerunning'
        else:
            write(ownership, {'target': str(target), 'source': str(source)})
            run('clone', [[args.conda, 'create', '--yes', '--prefix', target, '--clone', source]])
        current = json.loads(capture([python, '-c', INVENTORY]))
        for name, version in baseline['packages'].items():
            assert current['packages'].get(name) == version, f'Clone changed base package: {name}'
        pip = [python, '-m', 'pip', '--isolated']
        # Runtime dependencies resolve under base constraints. Source packages use --no-deps
        # so o_voxel cannot fetch moving CuMesh/FlexGEMM branches from its metadata.
        run('dependencies', [pip + ['install', '--index-url', 'https://pypi.org/simple',
            '-c', ROOT / 'environments/requirements/core-cu126-py312.constraints.txt',
            '-r', ROOT / 'environments/requirements/3d-cu126-py312.requirements.txt',
            '--report', OUT / 'dependencies.json']])
        sources = json.loads((ROOT / 'environments/requirements/3d-sources.json').read_text())
        for spec in sources:
            name = spec['name']
            receipt = OUT / 'receipts' / f'{name}.json'
            installed = json.loads(capture([python, '-c', INVENTORY]))['packages']
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old['recipe'] == identity and installed.get(name) == old['version']:
                    state['stages'][name] = {'status': 'reused', 'receipt': str(receipt)}
                    print(f'{name}: reused', flush=True)
                    continue
            checkout = OUT / 'sources' / name
            commands = []
            if not (checkout / '.git').exists():
                commands += [['git', 'init', checkout], ['git', '-C', checkout, 'remote', 'add', 'origin', spec['url']]]
            commands += [['git', '-C', checkout, 'fetch', '--depth', '1', 'origin', spec['commit']],
                         ['git', '-C', checkout, 'checkout', '--detach', spec['commit']],
                         ['git', '-C', checkout, 'submodule', 'update', '--init', '--recursive', '--depth', '1']]
            run(name + '_source', commands)
            assert capture(['git', '-C', checkout, 'rev-parse', 'HEAD']) == spec['commit']
            run(name, [pip + ['install', '--no-deps', '--no-build-isolation', '--force-reinstall',
                             checkout / spec['subdir']]])
            version = json.loads(capture([python, '-c', INVENTORY]))['packages'][name]
            write(receipt, {'recipe': identity, 'source': spec, 'version': version,
                           'submodules': capture(['git', '-C', checkout, 'submodule', 'status', '--recursive'])})
        run('pip_check', [pip + ['check']])
        final = json.loads(capture([python, '-c', INVENTORY]))
        for name, version in baseline['packages'].items():
            assert final['packages'].get(name) == version, f'Extension install changed base dependency: {name}'
        assert json.loads(capture([source / 'bin/python', '-c', INVENTORY])) == baseline, 'Source environment changed'
        write(OUT / 'installed.json', final)
        run('imports', [[python, '-c', "import flash_attn,flex_gemm,nvdiffrast.torch,nvdiffrec_render,cumesh,o_voxel,utils3d; print('3D imports OK')"]])
        state.update(status='installed_pending_runtime', stage='complete',
                     next='bash environments/checks/validate_3d_environment.sh')
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='failed', stage=stage, error=f'{type(error).__name__}: {error}')
        print(f'{stage}: failed; {state["error"]}\nSee {OUT / (stage + ".log")}', flush=True)
        return 1
    finally:
        write(OUT / 'summary.json', state)
        print(f'summary: {OUT / "summary.json"}', flush=True)
    print('Installed. CUDA runtime and real model validation are still pending.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
