"""Fresh small runtime checks; no downloads or pretrained-model training/inference."""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def distributed_checks(output):
    import torch
    import torch.distributed as dist
    rank = int(os.environ['LOCAL_RANK'])
    assert torch.cuda.device_count() >= 2, 'Two visible CUDA GPUs are required'
    torch.cuda.set_device(rank)
    dist.init_process_group('cpu:gloo,cuda:nccl', timeout=timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 2
        for device in ('cpu', f'cuda:{rank}'):
            value = torch.tensor([dist.get_rank() + 1.], device=device)
            dist.all_reduce(value)
            assert value.item() == 3., f'{device}: incorrect all_reduce result'
        # Both ranks must finish both checks before rank zero can report success.
        dist.barrier(device_ids=[rank])
        if dist.get_rank() == 0:
            write_json(output, dict(status='passed', world_size=2,
                                   collectives={'gloo_cpu': 3., 'nccl_cuda': 3.},
                                   scope='Real two-GPU communication; no FSDP/model training'))
    finally:
        dist.destroy_process_group()
    return 0


def run_stage(command, log, timeout):
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=ROOT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            # Kill this stage's torchrun workers too; never touch other GPU jobs.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def validate(report_dir, timeout):
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another environment check is running in this report directory')
        attempt = Path(tempfile.mkdtemp(prefix='run-', dir=report_dir))
        summary_path = report_dir / 'summary.json'
        sources = [
            'environments/checks/validate_core_environment.sh', 'environments/checks/validate_core_environment.py',
            'environments/checks/check_environment.py', 'environments/checks/check_flash_attention.py',
            'environments/checks/check_runtime_dependencies.py',
        ]
        summary = dict(status='running', python=sys.executable, prefix=sys.prefix,
                       fresh_run=True, attempt=str(attempt), stages=[],
                       scope='Small runtime checks only; excludes pretrained-model runs and 3D extensions',
                       source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in sources})

        def save():
            write_json(attempt / 'summary.json', summary)
            write_json(summary_path, summary)

        stages = [
            ('runtime', [sys.executable, '-m', 'environments.checks.check_environment',
                         '--expected-prefix', sys.prefix, '--cuda']),
            ('two_gpu_collectives', [sys.executable, '-m', 'torch.distributed.run',
                                     '--standalone', '--nproc_per_node=2', '--module',
                                     'environments.checks.validate_core_environment', '--worker', 'distributed']),
        ]
        print('Fresh runtime checks: FlashAttention, preprocessing/media, two-GPU collectives.', flush=True)
        print('No downloads or pretrained-model training/inference. Full logs stay on disk.', flush=True)
        save()
        for name, command in stages:
            output, log = attempt / f'{name}.json', attempt / f'{name}.log'
            command += ['--output', str(output)]
            stage = dict(name=name, status='running', evidence=str(output), log=str(log), command=command)
            summary['stages'].append(stage)
            save()
            print(f'{name}: running', flush=True)
            try:
                code = run_stage(command, log, timeout)
                stage['exit_code'] = code
                payload = json.loads(output.read_text()) if output.exists() else {}
                passed = code == 0 and payload.get('status') == 'passed'
                stage['status'] = 'passed' if passed else 'failed'
                if not passed:
                    stage['error'] = '; '.join(
                        item.get('error', item['name']) for item in payload.get('checks', [])
                        if item['status'] != 'passed'
                    )[:1200] or f'Worker exit={code}; inspect stage evidence/log'
            except KeyboardInterrupt:
                stage.update(status='interrupted', error='Interrupted by user or termination signal')
            except Exception as error:
                stage.update(status='failed', error=f'{type(error).__name__}: {error}'[:1200])
            print(f"{name}: {stage['status']}", flush=True)
            if stage['status'] != 'passed':
                summary['status'] = stage['status']
                save()
                print(f'summary: {summary_path}', flush=True)
                return 1
            save()
        summary['status'] = 'passed'
        save()
        print(f'summary: {summary_path}', flush=True)
        print('Small runtime checks passed. Pretrained-model validation remains separate.', flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'LOCAL_DEV/manual_environment')
    parser.add_argument('--timeout', type=int, default=600, help='Maximum seconds per stage')
    parser.add_argument('--worker', choices=['distributed'], help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        if args.output is None:
            parser.error('--worker requires --output')
        return distributed_checks(args.output)
    if Path(sys.prefix).name != 'smd' or not (Path(sys.prefix) / 'conda-meta').is_dir():
        parser.error('Activate the clean Conda environment: conda activate smd')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    return validate(args.report_dir.resolve(), args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
