# SPDX-License-Identifier: Apache-2.0
"""Restore pinned preview assets, specialist teachers and initialization weights."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import sys
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def checked_target(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in relative:
        raise ValueError(f"Invalid resource path: {relative}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Resource path escapes its destination: {relative}")
    return target


def matches(path, spec):
    return path.is_file() and path.stat().st_size == spec["bytes"] and digest(path) == spec["sha256"]


def download_plan(models, teachers=(), *, scope="runtime", root=ROOT, model_root=None):
    """Resolve aliases and dependencies without importing training configurations."""
    if scope not in {"runtime", "rebuild"}:
        raise ValueError(f"Unknown resource scope: {scope}")
    root = Path(root).expanduser().resolve()
    model_root = Path(model_root or os.environ.get("SMD_MODEL_ROOT", root / "models")).expanduser().resolve()
    manifest = json.loads((DATA / "resource_manifest.json").read_text())
    lock = json.loads((DATA / "resource_lock.json").read_text())
    families = manifest["families"]
    aliases = {alias: family for family, group in families.items() for alias in group["experiments"]}
    selected = set()
    selected_teachers = set(teachers)
    for model in models:
        if model == "all":
            selected.update(families)
            selected_teachers.update(lock["teachers"])
        else:
            family = aliases.get(model, model)
            if family not in families:
                raise ValueError(f"Unknown resource model: {model}")
            selected.add(family)
            selected_teachers.update(lock["experiment_teachers"].get(model, []))
    if "all" in selected_teachers:
        selected_teachers.remove("all")
        selected_teachers.update(lock["teachers"])
    unknown = selected_teachers - lock["teachers"].keys()
    if unknown:
        raise ValueError(f"Unknown teachers: {', '.join(sorted(unknown))}")
    plan = {}
    source = lock["resources"]
    for family in sorted(selected):
        for relative in families[family][scope]:
            target = checked_target(root, relative)
            plan[target] = dict(
                root=root,
                relative=relative,
                spec=manifest["files"][relative],
                source=dict(
                    repo_id=source["repo_id"],
                    repo_type=source["repo_type"],
                    revision=source["revision"],
                    filename=relative,
                ),
            )
    for teacher in sorted(selected_teachers):
        source = lock["teachers"][teacher]
        for filename, spec in source["files"].items():
            relative = f"sd35_teachers/{teacher}/{filename}"
            target = checked_target(model_root, relative)
            remote = "/".join(p for p in (source.get("subfolder", ""), filename) if p)
            plan[target] = dict(
                root=model_root,
                relative=relative,
                spec=spec,
                source=dict(repo_id=source["repo_id"], repo_type="model", revision=source["revision"], filename=remote),
            )
    for family in sorted(selected):
        for name in lock.get("family_checkpoints", {}).get(family, []):
            source = lock["checkpoints"][name]
            for relative, spec in source["files"].items():
                target = checked_target(model_root, relative)
                plan[target] = dict(
                    root=model_root,
                    relative=relative,
                    spec=spec,
                    source=dict(
                        repo_id=source["repo_id"], repo_type="model", revision=source["revision"],
                        filename=spec["filename"],
                    ),
                )
    return plan


def prepare_resources(
    models=(), *, teachers=(), scope="runtime", root=ROOT, model_root=None, cache_dir=None, check_only=False
):
    plan = download_plan(models, teachers, scope=scope, root=root, model_root=model_root)
    if not plan:
        raise ValueError("Select at least one model or teacher")
    root = Path(root).expanduser().resolve()
    # Serialize preparation across torchrun ranks. Locks and staging files must
    # stay outside the H3 fixture trees, whose complete contents enter signatures.
    with ExitStack() as stack:
        if not check_only:
            for destination in sorted({item["root"] for item in plan.values()}):
                directory = destination / "LOCAL_DEV" / "resource_download"
                directory.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context((directory / "download.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX)
        missing = []
        for target, item in plan.items():
            if target.exists() or target.is_symlink():
                if not matches(target, item["spec"]):
                    raise RuntimeError(f"Existing resource differs; preserve or move it before retrying: {target}")
            else:
                missing.append((target, item))
        if missing and check_only:
            paths = "\n".join(str(target) for target, _ in missing[:8])
            raise FileNotFoundError(
                f"Missing {len(missing)} resource files. Run tools/download_resources.py first:\n{paths}"
            )
        if missing:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_dir = Path(cache_dir or HF_HUB_CACHE).expanduser().resolve()
            if cache_dir.is_relative_to(root / "assets"):
                raise ValueError("The Hugging Face download cache must be outside assets/")
            for target, item in missing:
                downloaded = Path(hf_hub_download(**item["source"], cache_dir=str(cache_dir)))
                if not matches(downloaded, item["spec"]):
                    raise RuntimeError(f"Downloaded resource checksum mismatch: {item['relative']}")
                staging = item["root"] / "LOCAL_DEV" / "resource_download"
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=staging, delete=False) as output:
                        temporary = Path(output.name)
                        with downloaded.open("rb") as source:
                            shutil.copyfileobj(source, output)
                    if not matches(temporary, item["spec"]):
                        raise RuntimeError(f"Copied resource checksum mismatch: {item['relative']}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        if not matches(target, item["spec"]):
                            raise RuntimeError(f"Concurrent conflicting resource: {target}")
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
    return dict(files=len(plan), bytes=sum(item["spec"]["bytes"] for item in plan.values()), installed=len(missing))


def prepare_for_config(config_path, *, model=None):
    """Prepare before create_config(), including for a custom config with an explicit model."""
    if model is None:
        path = Path(config_path).resolve()
        if path.parent != ROOT / "smd" / "configs" / "experiments" or path.suffix != ".py":
            raise ValueError("For a custom config, also supply --resource-model EXPERIMENT")
        model = path.stem
    return prepare_resources([model])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", default=[], help="Experiment or family names, or all")
    parser.add_argument("--teachers", nargs="+", default=[], help="Additional teacher names, or all")
    parser.add_argument("--scope", choices=["runtime", "rebuild"], default="runtime")
    parser.add_argument("--root", type=Path, default=ROOT, help="SMD checkout to populate")
    parser.add_argument(
        "--model-root", type=Path, help="Extra weight destination; defaults to SMD_MODEL_ROOT or ROOT/models"
    )
    parser.add_argument("--cache-dir", type=Path, help="Hub cache outside assets/")
    parser.add_argument("--check-only", action="store_true", help="Verify local files without downloading")
    parser.add_argument("--list", action="store_true", help="List the selected files without downloading")
    args = parser.parse_args()
    if not args.model and not args.teachers:
        parser.error("Select --model or --teachers")
    options = dict(teachers=args.teachers, scope=args.scope, root=args.root, model_root=args.model_root)
    if args.list:
        for target, item in sorted(download_plan(args.model, **options).items()):
            print(f"{item['spec']['bytes']}\t{target}")
        return
    result = prepare_resources(args.model, **options, cache_dir=args.cache_dir, check_only=args.check_only)
    print(f"Verified {result['files']} files ({result['bytes']} bytes); installed {result['installed']}.")


if __name__ == "__main__":
    main()
